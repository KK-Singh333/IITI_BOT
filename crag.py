# Copyright © 2024 Pathway
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from warnings import warn
import re
import requests

import pathway as pw
from pathway.xpacks.llm.llm_agents import CallAgent,web_search_tool,QueryBreaker,Refiner,FinalLayer
from pathway.internals import ColumnReference, Table, udfs
from pathway.stdlib.indexing import DataIndex
from pathway.xpacks.llm import Doc, llms, prompts
from pathway.xpacks.llm.document_store import (
    DocumentStore,
    SlidesDocumentStore,
    _get_jmespath_filter,
)
from pathway.xpacks.llm.llms import BaseChat, prompt_chat_single_qa
from pathway.xpacks.llm.vector_store import (
    SlidesVectorStoreServer,
    VectorStoreClient,
    VectorStoreServer,
)

if TYPE_CHECKING:
    from pathway.xpacks.llm.servers import QARestServer, QASummaryRestServer


@pw.udf
def _limit_documents(documents: list[pw.Json], k: int) -> list[pw.Json]:
    return documents[:k]


_answer_not_known = "I could not find an answer."
_answer_not_known_open_source = "No information available."


class BaseContextProcessor(ABC):
    """Base class for formatting documents to LLM context.

    Abstract method ``docs_to_context`` defines the behavior for converting documents to context.
    """

    def maybe_unwrap_docs(self, docs: pw.Json | list[pw.Json] | list[Doc]):
        if isinstance(docs, pw.Json):
            doc_ls: list[Doc] = docs.as_list()
        elif isinstance(docs, list) and all([isinstance(dc, dict) for dc in docs]):
            doc_ls = docs  # type: ignore
        elif all([isinstance(doc, pw.Json) for doc in docs]):
            doc_ls = [doc.as_dict() for doc in docs]  # type: ignore
        else:
            raise ValueError(
                """`docs` argument is not instance of (pw.Json | list[pw.Json] | list[Doc]).
                            Please check your pipeline. Using `pw.reducers.tuple` may help."""
            )

        if len(doc_ls) == 1 and isinstance(doc_ls[0], list | tuple):  # unpack if needed
            doc_ls = doc_ls[0]

        return doc_ls

    def apply(self, docs: pw.Json | list[pw.Json] | list[Doc]) -> str:
        unwrapped_docs = self.maybe_unwrap_docs(docs)
        return self.docs_to_context(unwrapped_docs)

    @abstractmethod
    def docs_to_context(self, docs: list[dict] | list[Doc]) -> str: ...

    def as_udf(self) -> pw.UDF:
        return pw.udf(self.apply)


@dataclass
class SimpleContextProcessor(BaseContextProcessor):
    """Context processor that filters metadata fields and joins the documents."""

    context_metadata_keys: list[str] = field(default_factory=lambda: ["path"])
    context_joiner: str = "\n\n"

    def simplify_context_metadata(self, docs: list[Doc]) -> list[Doc]:
        filtered_docs = []
        for doc in docs:
            filtered_doc = {"text": doc["text"]}
            doc_metadata: dict = doc.get("metadata", {})  # type: ignore

            for key in self.context_metadata_keys:

                if key in doc_metadata:
                    filtered_doc[key] = doc_metadata[key]

            filtered_docs.append(filtered_doc)

        return filtered_docs

    def docs_to_context(self, docs: list[dict] | list[Doc]) -> str:
        docs = self.simplify_context_metadata(docs)

        context = self.context_joiner.join(
            [json.dumps(doc, ensure_ascii=False) for doc in docs]
        )

        return context


@pw.udf
def _prepare_RAG_response(
    response: str, docs: list[dict], return_context_docs: bool
) -> pw.Json:
    api_response: dict = {"response": response}
    if return_context_docs:
        api_response["context_docs"] = docs

    return pw.Json(api_response)


def _get_RAG_prompt_udf(prompt_template: str | Callable[[str, str], str] | pw.UDF):
    if isinstance(prompt_template, pw.UDF) or callable(prompt_template):
        verified_template: prompts.BasePromptTemplate = (
            prompts.RAGFunctionPromptTemplate(function_template=prompt_template)
        )
    elif isinstance(prompt_template, str):
        verified_template = prompts.RAGPromptTemplate(template=prompt_template)
    else:
        raise ValueError(
            f"Prompt template is not of expected type. Got: {type(prompt_template)}."
        )

    return verified_template.as_udf()


def _get_context_processor_udf(
    context_processor: (
        BaseContextProcessor | Callable[[list[dict] | list[Doc]], str] | pw.UDF
    )
) -> pw.UDF:
    if isinstance(context_processor, BaseContextProcessor):
        return context_processor.as_udf()
    elif isinstance(context_processor, pw.UDF):
        return context_processor
    elif callable(context_processor):
        return pw.udf(context_processor)
    else:
        raise ValueError(
            "Context processor must be type of one of the following: \
            ~BaseContextProcessor | Callable[[list[dict] | list[Doc]], str] | ~pw.UDF"
        )


def _query_chat(
    chat: BaseChat,
    t: Table,
    prompt_udf: pw.UDF,
    no_answer_string: str,
    context_processor: pw.UDF,
) -> pw.Table:
    t = t.with_columns(context=context_processor(t.documents))
    t += t.select(prompt=prompt_udf(t.context, t.query))
    answer = t.select(answer=chat(prompt_chat_single_qa(t.prompt)))

    answer = answer.select(
        answer=pw.if_else(pw.this.answer == no_answer_string, None, pw.this.answer)
    )
    return answer


def _query_chat_with_k_documents(
    chat: BaseChat,
    k: int,
    t: pw.Table,
    prompt_udf: pw.UDF,
    no_answer_string: str,
    context_processor: pw.UDF,
) -> pw.Table:
    limited_documents = t.select(
        pw.this.query, documents=_limit_documents(t.documents, k)
    )
    result = _query_chat(
        chat, limited_documents, prompt_udf, no_answer_string, context_processor
    ).with_columns(documents=limited_documents.documents)
    return result


def answer_with_geometric_rag_strategy(
    questions: ColumnReference,
    documents: ColumnReference,
    llm_chat_model: BaseChat,
    *,
    prompt_template: (
        str | Callable[[str, str], str] | pw.UDF
    ) = prompts.prompt_qa_geometric_rag,
    no_answer_string: str = "No information found.",
    context_processor: (
        BaseContextProcessor | Callable[[list[dict] | list[Doc]], str] | pw.UDF
    ) = SimpleContextProcessor(),
    n_starting_documents: int = 2,
    factor: int = 2,
    max_iterations: int = 4,
    return_context_docs: pw.ColumnExpression | bool = False,
) -> ColumnReference:
    

    @pw.udf
    def make_json(docs: list[str]) -> list[dict]:
        return [{"text": doc} for doc in docs]

    prompt_udf = _get_RAG_prompt_udf(prompt_template)
    n_documents = n_starting_documents
    t = Table.from_columns(query=questions, documents=documents)
    t = t.with_columns(
        answer=None,
        documents_used=None,
        return_context_docs=return_context_docs,
        documents=make_json(documents),
    )
    context_processor = _get_context_processor_udf(context_processor)
    for _ in range(max_iterations):
        rows_without_answer = t.filter(pw.this.answer.is_none())
        results = _query_chat_with_k_documents(
            llm_chat_model,
            n_documents,
            rows_without_answer,
            prompt_udf,
            no_answer_string,
            context_processor=context_processor,
        )
        new_answers = rows_without_answer.with_columns(
            answer=results.answer, documents_used=results.documents
        )
        t = t.update_rows(new_answers)
        n_documents *= factor

    t = t.select(
        answer=pw.if_else(
            pw.this.answer.is_none(),
            _prepare_RAG_response(
                no_answer_string, pw.this.documents, pw.this.return_context_docs
            ),
            _prepare_RAG_response(
                pw.this.answer, pw.this.documents_used, pw.this.return_context_docs
            ),
        )
    )

    return t.answer


def answer_with_geometric_rag_strategy_from_index(
    questions: ColumnReference,
    index: DataIndex,
    documents_column: str | ColumnReference,
    llm_chat_model: BaseChat,
    *,
    prompt_template: (
        str | Callable[[str, str], str] | pw.UDF
    ) = prompts.prompt_qa_geometric_rag,
    no_answer_string: str = "No information found.",
    context_processor: (
        BaseContextProcessor | Callable[[list[dict] | list[Doc]], str] | pw.UDF
    ) = SimpleContextProcessor(),
    n_starting_documents: int = 2,
    factor: int = 2,
    max_iterations: int = 4,
    return_context_docs: pw.ColumnExpression | bool = False,
    metadata_filter: pw.ColumnExpression | None = None,
) -> ColumnReference:
   
    max_documents = n_starting_documents * (factor ** (max_iterations - 1))

    if isinstance(documents_column, ColumnReference):
        documents_column_name = documents_column.name
    else:
        documents_column_name = documents_column

    query_context = questions.table + index.query_as_of_now(
        questions,
        number_of_matches=max_documents,
        collapse_rows=True,
        metadata_filter=_get_jmespath_filter(metadata_filter, ""),
    ).select(
        documents_list=pw.coalesce(pw.this[documents_column_name], ()),
    )

    return answer_with_geometric_rag_strategy(
        questions,
        query_context.documents_list,
        llm_chat_model,
        n_starting_documents=n_starting_documents,
        factor=factor,
        max_iterations=max_iterations,
        return_context_docs=return_context_docs,
        prompt_template=prompt_template,
        no_answer_string=no_answer_string,
        context_processor=context_processor,
    )


class BaseQuestionAnswerer:
    AnswerQuerySchema: type[pw.Schema] = pw.Schema
    RetrieveQuerySchema: type[pw.Schema] = pw.Schema
    StatisticsQuerySchema: type[pw.Schema] = pw.Schema
    InputsQuerySchema: type[pw.Schema] = pw.Schema

    @abstractmethod
    def answer_query(self, pw_ai_queries: pw.Table) -> pw.Table: ...

    @abstractmethod
    def retrieve(self, retrieve_queries: pw.Table) -> pw.Table: ...

    @abstractmethod
    def statistics(self, statistics_queries: pw.Table) -> pw.Table: ...

    @abstractmethod
    def list_documents(self, list_documents_queries: pw.Table) -> pw.Table: ...


class SummaryQuestionAnswerer(BaseQuestionAnswerer):
    SummarizeQuerySchema: type[pw.Schema] = pw.Schema

    @abstractmethod
    def summarize_query(self, summarize_queries: pw.Table) -> pw.Table: ...

class CRAG(SummaryQuestionAnswerer):
    def __init__(
        self,
        indexer: VectorStoreServer | DocumentStore,
    ) -> None:

        self.indexer=indexer
        

    
    @pw.table_transformer
    def answer_query(self, pw_ai_queries: pw.Table,pw_history_table:pw.Table) -> pw.Table:
        """Answer a question based on the available information."""
        #pw_ai_queries will be connected to a mongodB database, format: QueryId |  User ID  | Query (str)
        #pw_history_table will contain a past history of user chats, format: User ID | History(pw.Json)
        pw_ai_queries+=pw_ai_queries.join_left(pw_history_table,pw_ai_queries.userid==pw_history_table.userid)
        pw_ai_queries+=pw_ai_queries.select(context=self.provide_context(pw.this.query,pw.this.docs))
        pw_ai_queries+=pw_ai_queries.select(subqueries=self.create_subquery(pw.this.query))
        pw_ai_queries+=pw_ai_queries.select(refined_context=self.refine_document(pw.this.context))
        pw_ai_queries+=pw_ai_queries.select(response=self.final_response(pw.this.subqueries,pw.this.refined_context))

        return pw_ai_queries
    #    QueryID| UserID | Response
    @pw.udf
    def provide_context(self,question: str, docs: pw.Json | list[str],depth: int = 0, max_depth: int = 1)-> str:
        '''Main Implementation of CRAG System'''
        create_splitter=CallAgent('SPLITTER',question,docs)
        create_rewriter=CallAgent('REWRITER',question,docs)
        retrieval_splitter=create_splitter()
        question_rewriter=create_rewriter()
        context=''
        def web_content(question:str):
            results = web_search_tool.invoke(question)

            content_list = [result['content'] for result in results if 'content' in result]
            full_content = "\n".join(content_list)

            return full_content
        for doc in docs:
            if depth > max_depth:
                    return []

            split = retrieval_splitter.invoke({'document': doc, 'question': question})
            high_conf_lst = split.split_doc_high_confidence
            low_conf_lst = split.split_doc_low_confidence

            for chunk in low_conf_lst:
                try:
                    print('Initiated web search')
                    web_question = question_rewriter.invoke({'question': question, 'doc': chunk}).content
                    content = web_content(web_question)
                    high_conf_lst.extend(retrieval_splitter.invoke({'question':question,'document':content}).split_doc_high_confidence)
                except Exception as e:
                    print(e)
                    continue
            cleaned_docs = set()

            for doc in high_conf_lst:
                # Fix missing spaces between numbers and words
                doc = re.sub(r"([a-zA-Z])(\d)', r'\1 \2", doc)
                doc = re.sub(r"(\d)([a-zA-Z])', r'\1 \2", doc)

                # Remove extra whitespace
                doc = doc.strip()

                # Deduplicate using a set
                cleaned_docs.add(doc)
            final_summary=" ".join(cleaned_docs)
            context=context+'\n'+final_summary
        return context
    @pw.udf
    def create_subquery(self,query:str)->str:
        breaker=QueryBreaker()
        return breaker(query)
    @pw.udf
    def refine_document(self,document:str)->str:
        refiner=Refiner()
        return refiner(document)
    @pw.udf
    def final_response(self,subqueries: str,context: str)-> str:
        answerer=FinalLayer()
        return answerer(subqueries,context)




    
    
    @pw.table_transformer
    def retrieve(self, retrieve_queries: pw.Table) -> pw.Table:
        """
        Retrieve documents from the index.
        """
        return self.indexer.retrieve_query(retrieve_queries)

   