import pathway as pw
from pathway.xpacks.llm import llms, embedders, rerankers
import os
from pathway.udfs import ExponentialBackoffRetryStrategy
import numpy as np
import ast
from pymongo import MongoClient
from pathway.stdlib.ml.index import KNNIndex
import litellm

from langchain_community.tools.tavily_search import TavilySearchResults

mongo=MongoClient("mongodb://mongodb:27017")
database = mongo['IITI_BOT']
users= database['users']

model="groq/meta-llama/llama-4-scout-17b-16e-instruct"

system_prompt_retriever = """
You are an AI language model assistant.
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these five alternative questions separated by newlines.
Output only the generated queries not including any other text.
"""

# Doc Store
path =r"/data/result_2.csv"



class InputCSVDataSchema(pw.Schema):
    row_id: str
    chunk: str
    embedding: str

@pw.udf
def split_lines(text: str) -> list[str]:
    return text.splitlines()

@pw.reducers.stateful_many
def unique_docs(state: list | None, rows) -> list:

  state = [[],[]]
  for row, cnt in rows:
    doc_ids, docs = row
    for i, doc_id in enumerate(doc_ids):
      if doc_id not in state[0]:
        state[0].append(doc_id)
        state[1].append(docs[i])
  return state


def web_content(question:str):
    web_search_tool = TavilySearchResults(k=2)
    results = web_search_tool.invoke(question)
    content_list = [result['content'] for result in results]
   
    return content_list

@pw.reducers.stateful_many
def CRAG_good_docs(state: list | None, rows) -> list:
  if state == None:
    state = [[],[]]
  for row, cnt in rows:
    queries, doc, rank = row
    if rank > 2:
      state[0].append(doc)
      state[1].append(rank)
  if len(state[1]) < 3:
    state[0] += web_content(rows[0][0][0])
  return state[0]
@pw.reducers.stateful_many
def create_history(state: list | None, rows):
    if state is None:
      state = []
    for row, cnt in rows:
      query=row[0]
      result=row[1]
      state.append({'role': 'user','content':query})
      state.append({'role': 'assistant','content':result})
    return state
@pw.udf
def retrieve_history(email:str,chat_id:int)->list[dict]:
    user_history = users.find_one({"email": email})
    if not user_history:
        return []
    chat_history = user_history.get('chats', [])
    return chat_history[chat_id] if chat_id < len(chat_history) else []
@pw.udf
def update_history(email: str, chat_id: int, query: str, result: str) -> int:
    user = users.find_one({"email": email})
    
    
    if not user:
        return 0

    chats = user.get('chats', [])

    
    while len(chats) <= chat_id:
        chats.append([])

    
    chats[chat_id].append({'role': 'user', 'content': query})
    chats[chat_id].append({'role': 'assistant', 'content': result})

    
    users.update_one(
        {"email": email},
        {"$set": {"chats": chats}}
    )
    return 0
    
class Retriever():

  def __init__(self, model:str = model, system_prompt:str = system_prompt_retriever, path_csv:str = path):
    self.llm = llms.LiteLLMChat(model=model, retry_strategy=ExponentialBackoffRetryStrategy(max_retries=2))
    self.system_prompt = system_prompt
    self.embedder = embedders.SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    self.reranker = rerankers.LLMReranker(llm=llms.LiteLLMChat(model=model, retry_strategy=ExponentialBackoffRetryStrategy(max_retries=2), response_format={'type': 'json_object'}))
    self.csv_data = pw.io.csv.read(
    path_csv,
    schema=InputCSVDataSchema,
    mode="streaming"
    )
    def parse_nested_embedding(embedding_str):
      try:
          parsed = ast.literal_eval(embedding_str)
          embedding_vector = parsed
          return np.array(embedding_vector, dtype=np.float32)
      except Exception as e:
          print(f"Error parsing embedding: {e}")
          return None

    self.vector_store = self.csv_data.select(
    doc_id=pw.this.row_id,
    chunks=pw.this.chunk,
    embedding=pw.apply(parse_nested_embedding, self.csv_data.embedding),
    )



  @pw.table_transformer
  def equivalent_queries(self, queries:pw.Table):

    @pw.udf
    def query_parser(args) -> list[dict]:
      return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": f"{args}"}]

    query_table = queries.select(user_id = pw.this.user_id, questions = query_parser(pw.this.queries))
    responses = query_table.select(user_id = pw.this.user_id, result = self.llm(pw.this.questions, temperature=0.0))

    split_table = responses.select(
    user_id=pw.this.user_id,
    questions = split_lines(pw.this.result) 
    )
    response = split_table.flatten(pw.this.questions)
    return response.filter(pw.this.questions != "")
  
  @pw.table_transformer
  def __call__(self, queries:pw.Table):
    
    
    response = self.equivalent_queries(queries)
    response += response.select(embedding=self.embedder(pw.this.questions))
    
    doc_index = KNNIndex(
    self.vector_store.embedding,
    self.vector_store,
    n_dimensions= self.embedder.get_embedding_dimension(), 
    distance_type = "cosine"
    )

    results = doc_index.get_nearest_items(
    response.embedding,
    k=3
    ).select(user_id = response.user_id, doc_id = pw.this.doc_id, chunks = pw.this.chunks)
    
    results = results.groupby(pw.this.user_id,id= pw.this.user_id).reduce(pw.this.user_id, chunks = unique_docs(pw.this.doc_id, pw.this.chunks)[1])
    results = results.join(queries, pw.this.user_id == queries.user_id,id=pw.this.id).select(user_id = results.user_id, queries = queries.queries, chunks = results.chunks,chat_id = queries.chat_id,email = queries.email,user_query= queries.queries)
    results_flatten = results.flatten(pw.this.chunks)
    results_flatten += results_flatten.select(rank=self.reranker(pw.this.chunks, pw.this.queries))
    results = results_flatten.groupby(pw.this.user_id,id=pw.this.user_id).reduce(pw.this.user_id,chat_id=pw.reducers.any(pw.this.chat_id),email=pw.reducers.any(pw.this.email) ,queries = pw.reducers.any(pw.this.queries),user_query=pw.reducers.any(pw.this.user_query) ,chunks = CRAG_good_docs(pw.this.queries, pw.this.chunks, pw.this.rank))
    results+=results.select(history=retrieve_history(pw.this.email,pw.this.chat_id))
    
    @pw.udf
    def multiple_queries_parser(query:str, docs:tuple[str],completions: list[dict]) -> list[dict]:
      system_prompt = '''You are AI chat bot designed specifically for iit indore. You answer questions realted to iit indore on basis of the information provided in the documents.

The Final answer must never contain any words like 'Context does not provide enough information', in such a case simply information is not available in this regard

The final response must be:

Comprehensive: Cover all aspects asked in the query

Accurate: Stick strictly to the context; do not make up information

Useful: Present information clearly and helpfully

Only wherever necessary, you may organize the final response using headings and bullet points for clarity.

You are an expert assistant responding to a user.

You have access to a set of factual details (not visible to the user).

Given a set of procedural subqueries, answer them as naturally and helpfully as possible, using only the information provided.

Do not refer to the data as “context,” “input,” or “scraped.”

Do not say that something is “missing” from the data.

If certain specifics are unavailable, answer only what can be said confidently, and phrase it as a general insight.

The final response must be:

Helpful and natural

 Accurate to the available information

 Free from any mention of internal system details

You may suggest realistic next steps for the user (e.g., contacting faculty), but only if they match normal user-facing behavior.

 Never say: “The context does not mention...”, “The input says...”, or anything that reveals backend mechanics.
 Only provide the final response, clearly structured, as if in a natural conversation.
Output only the final response, not the subqueries or intermediate steps.

**If you feel that information given through context is not enough just simply say Information Not Available for Given Query. Don't make-up answers yourself.
'''

      return [{"role": "system", "content": system_prompt}]+list(completions)+[{'role':"assistant", "content": "\n".join(docs)} ,{"role": "user", "content": query}]
    questions = results.select(
        user_id=pw.this.user_id,
        queries = pw.this.queries,
        email=pw.this.email,
        chat_id=pw.this.chat_id,
        user_query=pw.this.user_query,
        prompt=multiple_queries_parser(pw.this.queries,pw.this.chunks,pw.this.history))
    answers = questions.select(user_id = pw.this.user_id,
                               queries = pw.this.queries,
                                email=pw.this.email,
                                chat_id=pw.this.chat_id,
                               user_query=pw.this.user_query,
                               answer = self.llm(pw.this.prompt, temperature=0.0))
    print(answers.schema)
    _=answers.select(k=update_history(pw.this.email,pw.this.chat_id,pw.this.user_query,pw.this.answer))
    pw.io.null.write(_)
    return answers
