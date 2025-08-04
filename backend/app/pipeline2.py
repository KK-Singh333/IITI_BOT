import pathway as pw
from pathway.xpacks.llm import llms, embedders, rerankers, prompts
import os
import json
from pathway.udfs import ExponentialBackoffRetryStrategy
import numpy as np
import ast
from pathway.stdlib.ml.index import KNNIndex
from pymongo import MongoClient
# from pathway.internals.json import Json
from langchain_community.tools.tavily_search import TavilySearchResults

# import time

mongo=MongoClient("mongodb://mongodb:27017")
database = mongo['IITI_BOT']
users= database['users']


# Setting Model
model="groq/meta-llama/llama-4-scout-17b-16e-instruct"

system_prompt_retriever = """
You are an AI language model assistant.
Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these three alternative questions separated by newlines.
Output only the generated queries not including any other text.
"""

# Doc Store
path = r"/data/result_2.csv"


@pw.udf
def prompt_web_search(query:str) -> str:
  prompt = f"""
  You are a query rewriter for a web-search assistant. Your task is to rephrase user queries to make them suitable for web search. Output only the rewritten query, optimized for extracting relevant information from web search results.
  If the query is too vague, ambiguous, or cannot yield meaningful results from a web search, respond with: "No web search"
  Only output the rewritten query or "No web search". Do not add explanations.
  Question: '{query}'
  """
  prompt += "Rewritten query:"
  return prompt

@pw.udf
def prompt_final_ans(query:str, docs:tuple, completions) -> list[dict]:
  system_prompt = """
  You are a helpful and intelligent assistant. Use the provided context to help you better understand and answer the user's question.
  Prefer answers that are informed by the context.
  If there are relevant details in the documents, only include knowledge of documents in your answer.
  Always aim for clarity, helpfulness, and accuracy in your answer.
  Output just the answer to user query.
  """
  
  print(list(completions))
  ans = [{"role": "system", "content": system_prompt}]+list(completions)+[{'role':"assistant", "content": "\n".join(docs)} ,{"role": "user", "content": query}]
  print(ans)
  return ans

@pw.udf
def extract_list(file:pw.Json) -> tuple:
  return file["docs"].as_list()

@pw.reducers.stateful_many
def CRAG_good_docs(state: dict | None, rows) -> pw.Json:
  if state == None:
    state = {
        "doc_id":[],
        "docs":[],
        "number": 0
    }
  else:
    state = state.as_dict()
    
  for row, cnt in rows:
    doc_ids, docs, ranks = row
    for i, doc_id in enumerate(doc_ids):
      if ranks[i] > 1 and doc_id not in state["doc_id"]:
        state["doc_id"].append(doc_id)
        state["docs"].append(docs[i])
        state["number"] += 1
  
  print("length is: ", len(state["doc_id"]), "State is: ", state)

  return pw.Json(state)

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

class InputCSVDataSchema(pw.Schema):
    # row_id: str
    chunk: str
    embedding: str
    # url: str

@pw.udf
def split_lines(text: str) -> list[str]:
    return text.splitlines()

@pw.udf
def web_content(question:str) -> tuple:
    print("Web Searching...")
    web_search_tool = TavilySearchResults(k=2)
    results = web_search_tool.invoke(question)
    content_list = tuple(result["content"] for result in results)
    return content_list




class Retriever():

  def __init__(self, model:str = model, system_prompt:str = system_prompt_retriever, path_csv:str = path):
    # Setting llm
    self.llm = llms.LiteLLMChat(model=model, retry_strategy=ExponentialBackoffRetryStrategy(max_retries=2))
    self.system_prompt = system_prompt
    self.embedder = embedders.SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    self.reranker = rerankers.LLMReranker(llm=llms.LiteLLMChat(model=model, retry_strategy=ExponentialBackoffRetryStrategy(max_retries=1), response_format={'type': 'json_object'}))
   
    self.number_of_web_search = 0
    self.csv_data = pw.io.csv.read(
    path_csv,
    schema=InputCSVDataSchema,
    mode="streaming"
    )
    def parse_nested_embedding(embedding_str):
      try:
          # Parse the string to get nested list
          parsed = ast.literal_eval(embedding_str)
          if isinstance(parsed, dict):
            return parsed["elements"]
          # Extract the first (inner) list
          # embedding_vector = parsed[0]
          # Convert to numpy array
          return np.array(parsed, dtype=np.float64)
      except Exception as e:
          print(f"Error parsing embedding: {e}")
          return None

    self.vector_store = self.csv_data.select(
    doc_id=pw.apply_with_type(lambda x: hash(x), int, pw.this.chunk),
    chunks=pw.this.chunk,
    # url=pw.this.url,
    embedding=pw.apply(parse_nested_embedding, self.csv_data.embedding),
    # Include other columns you need
    )

    self.doc_index = KNNIndex(
    self.vector_store.embedding,
    self.vector_store,
    n_dimensions= self.embedder.get_embedding_dimension(),  # dimension for all-MiniLM-L6-v2
    distance_type = "cosine"
    # n_and_d=2
    )

  def Custom_Reranker(self, question, docs):
    ranks = []
    for i in range(3):
      doc = docs.get(i) # Default is None
      rank = pw.if_else(doc.is_none(), None, self.reranker(doc, question))
      
      ranks.append(rank)
    
    return pw.make_tuple(*ranks)

  def web_search(self, questions) -> str | None:
    prompt = prompt_web_search(questions)
    question = self.llm(llms.prompt_chat_single_qa(prompt), temperature = 0.0)
    return pw.if_else(question == "No web search", None, web_content(question))


  @pw.table_transformer
  def equivalent_queries(self, queries:pw.Table):

    @pw.udf
    def query_parser(args) -> list[dict]:
      return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": f"{args}"}]

    response = queries.with_columns(questions = split_lines(self.llm(query_parser(pw.this.queries), temperature=0.0)))

    response = response.flatten(pw.this.questions).filter(pw.this.questions != " ")
    original_response = queries.with_columns(questions=pw.this.queries)
    # return original_response
    pw.universes.promise_are_pairwise_disjoint(response, original_response)
    response = response.concat(original_response)
    return response

  @pw.table_transformer
  def retrieve_docs(self, response:pw.Table):
    response = response.with_columns(embedding=self.embedder(pw.this.questions))
    results = self.doc_index.get_nearest_items(
    response.embedding,
    k=3,
    
    ).select(user_id = response.user_id,query = response.queries, questions = response.questions, chat_id = response.chat_id, email = response.email, doc_id = pw.this.doc_id, chunks = pw.this.chunks)
    
    results = results.with_columns(rank = self.Custom_Reranker(pw.this.questions, pw.this.chunks))

    return results

  @pw.table_transformer
  def CRAG(self, results:pw.Table):
    results = results.groupby(pw.this.user_id, id = pw.this.user_id).reduce(pw.this.user_id, query = pw.reducers.any(pw.this.query), chat_id = pw.reducers.any(pw.this.chat_id), email = pw.reducers.any(pw.this.email), docs = CRAG_good_docs(pw.this.doc_id, pw.this.chunks, pw.this.rank)).update_types(docs = pw.Json)
   
    results = results.with_columns(docs = pw.if_else(pw.this.docs.get("number").as_int(unwrap=True) < 1, web_content(pw.this.query), extract_list(pw.this.docs)))
    
    return results

  @pw.table_transformer
  def final_ans(self, results):
    results = results.with_columns(history=retrieve_history(pw.this.email, pw.this.chat_id))
    results = results.with_columns(result = self.llm(prompt_final_ans(pw.this.query, pw.this.docs, pw.this.history), temperature = 0.7))
    _ = results.select(k=update_history(pw.this.email, pw.this.chat_id, pw.this.query, pw.this.result))
    pw.io.null.write(_)
    return results

    
  @pw.table_transformer
  def __call__(self, queries:pw.Table):
    return self.final_ans(self.CRAG(self.retrieve_docs(self.equivalent_queries(queries))))
