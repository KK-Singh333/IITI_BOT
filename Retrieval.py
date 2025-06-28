# Radhey Radhey
# Libraries Required - !pip install pathway pathway[xpack-llm] litellm
import pathway as pw
from pathway.xpacks.llm import llms, embedders
import os
from pathway.udfs import ExponentialBackoffRetryStrategy
import numpy as np
import ast
from pathway.stdlib.ml.index import KNNIndex
# from pathway.xpacks.llm import embedders

# from getpass import getpass
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY") or "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"

# Setting Model
model="groq/meta-llama/llama-4-scout-17b-16e-instruct"

system_prompt_retriever = """
You are an AI language model assistant.
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines.
Output only the generated queries not including any other text.
"""

# Doc Store
path = 'embedding_sample.csv'

class InputCSVDataSchema(pw.Schema):
    row_id: str
    chunk: str
    embedding: str
    url: str

@pw.udf
def split_lines(text: str) -> list[str]:
    return text.splitlines()

class Retriever():

  def __init__(self, model:str = model, system_prompt:str = system_prompt_retriever, path_csv:str = path):
    # Setting llm
    self.llm = llms.LiteLLMChat(model=model, retry_strategy=ExponentialBackoffRetryStrategy(max_retries=2))
    self.system_prompt = system_prompt
    self.embedder = embedders.SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    self.csv_data = pw.io.csv.read(
    path_csv,
    schema=InputCSVDataSchema,
    mode="static"
    )
    def parse_nested_embedding(embedding_str):
      try:
          # Parse the string to get nested list
          parsed = ast.literal_eval(embedding_str)
          # Extract the first (inner) list
          embedding_vector = parsed[0]
          return np.array(embedding_vector, dtype=np.float32)
      except Exception as e:
          print(f"Error parsing embedding: {e}")
          return None

    self.vector_store = self.csv_data.select(
    doc_id=pw.this.row_id,
    chunks=pw.this.chunk,
    url=pw.this.url,
    embedding=pw.apply(parse_nested_embedding, self.csv_data.embedding),
    # Include other columns you need
    ).filter(pw.this.embedding.is_not_none())

  @pw.table_transformer
  def __call__(self, queries:pw.Table):
    @pw.udf
    def query_parser(args) -> list[dict]:
      return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": f"{args}"}]

    query_table = queries.select(user_id = pw.this.user_id, questions = query_parser(pw.this.queries))
    responses = query_table.select(user_id = pw.this.user_id, result = self.llm(pw.this.questions))

    split_table = responses.select(
    user_id=pw.this.user_id,
    questions = split_lines(pw.this.result)  # This gives a List[str]
    )
    response = split_table.flatten(pw.this.questions)

    response += response.select(embedding=self.embedder(pw.this.questions))
    # return response

    doc_index = KNNIndex(
    self.vector_store.embedding,
    self.vector_store,
    n_dimensions= self.embedder.get_embedding_dimension(),  # dimension for all-MiniLM-L6-v2
    distance_type = "cosine"
    # n_and_d=2
    )

    results = doc_index.get_nearest_items(
    response.embedding,
    k=3  # top 5 most similar documents
    ).select(user_id = response.user_id, doc_id = pw.this.doc_id, chunks = pw.this.chunks, url = pw.this.url)

    results = results.groupby(pw.this.user_id).reduce(pw.this.user_id,chunks = pw.reducers.tuple(pw.this.chunks))
    return results

# Lets try once
# query = pw.debug.table_from_rows(
#     schema = pw.schema_from_types(user_id = str, queries=str),
#     rows = [
#         (
#             "1",
#             "What are the facilities in IIT Indore"
#         ),
#         (
#             "2",
#             "Who is Director of IIT Indore"
#         )
#     ]
# )

# custom_retriever = Retriever()
# pw.debug.compute_and_print(custom_retriever(queries=query))
# # custom_retriever(queries=query).typehints()
