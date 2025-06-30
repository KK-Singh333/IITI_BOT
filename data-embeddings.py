import pathway as pw
from pathway.xpacks.llm.splitters import RecursiveSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from pathway.xpacks.llm.splitters import RecursiveSplitter

# schema for data
class IITIWebSchema(pw.Schema):
    row_id: int
    url: str
    title: str
    keywords: str
    description: str
    body_text: str
    error: str
    status: str

#load csv file
init_table = pw.io.csv.read(
    "output_data.csv",
    schema=IITIWebSchema,
    mode="static",
    # autocommit_duration_ms=600000
)

# check for non-empty body_text
final_table = init_table.filter(pw.this.body_text != "")

# Setup RecursiveSplitter
splitter = RecursiveSplitter(
    chunk_size=500,
    chunk_overlap=150,
    separators=["\n#", "\n##", "\n\n", "\n","."],
    model_name="gpt-4o-mini",
)

# Apply splitter
chunked = final_table.select(
    row_id=pw.this.row_id,
    url=pw.this.url,
    title=pw.this.title,
    chunks=splitter(pw.this.body_text)
)

#Flatten the chunks
flattened = chunked.flatten(pw.this.chunks)

# Load your model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunk(text: str) -> list[float]:
    """Returns a 1D embedding vector for a single text chunk"""
    # Process single text and return 1D list
    return model.encode(text).tolist()

# Add embeddings to each chunk
embedded = flattened.select(
    row_id=pw.this.row_id,
    chunk=pw.this.chunks,
    embedding=embed_chunk(pw.this.chunks),  # This batches internally
    url=pw.this.url,
)

pw.io.csv.write(
    table=embedded,
    filename="embedded_data_1.csv"
)

pw.run()