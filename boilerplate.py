# app.py
import threading
import pathway as pw
from fastapi import FastAPI, Request
from pymongo import MongoClient
from datetime import datetime
import uuid

from pathway.stdlib.io import mongodb

# FastAPI app
app = FastAPI()

client = MongoClient("mongodb://localhost:27017")
history=MongoClient("mongodb://localhost:27018")
db = client["chatdb"]
query_collection = db["queries"]
response_collection = db["responses"]

@app.post("/ask")
async def submit_query(request: Request):
    body = await request.json()
    user_id = body["user_id"]
    query = body["query"]
    query_id = str(uuid.uuid4())

    query_collection.insert_one({
        "query_id": query_id,
        "user_id": user_id,
        "query": query,
        "createdAt": datetime.utcnow()
    })

    return {"message": "Query submitted", "query_id": query_id}

@app.get("/response/{query_id}")
def get_response(query_id: str):
    doc = response_collection.find_one({"query_id": query_id})
    return doc if doc else {"message": "Pending or not found"}

# Pathway schema and transformer logic
class QuerySchema(pw.Schema):
    query_id: str
    user_id: str
    query: str
    createdAt: datetime

class ResponseSchema(pw.Schema):
    query_id: str
    user_id: str
    response: str

@pw.udf
def crag_agent(query: str) -> str:
    # Your CRAG logic here
    return f"Response to: {query}"

@pw.table_transformer
def process(queries: pw.Table) -> pw.Table:
    return queries.select(
        query_id=pw.this.query_id,
        user_id=pw.this.user_id,
        response=crag_agent(pw.this.query)
    )

def run_pathway():
    source = mongodb.read(
        host="localhost",
        db="chatdb",
        collection="queries",
        schema=QuerySchema,
        primary_key="query_id",
        persistent=True
    )
    result = process(source)

    mongodb.write(
        result,
        host="localhost",
        db="chatdb",
        collection="responses",
    )

    pw.run()

# Run Pathway in a background thread
@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_pathway, daemon=True).start()