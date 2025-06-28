from crag import CRAG
import threading
import pathway as pw
from fastapi import FastAPI, Request
from pymongo import MongoClient
from datetime import datetime
import uuid

app=FastAPI()

mongo=MongoClient("mongodb://localhost:27017")
database=mongo['IITI_BOT']
input_collection=database['input_field']
history_collection=database['history_field']

@pw.table_transformer
def process_input(input_table: pw.Table,history_table: pw.Table)->pw.Table:
    rag=CRAG()
    final_table=rag.answer_query(input_table,history_table).select(queryid=pw.this.queryid,userid=pw.this.userid,query=pw.self.query,response=pw.this.response)
    return final_table
@pw.udf
def append_history(chat:pw.Json,query:str,response:str):
    return {'history':[{'query':query,'response':response},*chat['history']]}
@pw.table_transformer
def update_history(history_table :pw.Table,result:pw.Table)->pw.Table:
    joint_table=history_table.join(result,pw.left.userid==pw.right.userid)
    joint_table=joint_table.select(userid=pw.this.userid,history=append_history(pw.this.history,pw.this.query,pw.this.response))
    return joint_table
@app.post("/ask")
async def submit_query(request: Request):
    body = await request.json()
    user_id = body["user_id"]
    query = body["query"]
    query_id = str(uuid.uuid4())


    if not history_collection.find_one({"userid": user_id}):
        history_collection.insert_one({
        "userid": user_id,
        "history": { 
            "history": [] 
        }
    })


    input_collection.insert_one({
        "query_id": query_id,
        "user_id": user_id,
        "query": query,
        "createdAt": datetime.utcnow()
    })

    return {"message": "Query submitted", "query_id": query_id}

rdkafka_settings = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "iiti-bot-data",
    "auto.offset.reset": "earliest",
}

class InputSchema(pw.Schema):
    queryid : str =  pw.column_definition(primary_key=True)
    userid : str
    query : str
class HistorySchema(pw.Schema):
    userid: str
    history : pw.Json


def run_pathway():
   input = pw.io.debezium.read(
   rdkafka_settings,
   topic_name="IITI_BOT.input_field",
   schema=InputSchema
)
   history =  pw.io.debezium.read(
   rdkafka_settings,
   topic_name="IITI_BOT.history_field",
   schema=HistorySchema
)
   result = process_input(input,history)
   new_history=update_history(history,result)
   pw.io.mongodb.write(
       new_history,
       connection_string="mongodb://127.0.0.1:27017/",
       database='IITI_BOT',
       collection='history_field'
   )
#result format : QueryID| UserID | Query | Response
#history format : UserID | History(pw.Json)
   pw.io.mongodb.write(
        result,
        connection_string="mongodb://127.0.0.1:27017/",
        database='IITI_BOT',
        collection="responses",
    )

   pw.run()




