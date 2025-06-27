from crag import CRAG
@pw.table_transformer
def process_input(input_table: pw.Table,history_table: pw.Table)->pw.Table:
    rag=CRAG()
    final_table=rag.answer_query(input_table,history_table).select(queryid=pw.this.queryid,userid=pw.this.userid,query=pw.self.query,response=pw.this.response)
