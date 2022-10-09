import records
from typing import List


class WTQDBEngine:

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_wtq_query(self, sql_query: str):
        out = self.conn.query(sql_query)
        results = out.all()
        merged_results = []
        for i in range(len(results)):
            merged_results.extend(results[i].values())
        return merged_results

    def delete_rows(self, row_indices: List[int]):
        sql_queries = [
            "delete from w where id == {}".format(row) for row in row_indices
        ]
        for query in sql_queries:
            self.conn.query(query)
