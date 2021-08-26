# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import records
from typing import List


class DBEngine:
    """
    DB Engine is mainly used for constructing our pre-training corpus.
    """

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_sql_query(self, sql_query: str):
        """
        Execute a given SQL query and return its results over specific database `self.db`.
        :param sql_query: a SQL query whose table name is always `w` under the Squall setting.
        :return: a list of results which follows the order of sqlite/
        """
        out = self.conn.query(sql_query)
        results = out.all()
        merged_results = []
        for i in range(len(results)):
            merged_results.extend(results[i].values())
        return merged_results

    def delete_rows(self, row_indices: List[int]):
        """
        During our proposed table pre-training, we rely on the execution on database to obtain the correct results for
        SQL queries. However, there are some tables which have invalid size (e.g., too large to fit in 1024 tokens).
        For these tables, we will dynamically delete random rows both in DATABASE format and CSV format when loading.
        The function is to keep the CSV content consistent with the DATABASE content.
        :param row_indices: rows index which will be deleted, and here we assume the database is created by inserting a
        column as `id` which records the row index (starting from 1).
        """
        sql_queries = [
            "delete from w where id == {}".format(row) for row in row_indices
        ]
        for query in sql_queries:
            self.conn.query(query)
