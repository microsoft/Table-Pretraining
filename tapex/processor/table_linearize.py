# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
import abc
from typing import Dict


class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def __init__(self, lower_case):
        # if lower case, return the uncased table str; otherwise the cased.
        self.lower_case = lower_case

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        _table_str = "col : " + " | ".join(table_content["header"]) + " "
        for i, row_example in enumerate(table_content["rows"]):
            # start from row 1 not from row 0
            _table_str += "row " + str(i + 1) + " : "
            row_cell_values = []
            for cell_value in row_example:
                if isinstance(cell_value, int):
                    row_cell_values.append(str(cell_value))
                elif self.lower_case:
                    row_cell_values.append(cell_value.lower())
                else:
                    row_cell_values.append(cell_value)
            _table_str += " | ".join(row_cell_values) + " "
        return _table_str
