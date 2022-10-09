"""
Utils for squeezing the table content into a flatten sequence
"""

from typing import Dict

"""
Flatten Schema
"""


def flatten_schema(_table_content: Dict, input_query: str, start_row_idx: int = 0):
    compact_str = input_query.lower().strip() + " " + build_schema(_table_content, start_row_idx)
    return compact_str.strip()


def build_schema(_table_content: Dict, start_row_idx: int):
    return del_schema_format(_table_content, start_row_idx)


def del_schema_format(_table_content: Dict, start_row_idx: int):
    """
    Data format: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """
    _table_str = "col : " + " | ".join(_table_content["header"]) + " "
    _table_str = _table_str.lower()
    for i, row_example in enumerate(_table_content["rows"]):
        _table_str += "row " + str(start_row_idx + i + 1) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row_example]
        _table_str += " | ".join(row_cell_values) + " "

    return _table_str
