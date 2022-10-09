from typing import List, Dict, Optional
from .dbengine import WTQDBEngine
from collections import defaultdict
import re
import random
from copy import deepcopy
from sqlalchemy.exc import SQLAlchemyError


class KeywordType:
    column = "Column"
    value_string = "Literal.String"
    value_number = "Literal.Number"


def retrieve_wtq_query_answer(_engine, _table_content, _sql_struct: List):
    # do not append id / agg
    headers = _table_content["header"]

    def flatten_sql(_ex_sql_struct: List):
        # [ "Keyword", "select", [] ], [ "Column", "c4", [] ]
        _encode_sql = []
        _execute_sql = []
        for _ex_tuple in _ex_sql_struct:
            keyword = str(_ex_tuple[1])
            # extra column, which we do not need in result
            if keyword == "w" or keyword == "from":
                pass
            elif re.fullmatch(r"c\d+(_.+)?", keyword):
                # only take the first part
                index_key = int(keyword.split("_")[0][1:]) - 1
                _encode_sql.append(headers[index_key])
            else:
                _encode_sql.append(keyword)
            # c4_list, replace it with the original one
            if "_address" in keyword or "_list" in keyword:
                keyword = re.findall(r"c\d+", keyword)[0]

            _execute_sql.append(keyword)

        return " ".join(_execute_sql), " ".join(_encode_sql)

    _exec_sql_str, _encode_sql_str = flatten_sql(_sql_struct)
    try:
        _sql_answers = _engine.execute_wtq_query(_exec_sql_str)
    except SQLAlchemyError as e:
        _sql_answers = []
    _norm_sql_answers = [str(_).replace("\n", " ") for _ in _sql_answers if _ is not None]
    if "none" in _norm_sql_answers:
        _norm_sql_answers = []
    return _encode_sql_str, _norm_sql_answers, _exec_sql_str


def apply_sql_on_target_table(_sql_struct: List,
                              _src_table: Dict,
                              _tgt_table: Dict,
                              _tgt_dbengine: WTQDBEngine,
                              _unexec_prob: float = 0.0):
    """
    Apply the sql struct on the table to produce a new SQL. The basic idea is as following:
    1. Identify the column and column type

    :param _sql_struct: the sql whose structure follows the same format as in squall
    :param _src_table: the original table content, to identify the possible cell value position
    :param _tgt_table: the table content which should be applied on the SQL
    :param _tgt_dbengine: the dbengine is employed to validate the produced SQL, to ensure it can return a reasonable
    result.
    :param _unexec_prob: the probability of producing un-executable SQL queries, by default 0.0. You can try different choices
    :return: the encoded SQL and its corresponding answer
    """

    def is_number_col(_col_type: str) -> bool:
        if "num" in _col_type or "time" in _col_type or "timespan" in _col_type:
            return True
        else:
            return False

    # find types of columns in _origin_table
    src_col_content = list(map(list, zip(*_src_table["rows"])))
    src_val_records = defaultdict(lambda: set())
    for i in range(len(src_col_content)):
        col_example = src_col_content[i]
        for j in range(len(col_example)):
            # map from value to a column
            src_val_records[str(col_example[j])].add("c" + str(i + 1))

    # use for sample value to replace the original one
    tgt_col_content = list(map(list, zip(*_tgt_table["rows"])))
    tgt_val_list = []
    for example in tgt_col_content:
        tgt_val_list.extend([_ for _ in example if _ != "none"])

    tgt_col_alias = _tgt_table["alias"]

    tgt_num_cols, tgt_str_cols = [], []
    for i in range(len(_tgt_table["header"])):
        if is_number_col(_tgt_table["types"][i]):
            tgt_num_cols.append("c" + str(i + 1))
        else:
            tgt_str_cols.append("c" + str(i + 1))

    valid_col_cands = {}
    src_col_num = {}
    # find the column types
    for i, src_type in enumerate(_src_table["types"]):
        col_name = "c" + str(i + 1)
        if is_number_col(src_type):
            valid_col_cands[col_name] = tgt_num_cols
            src_col_num[col_name] = True
        else:
            valid_col_cands[col_name] = tgt_str_cols
            src_col_num[col_name] = False

    _tgt_encode_sql, _tgt_answer, _tgt_exec_sql = "", [], ""
    for index in range(10):
        src_map_to_tgt = {}
        _tgt_sql_struct = deepcopy(_sql_struct)
        for i in range(len(_sql_struct)):
            keyword_type, keyword_name, _ = _sql_struct[i]
            # if there has establish the mapping, directly replace it
            if keyword_name in src_map_to_tgt:
                pass
            # otherwise, if it is a column
            elif keyword_type == KeywordType.column:
                src_col_name = keyword_name.split("_")[0]
                valid_tgt_col_cands = valid_col_cands[src_col_name]
                # such a template cannot apply on the target database
                if len(valid_tgt_col_cands) == 0:
                    return "", [], ""
                # any column has at least one valid target
                tgt_col_name = random.choice(valid_tgt_col_cands)
                # if there is any suffix, try to match it
                if "_" in keyword_name:
                    src_col_suffix = keyword_name.split("_")[1]
                    # try to match the original suffix
                    if tgt_col_name + "_" + src_col_suffix in tgt_col_alias:
                        tgt_col_name = tgt_col_name + "_" + src_col_suffix
                    # add number suffix
                    elif src_col_num[src_col_name]:
                        tgt_col_name = tgt_col_name + "_number"
                    # keep the original one
                    else:
                        pass
                src_map_to_tgt[keyword_name] = tgt_col_name
            # if it is a value (number)
            elif keyword_type in [KeywordType.value_number, KeywordType.value_string]:
                src_val_name = str(keyword_name).strip("'")
                if src_val_name in src_val_records:
                    src_val_pos = src_val_records[src_val_name]
                    # existing src names with the table position
                    src_used_col = set([_.split("_")[0] for _ in src_map_to_tgt.keys()])
                    src_val_col = list(src_val_pos & src_used_col)
                    # if src_val_col is empty, skip
                    if len(src_val_col) != 0:
                        src_val_col = random.choice(src_val_col)
                        # take the mapping column
                        if src_val_col not in src_map_to_tgt:
                            for src_col_name in src_map_to_tgt.keys():
                                if src_val_col in src_col_name:
                                    src_val_col = src_col_name
                        tgt_val_col = src_map_to_tgt[src_val_col]
                        tgt_val_col_ind = int(tgt_val_col.split("_")[0][1:]) - 1
                        # find the content, randomly take one value as the replacement
                        tgt_rand_val = tgt_col_content[tgt_val_col_ind]
                        tgt_rand_val = random.choice(tgt_rand_val)
                        try:
                            src_map_to_tgt[keyword_name] = int(tgt_rand_val)
                        except ValueError:
                            src_map_to_tgt[keyword_name] = "'{}'".format(tgt_rand_val)
                else:
                    if keyword_type == KeywordType.value_number:
                        random_val = str(random.randint(0, 2020))
                    else:
                        random_val = "'{}'".format(random.choice(tgt_val_list))
                    src_map_to_tgt[keyword_name] = random_val

            # do not replace reserved key words
            if keyword_name in src_map_to_tgt and \
                    keyword_type in [KeywordType.column,
                                     KeywordType.value_number,
                                     KeywordType.value_string]:
                _tgt_sql_struct[i][1] = src_map_to_tgt[keyword_name]

        _tgt_encode_sql, _tgt_answer, _tgt_exec_sql = \
            retrieve_wtq_query_answer(_tgt_dbengine, _tgt_table, _tgt_sql_struct)
        real_prob = random.random()

        if 0 < len(_tgt_answer) <= 10 and real_prob >= _unexec_prob:
            break
        elif len(_tgt_answer) == 0 and real_prob < _unexec_prob:
            # fake a universal answer
            _tgt_answer = ["empty"]
        else:
            _tgt_encode_sql = ""
            _tgt_answer = []
            continue

    return _tgt_encode_sql, _tgt_answer, _tgt_exec_sql
