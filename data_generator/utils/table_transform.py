import math
import random
from typing import List
import logging
from data_generator.utils.common import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")

logger = logging.getLogger(__name__)

TGT_DEL = ", "
CHUNK_TOKEN = " <chunk> "


def build_fairseq_parsing_example(_args, _question: str, _sql: str, _table: Dict,
                                  _table_name: str, is_train: bool, max_sen_length):
    input_sources = []
    output_targets = []

    truncate_table_content, table_mapping = truncate_database_values(_table,
                                                                     _args.max_cell_length,
                                                                     _args.max_cell_truncate)
    # the chunk size of validation can be scale to a larger upperbound
    max_chunk_size = _args.max_chunk_size if is_train else _args.max_chunk_size_valid
    flatten_output = _sql.lower().strip()
    # we should split into multiple chunks to save memory
    chunk_table_contents, cover_ratio = split_long_table(truncate_table_content,
                                                         _question,
                                                         max_sen_length=max_sen_length)
    # in training, we should recall these training examples
    if len(chunk_table_contents) > max_chunk_size:
        maximum_val = 1.1 if max_chunk_size != 1 else 1.05
        drop_ratio = maximum_val - cover_ratio
        # truncate the table
        small_table_content = truncate_training_database(_table_name,
                                                         truncate_table_content,
                                                         _question,
                                                         drop_ratio,
                                                         _sql=_sql if is_train else None)
        if max_chunk_size == 1:
            split_mode = "greedy"
        else:
            split_mode = "average"
        chunk_table_contents, _ = split_long_table(small_table_content,
                                                   _question,
                                                   max_sen_length=max_sen_length,
                                                   split_mode=split_mode)

    flatten_inputs = []
    # the initial value of row
    row_idx = 0
    early_stop = False
    for _chunk_id, _table_content in enumerate(chunk_table_contents):
        if _chunk_id >= max_chunk_size:
            logger.warning("Table: {} has {} chunks which is too large to handle, truncate it.".format(
                _table_name, len(chunk_table_contents)))
            early_stop = True
        if early_stop:
            break

        flatten_inputs.append(flatten_schema(_table_content, _question, start_row_idx=row_idx))
        row_idx = len(_table_content["rows"]) + 1

    flatten_input = CHUNK_TOKEN.join(flatten_inputs).strip()
    input_sources.append(flatten_input)
    output_targets.append(flatten_output)

    if _args.permute_table > 0 and is_train:
        for _ in range(_args.permute_table):
            shuffle_table = permute_table(truncate_table_content)
            flatten_input = flatten_schema(shuffle_table, _question)
            input_sources.append(flatten_input)
            output_targets.append(flatten_output)

    return input_sources, output_targets


def build_fairseq_example(_args, _question: str, _answer: List, _table: Dict,
                          _table_name: str, is_train: bool, max_sen_length):
    input_sources = []
    output_targets = []

    truncate_table_content, table_mapping = truncate_database_values(_table,
                                                                     _args.max_cell_length,
                                                                     _args.max_cell_truncate)
    # the chunk size of validation can be scale to a larger upperbound
    max_chunk_size = _args.max_chunk_size if is_train else _args.max_chunk_size_valid
    if len(_answer) > 0:
        for i, case in enumerate(_answer):
            if case in table_mapping.keys():
                _answer[i] = table_mapping[case]
        flatten_output = TGT_DEL.join([str(case).lower() for case in _answer])
        # we should split into multiple chunks to save memory
        chunk_table_contents, cover_ratio = split_long_table(truncate_table_content,
                                                             _question,
                                                             max_sen_length=max_sen_length)
        # in training, we should recall these training examples
        if len(chunk_table_contents) > max_chunk_size:
            maximum_val = 1.1 if max_chunk_size != 1 else 1.05
            drop_ratio = maximum_val - cover_ratio
            # truncate the table
            small_table_content = truncate_training_database(_table_name,
                                                             truncate_table_content,
                                                             _question,
                                                             drop_ratio,
                                                             _answers=_answer if is_train else None)
            if max_chunk_size == 1:
                split_mode = "greedy"
            else:
                split_mode = "average"
            chunk_table_contents, _ = split_long_table(small_table_content,
                                                       _question,
                                                       max_sen_length=max_sen_length,
                                                       split_mode=split_mode)

        flatten_inputs = []
        # the initial value of row
        row_idx = 0
        early_stop = False
        for _chunk_id, _table_content in enumerate(chunk_table_contents):
            if _chunk_id >= max_chunk_size:
                logger.warning("Table: {} has {} chunks which is too large to handle, truncate it.".format(
                    _table_name, len(chunk_table_contents)))
                early_stop = True
            if early_stop:
                break

            flatten_inputs.append(flatten_schema(_table_content, _question, start_row_idx=row_idx))
            row_idx = len(_table_content["rows"]) + 1

        flatten_input = CHUNK_TOKEN.join(flatten_inputs).strip()
        input_sources.append(flatten_input)
        output_targets.append(flatten_output)

        if _args.permute_table > 0 and is_train:
            for _ in range(_args.permute_table):
                shuffle_table = permute_table(truncate_table_content)
                flatten_input = flatten_schema(shuffle_table, _question)
                input_sources.append(flatten_input)
                output_targets.append(flatten_output)
    else:
        logger.warning("Empty answer in case: {}".format(_question))

    return input_sources, output_targets


def setup_parser(_parser):
    _parser.add_argument(
        "--max-cell-length",
        type=int,
        default=15,
        help="if the cell's length is larger that this, it should be processed",
    )

    _parser.add_argument(
        "--max-cell-truncate",
        type=int,
        default=15,
        help="truncate cell's length into a value less than this one",
    )

    _parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1,
        help="truncate table length",
    )
    _parser.add_argument(
        '--permute_table',
        help='augment count',
        type=int,
        default=0)

    _parser.add_argument(
        "--max-chunk-size-valid",
        type=int,
        default=1,
        help="truncate table length",
    )

    return _parser


def permute_table(_wtq_table_content: Dict):
    # shuffle header orders
    header = _wtq_table_content["header"]
    header_content = list(map(list, zip(*_wtq_table_content["rows"])))

    header_num = len(_wtq_table_content["header"])
    header_range = list(range(header_num))
    random.shuffle(header_range)

    # map from the original to the new
    origin_to_shuffle = {i: header_range[i] for i in range(header_num)}
    shuffle_header = [header[origin_to_shuffle[i]] for i in range(header_num)]
    shuffle_content = [header_content[origin_to_shuffle[i]] for i in range(header_num)]
    shuffle_rows = list(map(list, zip(*shuffle_content)))

    # random.shuffle(shuffle_rows)

    return {
        "header": shuffle_header,
        "rows": shuffle_rows
    }


def split_long_table(_normalized_table: Dict, input_query: str, max_sen_length, split_mode="greedy"):
    assert "header" in _normalized_table
    assert "rows" in _normalized_table
    number_of_rows = len(_normalized_table["rows"])
    # TODO: avg split table into relatively average chunks
    query_tokens = tokenizer.tokenize(input_query, add_special_tokens=True)
    header_string = "col : " + " | ".join(_normalized_table["header"]) + " "
    header_tokens = tokenizer.tokenize(header_string, add_special_tokens=False)
    # split all cell values into tokens and see how many can be adapt
    used_token_len = len(query_tokens) + len(header_tokens)
    # remaining length
    remain_token_len = max_sen_length - 2 - used_token_len

    value_string = ""
    for _, row_example in enumerate(_normalized_table["rows"]):
        # generally we do not want to make
        value_string += "row " + str(100) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row_example]
        value_string += " | ".join(row_cell_values) + " "
    value_token_len = len(tokenizer.tokenize(value_string))
    # used to estimate the busy ratio
    whole_token_len = used_token_len + value_token_len

    # maximum chunk size
    chunk_size = math.ceil(value_token_len / remain_token_len)
    if chunk_size == 1:
        return [_normalized_table], 1.0

    if split_mode == "greedy":
        remain_token_len = remain_token_len
    elif split_mode == "average":
        remain_token_len = min(remain_token_len, 100 + math.ceil(value_token_len / chunk_size))
    else:
        raise Exception("Do not support split_mode {}".format(split_mode))

    current_chunk_remain_size = remain_token_len
    current_chunk_row = 0
    split_table_contents = []
    for ind, row_example in enumerate(_normalized_table["rows"]):
        value_string = "row " + str(ind) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row_example]
        value_string += " | ".join(row_cell_values)
        value_token_len = len(tokenizer.tokenize(value_string))
        # over the size limit, and take action
        if value_token_len > current_chunk_remain_size:
            split_table_contents.append({
                "header": _normalized_table["header"],
                "rows": _normalized_table["rows"][current_chunk_row: ind]
            })
            # reset every thing
            current_chunk_row = ind
            current_chunk_remain_size = remain_token_len
        current_chunk_remain_size -= value_token_len

    if current_chunk_row != (number_of_rows - 1):
        split_table_contents.append({
            "header": _normalized_table["header"],
            "rows": _normalized_table["rows"][current_chunk_row:]
        })

    return split_table_contents, float(max_sen_length / whole_token_len)


def truncate_training_database(_table_id, _table_content, _question, _drop_ratio: float, _answers=None, _sql=None):
    truncated_unrelated_indices = []
    related_indices = []
    if _answers is None:
        answer_set = set([])
    else:
        answer_set = set([ans_ex.lower() for ans_ex in _answers])
    # add _sql into answer_set
    if _sql is not None:
        answer_set.update(_sql.split())
    question_set = set(_question.strip("?!.,").split(" "))
    row_max_len = len(_table_content["rows"])
    for _row_idx, row in enumerate(_table_content["rows"]):
        lower_row = set([str(cell).lower() for cell in row])
        if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
            truncated_unrelated_indices.append(_row_idx)
        else:
            # add neighbours to preserve information aggressively
            related_indices.append([_row_idx - 2, _row_idx - 1, _row_idx, _row_idx + 1, _row_idx + 2])

    # remove the neighbours
    truncated_unrelated_indices = [_row_idx for _row_idx in truncated_unrelated_indices
                                   if _row_idx not in related_indices]
    # select some cases to drop
    drop_items = min(len(truncated_unrelated_indices), int(len(_table_content["rows"]) * _drop_ratio))
    drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

    for _row_idx in reversed(range(row_max_len)):
        if _row_idx in drop_row_indices:
            del _table_content["rows"][_row_idx]

    # only when the drop ratio is too large, logging for warning.
    if _drop_ratio >= 0.1:
        logger.warning("Drop {:.2f} rows in table {}".format(len(drop_row_indices), _table_id))

    return _table_content


def truncate_database_values(_table_content, max_cell_length, max_cell_truncate):
    """
    This function is to process the wikitablequestion answer content to avoid too long sequence.
    To achieve it, there are several principles:
    1.
    :param _table_content: `Dict` contains keys as `header` and `rows`
    :param max_cell_length: `int` which indicates the maximum cell value length
    :param max_cell_truncate: `int` which indicates the maximum cell value truncated length
    :return: a new table_content and a mapping from answer to a value
    """

    def _truncate_cell(cell_value):
        # do not process on these cases
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != "":
            try_tokens = tokenizer.tokenize(cell_value)
            if len(try_tokens) >= max_cell_length:
                retain_tokens = try_tokens[:max_cell_truncate]
                retain_cell_value = tokenizer.convert_tokens_to_string(retain_tokens)
                return retain_cell_value
            else:
                return None
        else:
            return cell_value

    cell_mapping = {}
    for row in _table_content["rows"]:
        for i, cell in enumerate(row):
            truncate_cell = _truncate_cell(cell)
            if truncate_cell is not None:
                cell_mapping[cell] = truncate_cell
                row[i] = truncate_cell
    return _table_content, cell_mapping
