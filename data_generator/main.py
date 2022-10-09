import json
import os
import random
import shutil
from argparse import ArgumentParser

import numpy
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.common import *
from utils.dbengine import WTQDBEngine
from utils.template import apply_sql_on_target_table

DATABASE_PATH = "squall/tables/db"
# the temp database path will not affect the original db
TEMP_DATABASE_PATH = "squall/tables/temp_db"
TABLE_PATH = "squall/tables/json"
# the delimiter to distinguish different answers in the output
TGT_DEL = ", "
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")


def permute_table(_wtq_table_content: Dict):
    # shuffle header orders
    header = _wtq_table_content["header"]
    types = _wtq_table_content["types"]
    header_content = list(map(list, zip(*_wtq_table_content["rows"])))

    header_num = len(_wtq_table_content["header"])
    header_range = list(range(header_num))
    random.shuffle(header_range)

    # map from the original to the new
    origin_to_shuffle = {i: header_range[i] for i in range(header_num)}
    shuffle_header = [header[origin_to_shuffle[i]] for i in range(header_num)]
    shuffle_types = [types[origin_to_shuffle[i]] for i in range(header_num)]
    shuffle_content = [header_content[origin_to_shuffle[i]] for i in range(header_num)]
    shuffle_rows = list(map(list, zip(*shuffle_content)))

    return {
        "header": shuffle_header,
        "types": shuffle_types,
        "rows": shuffle_rows,
        "alias": _wtq_table_content["alias"]
    }


def truncate_table(_normalized_table: Dict, max_table_length):
    """
    Given a normalized table and the maximum flatten length, we should truncate the table if necessary
    """
    assert "header" in _normalized_table
    assert "rows" in _normalized_table
    header_string = "col : " + " | ".join(_normalized_table["header"]) + " "
    header_tokens = tokenizer.tokenize(header_string, add_special_tokens=False)
    # split all cell values into tokens and see how many can be adapt
    used_token_len = len(header_tokens)
    # remaining length
    remain_token_len = max_table_length - used_token_len

    value_string = ""
    for _, row_example in enumerate(_normalized_table["rows"]):
        # generally we do not want to make
        value_string += "row " + str(10) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row_example]
        value_string += " | ".join(row_cell_values) + " "
    value_token_len = len(tokenizer.tokenize(value_string))

    # calc a drop rate
    drop_rate = 1.0 - remain_token_len / value_token_len
    current_chunk_remain_size = remain_token_len
    truncate_table_contents = _normalized_table
    keep_rows = []
    for ind, row_example in enumerate(_normalized_table["rows"]):
        drop_cur_row = random.random() < drop_rate
        if drop_cur_row:
            # do not append this row
            continue
        value_string = "row " + str(ind) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row_example]
        value_string += " | ".join(row_cell_values)
        value_token_len = len(tokenizer.tokenize(value_string))
        # over the size limit, and take action
        if value_token_len > current_chunk_remain_size:
            truncate_table_contents = {
                "header": _normalized_table["header"],
                "rows": [_normalized_table["rows"][i] for i in keep_rows],
                "types": _normalized_table["types"],
                "alias": _normalized_table["alias"]
            }
            # return
            break
        keep_rows.append(ind)
        current_chunk_remain_size -= value_token_len

    # the id in database starts from 1
    removed_rows = [i + 1 for i in range(len(_normalized_table["rows"])) if i not in keep_rows]
    return truncate_table_contents, removed_rows


def process_table_structure(_wtq_table_content: Dict, _add_all_column: bool = False):
    # remove id and agg column
    headers = [_.replace("\n", " ").lower() for _ in _wtq_table_content["headers"][2:]]
    header_map = {}
    for i in range(len(headers)):
        header_map["c" + str(i + 1)] = headers[i]
    header_types = _wtq_table_content["types"][2:]

    all_headers = []
    all_header_types = []
    vertical_content = []
    for column_content in _wtq_table_content["contents"][2:]:
        # only take the first one
        if _add_all_column:
            for i in range(len(column_content)):
                column_alias = column_content[i]["col"]
                # do not add the numbered column
                if "_number" in column_alias:
                    continue
                vertical_content.append([str(_).replace("\n", " ").lower() for _ in column_content[i]["data"]])
                if "_" in column_alias:
                    first_slash_pos = column_alias.find("_")
                    column_name = header_map[column_alias[:first_slash_pos]] + " " + \
                                  column_alias[first_slash_pos + 1:].replace("_", " ")
                else:
                    column_name = header_map[column_alias]
                all_headers.append(column_name)
                if column_content[i]["type"] == "TEXT":
                    all_header_types.append("text")
                else:
                    all_header_types.append("number")
        else:
            vertical_content.append([str(_).replace("\n", " ").lower() for _ in column_content[0]["data"]])
    row_content = list(map(list, zip(*vertical_content)))

    if _add_all_column:
        ret_header = all_headers
        ret_types = all_header_types
    else:
        ret_header = headers
        ret_types = header_types
    return {
        "header": ret_header,
        "rows": row_content,
        "types": ret_types,
        "alias": list(_wtq_table_content["is_list"].keys())
    }


if __name__ == '__main__':

    random.seed(42)
    numpy.random.seed(42)

    parser = ArgumentParser()
    # TODO: if you want to provide your SQL templates, you could organize your file with the format of SQUALL data
    #  and you should also prepare the corresponding database files / csv files for tables.
    parser.add_argument('--template_file', help='SQL query file which provides the template for synthesizing more SQL queries',
                        type=str, default='squall/data.json')
    # the mode is to avoid sampling tables in the dev set of squall to avoid potential data leakage
    parser.add_argument('--mode', help='train or dev for pre-training', type=str, default='train',
                        choices=["train", "dev"])
    parser.add_argument('--dev_id_file', help='the dev id file to avoid potential data leakage', type=str,
                        default='squall/dev-0.ids')
    parser.add_argument('--instance_number', help='the expected instance number corresponding to each template', type=int,
                        default=500)
    parser.add_argument('--max_source_length', help='the maximum length for the flattened table plus input SQL query',
                        type=int,
                        default=1024)

    args = parser.parse_args()

    mode = args.mode
    dev_table_ids = json.load(open(args.dev_id_file, "r", encoding="utf8"))

    output_f = open("{}.json".format(mode), "w", encoding="utf8")

    # source table should not be truncated!
    src_table_content_map = {}
    # tgt table should be truncated!
    tgt_table_content_map = {}
    table_drop_rows_map = {}
    db_engine_map = {}

    if not os.path.exists(TEMP_DATABASE_PATH):
        os.makedirs(TEMP_DATABASE_PATH)
    for table_json_file in os.listdir(TABLE_PATH):
        table_id = table_json_file[:-5]
        check_table_file = open(os.path.join(TABLE_PATH, table_json_file), "r", encoding="utf8")
        src_table_content = json.load(check_table_file)
        src_table_content = process_table_structure(src_table_content)
        # source should not be truncated
        src_table_content_map[table_id] = json.loads(json.dumps(src_table_content))
        src_table_content, table_drop_rows = truncate_table(src_table_content,
                                                            max_table_length=args.max_source_length - 40)
        tgt_table_content_map[table_id] = src_table_content
        table_drop_rows_map[table_id] = table_drop_rows

    for table_db_file in os.listdir(DATABASE_PATH):
        table_id = table_db_file[:-3]
        # copy table db file into a temp file since we may delete some rows
        database_path = os.path.join(DATABASE_PATH, table_db_file)
        temp_database_path = os.path.join(TEMP_DATABASE_PATH, table_db_file)
        if os.path.exists(temp_database_path):
            os.remove(temp_database_path)
        # future operations on the temp db to avoid effecting the original database
        shutil.copy(database_path, temp_database_path)
        db_engine_map[table_id] = WTQDBEngine(temp_database_path)
        if table_id in table_drop_rows_map and len(table_drop_rows_map[table_id]) != 0:
            table_drop_rows = table_drop_rows_map[table_id]
            db_engine_map[table_id].delete_rows(table_drop_rows)

    valid_table_ids = list(src_table_content_map.keys() - set(dev_table_ids)) if mode == "train" else dev_table_ids
    max_source_len = args.max_source_length
    # count the dataset size
    dataset_counter = 0
    with open(args.template_file) as fs:
        examples = json.load(open(args.template_file, "r", encoding="utf8"))
        for example in tqdm(examples):
            table_id = example["tbl"]
            if mode == "dev" and table_id not in dev_table_ids:
                continue
            elif mode == "train" and table_id in dev_table_ids:
                continue
            sql_struct = example["sql"]
            question = " ".join(example["nl"]).lower()

            engine, src_table_content = db_engine_map[table_id], src_table_content_map[table_id]

            # augment data
            if args.instance_number > 0:
                instance_number_count = 0
                instance_number_upper = args.instance_number
                maximum_try_times = instance_number_upper * 10
                # if we have tried for more than 10 times and cannot get a reasonable execution result, just ignore it
                while instance_number_count < instance_number_upper and maximum_try_times >= 0:
                    random_table_id = random.choice(valid_table_ids)
                    tgt_table_content = tgt_table_content_map[random_table_id]
                    tgt_db_engine = db_engine_map[random_table_id]
                    try:
                        random_sql, random_answer, exec_sql = apply_sql_on_target_table(sql_struct,
                                                                                        src_table_content,
                                                                                        tgt_table_content,
                                                                                        tgt_db_engine,
                                                                                        _unexec_prob=0.0)
                        if len(random_answer) > 0:
                            flatten_input = flatten_schema(tgt_table_content, random_sql)
                            flatten_output = TGT_DEL.join([str(case).lower() for case in random_answer])
                            tokens = tokenizer.tokenize(flatten_input)
                            instance_number_count += 1
                            output_f.write(json.dumps({
                                "template_SQL": sql_struct,
                                "executable_SQL": exec_sql,
                                "input_SQL": random_sql,
                                "output_answer": random_answer,
                                "table": tgt_table_content,
                                "fairseq_input": flatten_input,
                                "fairseq_output": flatten_output
                            }) + "\n")
                    except Exception as e:
                        print(e)
                    # when exceeding the upper limit
                    maximum_try_times -= 1
            else:
                print("Cannot synthesize meaningful SQL queries from SQL: {}, table_id: {}".format(exec_sql, table_id))

    output_f.close()
