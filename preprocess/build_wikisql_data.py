#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser
from copy import deepcopy

from tqdm import tqdm

from common.table_transform import *
from wikisql_utils.executor import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER
from wikisql_utils.wikisql_common import count_lines

TGT_DEL = ", "


def _parse_table(table):
    """Runs the type converter over the table cells."""
    ret_table = deepcopy(table)
    types = ret_table['types']
    ret_table['real_rows'] = ret_table['rows']
    typed_rows = []
    for row in ret_table['rows']:
        typed_row = []
        for column, cell_value in enumerate(row):
            typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
        typed_rows.append(typed_row)
    ret_table['rows'] = typed_rows
    return ret_table


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', help='source file for the prediction', type=str,
                        default='train')
    parser.add_argument('--source-file', help='source file for the prediction', type=str, default='wikisql/data/train.jsonl')
    parser.add_argument('--db-file', help='source database for the prediction', type=str, default='wikisql/data/train.db')
    parser.add_argument('--table-file', help='source table content for the prediction', type=str,
                        default='wikisql/data/train.tables.jsonl')
    parser.add_argument('--data-dir', help='data directory to store the dataset', type=str,
                        default='dataset/wikisql')
    parser = setup_parser(parser)
    args = parser.parse_args()

    mode = args.mode
    # record table id to table content
    table_content_map = {}

    for json_line in open(args.table_file, "r", encoding="utf8"):
        content = json.loads(json_line)
        table_content_map[content["id"]] = content

    folder = args.data_dir
    if not os.path.exists(folder):
        os.makedirs(folder)

    input_f = open("{}/{}.src".format(folder, mode), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(folder, mode), "w", encoding="utf8")

    with open(args.source_file) as fs:
        for ls in tqdm(fs, total=count_lines(args.source_file)):
            example = json.loads(ls)
            table_id = example["table_id"]
            table_content = table_content_map[table_id]
            question = example["question"].lower()
            tapas_table = _parse_table(table_content)
            answer = retrieve_wikisql_query_answer_tapas(tapas_table, example)
            input_sources, output_targets = build_example(args, question, answer,
                                                          table_content, table_id,
                                                          is_train=mode == "train",
                                                          max_sen_length=1024)
            for input_s, output_t in zip(input_sources, output_targets):
                input_f.write(input_s + "\n")
                output_f.write(output_t + "\n")

    input_f.close()
    output_f.close()
