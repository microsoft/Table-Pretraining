import os
from argparse import ArgumentParser

import numpy
import pandas as pd
from tqdm import tqdm

from common.table_transform import *

random.seed(42)
numpy.random.seed(42)

TABLE_PATH = os.path.join("raw_dataset", "sqa")


def read_table_from_file(_wtq_table_name: str):
    rows = []
    assert ".csv" in _wtq_table_name
    table_data = pd.read_csv(os.path.join(TABLE_PATH, _wtq_table_name))
    # the first line is header
    header = list(table_data.columns)
    for row_data in table_data.values:
        rows.append([str(_) for _ in list(row_data)])

    return {
        "header": header,
        "rows": rows
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', help='source file for the prediction', type=str,
                        default='test')
    parser.add_argument('--source-file', help='source file for the prediction', type=str,
                        default='sqa/test.tsv')
    parser.add_argument('--data-dir', help='data directory to store the dataset', type=str,
                        default='dataset/sqa_chunk')

    parser = setup_parser(parser)
    args = parser.parse_args()

    folder = args.data_dir
    if not os.path.exists(folder):
        os.makedirs(folder)

    mode = args.mode

    input_f = open("{}/{}.src".format(folder, mode), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(folder, mode), "w", encoding="utf8")

    table_content_map = {}
    db_engine_map = {}
    history = ""

    with open(args.source_file) as fs:
        examples = open(args.source_file, "r", encoding="utf8").readlines()
        idx = 0
        for example in tqdm(examples[1:]):
            try:
                anno_id, _, position, question, table_file, _, answer_text = example.strip("\n").split("\t")
                answer_text = answer_text.replace("\"\"", "\"").strip("\"'")
                if position == "0":
                    # reset history
                    history = ""
                question = question.lower()
                if history:
                    question = history + " " + question
                answer = eval(answer_text)
                table_content = read_table_from_file(table_file)
                input_sources, output_targets = build_example(args, question, answer,
                                                              table_content, table_file,
                                                              is_train=mode == "train",
                                                              max_sen_length=1024)
                for input_s, output_t in zip(input_sources, output_targets):
                    input_f.write(input_s + "\n")
                    output_f.write(output_t + "\n")
                # reset the history
                history = question
                idx += 1
            except Exception as e:
                print("Error case on Line: {}, {}".format(idx, question))
                print(e)
    input_f.close()
    output_f.close()
