import logging
import os
from argparse import ArgumentParser

import numpy
from tqdm import tqdm
from typing import Tuple
from table_transform import *

TABLE_PATH = "wtq_origin"

logger = logging.getLogger(__name__)


def read_table_from_file(_wtq_table_name: str):
    def _extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        # _vals = ["empty" if _ == "" else _ for _ in _vals]
        return _vals

    rows = []
    assert ".csv" in _wtq_table_name
    # use the normalized table file
    _wtq_table_name = _wtq_table_name.replace(".csv", ".tsv")
    with open(os.path.join(TABLE_PATH, _wtq_table_name), "r", encoding="utf8") as table_f:
        table_lines = table_f.readlines()
        # the first line is header
        header = _extract_content(table_lines[0])
        for line in table_lines[1:]:
            rows.append(_extract_content(line))

    return {
        "header": header,
        "rows": rows
    }


if __name__ == '__main__':

    random.seed(42)
    numpy.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('--mode', help='source file for the prediction', type=str,
                        default='train')
    parser.add_argument('--source-file', help='source file for the prediction', type=str,
                        default='wtq_origin/train.tsv')
    parser.add_argument('--data-dir', help='data directory to store the dataset', type=str,
                        default='dataset/wtq_chunk')
    parser.add_argument('--max-sen-len', help='data directory to store the dataset', type=int)

    # setup up table transformation operations
    parser = setup_parser(parser)
    args = parser.parse_args()

    mode = args.mode
    folder = args.data_dir
    if not os.path.exists(folder):
        os.makedirs(folder)

    input_f = open("{}/{}.src".format(folder, mode), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(folder, mode), "w", encoding="utf8")

    table_content_map = {}
    db_engine_map = {}

    with open(args.source_file) as fs:
        examples = open(args.source_file, "r", encoding="utf8").readlines()
        for example in tqdm(examples[1:]):
            _, question, table_name, answer = example.strip("\n").split("\t")
            question = question.lower()
            answer = answer.split("|")
            # must contain rows and header keys
            table_content = read_table_from_file(table_name)
            input_sources, output_targets = build_fairseq_example(args, question, answer, table_content,
                                                                  table_name, mode == "train", args.max_sen_len)
            for input_s, output_t in zip(input_sources, output_targets):
                input_f.write(input_s + "\n")
                output_f.write(output_t + "\n")

    input_f.close()
    output_f.close()
