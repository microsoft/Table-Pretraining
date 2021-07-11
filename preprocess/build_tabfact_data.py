# coding=utf8

import os
import json
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")


def read_jsonl(path: str) -> List[Dict]:
    data = list()
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
    return data


def flatten_table(table: List[List[str]]) -> str:
    str_rows = list()
    for rid, row in enumerate(table):
        values = " | ".join([value.lower() for value in row]) + " "
        if rid == 0:
            str_rows.append("col : %s" % values)
        else:
            str_rows.append("row %d : %s" % (rid, values))
    return "".join(str_rows)


def main(data_path: str, splits: List[str], task: str = "classification") -> None:
    for split in splits:
        path = os.path.join(data_path, "%s.jsonl" % split)

        examples = read_jsonl(path)

        if task == 'classification':
            save_input0_path = os.path.join(data_path, "%s.raw.input0" % split)
            save_label_path = os.path.join(data_path, "%s.label" % split)
        else:
            save_input0_path = os.path.join(data_path, "%s.src" % split)
            save_label_path = os.path.join(data_path, "%s.tgt" % split)

        with open(save_input0_path, "w", encoding="utf8") as input0_f, open(save_label_path, "w", encoding="utf8") as label_f:
            for ex in tqdm(examples):
                sentence = ex['statement']
                label = ex['label']
                table_text = flatten_table(ex['table_text'])
                input_line = "%s %s" % (sentence.lower(), table_text)
                try_input_tokens = tokenizer.tokenize(input_line)
                if len(try_input_tokens) > 1020:
                    try_input_tokens = try_input_tokens[: 1020]
                    input_line = tokenizer.convert_tokens_to_string(try_input_tokens)
                    print("Warning: truncate the source from {} token to 1020 tokens.".format(len(try_input_tokens)))
                input0_f.write(input_line + "\n")
                if task == 'classification':
                    label_f.write("%d\n" % label)
                else:
                    answer = "yes" if label == 1 else "no"
                    label_f.write("%s\n" % answer)


def split_test_data(test_data_path, split_json_files, out_dir):
    examples = read_jsonl(test_data_path)
    for split_file in split_json_files:
        split_mode = os.path.split(split_file)[-1][:-5]
        valid_id_list = json.load(open(split_file, "r", encoding="utf8"))
        valid_examples = [example for example in examples if example["table_id"] in valid_id_list]
        save_input0_path = os.path.join(out_dir, "%s.raw.input0" % ("test_" + split_mode))
        save_label_path = os.path.join(out_dir, "%s.label" % ("test_" + split_mode))

        with open(save_input0_path, "w", encoding="utf8") as input0_f, open(save_label_path, "w", encoding="utf8") as label_f:
            for ex in tqdm(valid_examples):
                sentence = ex['statement']
                label = ex['label']
                table_text = flatten_table(ex['table_text'])
                input_line = "%s %s" % (sentence.lower(), table_text)
                try_input_tokens = tokenizer.tokenize(input_line)
                if len(try_input_tokens) > 1020:
                    try_input_tokens = try_input_tokens[: 1020]
                    input_line = tokenizer.convert_tokens_to_string(try_input_tokens)
                    print("Warning: truncate the source")
                input0_f.write(input_line + "\n")
                label_f.write("%d\n" % label)


if __name__ == '__main__':
    data_path = "dataset/tabfact_classification"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    splits = ["train", "valid", "test"]
    main(data_path, splits, task="classification")
    split_test_data("tabfact/test.jsonl",
                    ["tabfact/complex.json",
                     "tabfact/small.json",
                     "tabfact/simple.json"],
                    "dataset/tabfact_classification")
