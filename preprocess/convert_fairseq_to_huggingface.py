import os
import json


def load_file_and_convert(fariseq_data_folder, huggingface_data_folder):
    train_src_file = os.path.join(fariseq_data_folder, "train.src")
    train_tgt_file = os.path.join(fariseq_data_folder, "train.tgt")
    dev_src_file = os.path.join(fariseq_data_folder, "valid.src")
    dev_tgt_file = os.path.join(fariseq_data_folder, "valid.tgt")

    out_train = open(os.path.join(huggingface_data_folder, "train.json"), "w", encoding="utf8")
    out_dev = open(os.path.join(huggingface_data_folder, "valid.json"), "w", encoding="utf8")
    with open(train_src_file, "r", encoding="utf8") as train_src_f, \
            open(train_tgt_file, "r", encoding="utf8") as train_tgt_f:
        src_lines = train_src_f.readlines()
        tgt_lines = train_tgt_f.readlines()
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            out_train.write(json.dumps({"input": src_line.strip(), "output": tgt_line.strip()}) + "\n")

    with open(dev_src_file, "r", encoding="utf8") as dev_src_f, \
            open(dev_tgt_file, "r", encoding="utf8") as dev_tgt_f:
        src_lines = dev_src_f.readlines()
        tgt_lines = dev_tgt_f.readlines()
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            out_dev.write(json.dumps({"input": src_line.strip(), "output": tgt_line.strip()}) + "\n")
