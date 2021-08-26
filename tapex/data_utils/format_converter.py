# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json


def convert_fairseq_to_hf(fairseq_folder, data_prefix):
    src_file = os.path.join(fairseq_folder, data_prefix + ".src")
    tgt_file = os.path.join(fairseq_folder, data_prefix + ".tgt")
    out_f = open(os.path.join(fairseq_folder, data_prefix + ".json"), "w", encoding="utf8")
    with open(src_file, "r", encoding="utf8") as src_f, open(tgt_file, "r", encoding="utf8") as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            out_f.write(json.dumps({"input": src_line.strip(), "output": tgt_line.strip()}) + "\n")
