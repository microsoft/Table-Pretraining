# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import zipfile
import logging
from tqdm import tqdm
import json
from tapex.common.download import download_file
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_classification
from tapex.data_utils.preprocess_binary import fairseq_binary_classification
from typing import List

RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PATH = os.path.join(RAW_DATASET_FOLDER, "tabfact")
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large, tapex.base, tapex.large
MODEL_NAME = "tapex.base"
logger = logging.getLogger(__name__)


def download_tabfact():
    """
    Download WikiTableQuestion dataset and unzip the files
    """
    tabfact_url = "https://github.com/microsoft/Table-Pretraining/"\
                  "releases/download/origin-data/tabfact.zip"
    tabfact_raw_path = os.path.join(RAW_DATASET_FOLDER, "tabfact")
    tabfact_zip_file = download_file(tabfact_url)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(tabfact_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_tabfact_path = os.path.join(RAW_DATASET_FOLDER, "TabFact")
    shutil.move(unzip_tabfact_path, tabfact_raw_path)
    # remove the original file
    os.remove(tabfact_zip_file)
    return tabfact_raw_path


def split_fine_grained_test(data_dir):
    test_examples = [json.loads(line)
                     for line in open(os.path.join(data_dir, "test.jsonl"), "r", encoding="utf8").readlines()]
    split_modes = ["complex", "simple", "small"]
    for split_mode in split_modes:
        valid_id_list = json.load(open(os.path.join(data_dir, split_mode + ".json"),
                                       "r", encoding="utf8"))
        valid_examples = [example for example in test_examples if example["table_id"] in valid_id_list]
        save_test_path = os.path.join(data_dir, "%s.jsonl" % ("test_" + split_mode))
        with open(save_test_path, "w", encoding="utf8") as save_f:
            for example in valid_examples:
                save_f.write(json.dumps(example) + "\n")


def build_tabfact_fairseq_dataset(out_prefix, src_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def _read_table(_tabfact_example: List):
        header = _tabfact_example[0]
        rows = _tabfact_example[1:]

        return {
            "header": header,
            "rows": rows
        }

    input_f = open("{}/{}.raw.input0".format(data_dir, out_prefix), "w", encoding="utf8")
    output_f = open("{}/{}.label".format(data_dir, out_prefix), "w", encoding="utf8")

    lines = open(src_file, "r", encoding="utf8").readlines()
    for line in lines:
        line = line.strip()
        example = json.loads(line)
        sentence = example['statement'].lower()
        label = example['label']
        table_content = _read_table(example['table_text'])
        input_source = TABLE_PROCESSOR.process_input(table_content, sentence, []).lower()
        # Here we use the paradigm of BART to conduct classification on TabFact.
        # Therefore, the output should be a label rather than a text.
        output_target = str(label)
        input_f.write(input_source + "\n")
        output_f.write(output_target + "\n")

    input_f.close()
    output_f.close()


def preprocess_tabfact_dataset(processed_data_dir):
    fairseq_bpe_classification(processed_data_dir, resource_name=MODEL_NAME)
    fairseq_binary_classification(processed_data_dir, resource_name=MODEL_NAME)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    logger.info("*" * 80)
    logger.info("Prepare to download preprocessed TabFact json line file from our released link...")
    tabfact_raw_data_dir = download_tabfact()

    logger.info("Download finished! The original TabFact dataset is saved in {}".format(tabfact_raw_data_dir))
    processed_tabfact_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, "tabfact")

    split_fine_grained_test(tabfact_raw_data_dir)
    logger.info("*" * 80)
    logger.info("Process the dataset and save the processed dataset in {}".format(processed_tabfact_data_dir))
    build_tabfact_fairseq_dataset("train", os.path.join(tabfact_raw_data_dir, "train.jsonl"),
                                  processed_tabfact_data_dir)
    build_tabfact_fairseq_dataset("valid", os.path.join(tabfact_raw_data_dir, "valid.jsonl"),
                                  processed_tabfact_data_dir)
    build_tabfact_fairseq_dataset("test", os.path.join(tabfact_raw_data_dir, "test.jsonl"),
                                  processed_tabfact_data_dir)
    build_tabfact_fairseq_dataset("test_simple", os.path.join(tabfact_raw_data_dir, "test_simple.jsonl"),
                                  processed_tabfact_data_dir)
    build_tabfact_fairseq_dataset("test_complex", os.path.join(tabfact_raw_data_dir, "test_complex.jsonl"),
                                  processed_tabfact_data_dir)
    build_tabfact_fairseq_dataset("test_small", os.path.join(tabfact_raw_data_dir, "test_small.jsonl"),
                                  processed_tabfact_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to BPE and build the dataset binaries in {0}/input0 and {0}/label".format(processed_tabfact_data_dir))
    preprocess_tabfact_dataset(processed_tabfact_data_dir)

    logger.info("*" * 80)
    logger.info("Now you can train models using {} as the <data_dir> argument. "
                "More details in `run_model.py`.".format(os.path.join(processed_tabfact_data_dir, MODEL_NAME)))
