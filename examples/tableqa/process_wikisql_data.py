# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import tarfile
from copy import deepcopy

from tapex.common.download import download_file
from tapex.data_utils.wikisql.executor import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.format_converter import convert_fairseq_to_hf


RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large, tapex.base, tapex.large
MODEL_NAME = "tapex.base"
logger = logging.getLogger(__name__)


def download_wikisql():
    """
    Download WikiSQL dataset and unzip the files
    """
    wikisql_url = "https://raw.github.com/salesforce/WikiSQL/master/data.tar.bz2"
    wikisql_raw_path = os.path.join(RAW_DATASET_FOLDER, "wikisql")
    wikisql_tar_file = download_file(wikisql_url)
    # unzip and move it into raw_dataset folder
    tar = tarfile.open(wikisql_tar_file, "r:bz2")
    tar.extractall(wikisql_raw_path)
    tar.close()
    # remove the original file
    os.remove(wikisql_tar_file)
    return wikisql_raw_path


def build_wikisql_fariseq_dataset(out_prefix, src_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def _convert_table_types(table):
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

    # load table content dictionary from files
    table_content_dict = {}
    table_file_path = "{}.tables.jsonl".format(src_file.split(".")[0])
    for json_line in open(table_file_path, "r", encoding="utf8"):
        content = json.loads(json_line)
        table_content_dict[content["id"]] = content

    input_f = open("{}/{}.src".format(data_dir, out_prefix), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(data_dir, out_prefix), "w", encoding="utf8")

    examples = open(src_file, "r", encoding="utf8").readlines()
    for example in examples:
        # each line is a json object
        example = json.loads(example)
        table_id = example["table_id"]
        table_content = table_content_dict[table_id]
        question = example["question"].lower()
        tapas_table = _convert_table_types(table_content)
        # retrieve wikisql answers by TaPaS script as ground-truth and training labels
        answer = retrieve_wikisql_query_answer_tapas(tapas_table, example)
        if out_prefix == "train":
            # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
            # otherwise, we cannot utilize answer information
            input_source = TABLE_PROCESSOR.process_input(table_content, question, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(table_content, question, []).lower()
        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        input_f.write(input_source + "\n")
        output_f.write(output_target + "\n")

    input_f.close()
    output_f.close()


def build_wikisql_huggingface_dataset(fairseq_data_dir):
    convert_fairseq_to_hf(fairseq_data_dir, "train")
    convert_fairseq_to_hf(fairseq_data_dir, "valid")
    convert_fairseq_to_hf(fairseq_data_dir, "test")


def preprocess_wikisql_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME)
    fairseq_binary_translation(processed_data_dir, resource_name=MODEL_NAME)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    logger.info("*" * 80)
    logger.info("Prepare to download WikiSQL from the official link...")
    wikisql_raw_data_dir = download_wikisql()
    logger.info("Download finished! The original WikiSQL dataset is saved in {}".format(wikisql_raw_data_dir))
    processed_wikisql_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, "wikisql")

    logger.info("*" * 80)
    logger.info("Process the dataset and save the processed dataset in {}".format(processed_wikisql_data_dir))
    build_wikisql_fariseq_dataset("train", os.path.join(wikisql_raw_data_dir, "data", "train.jsonl"),
                                  processed_wikisql_data_dir)
    build_wikisql_fariseq_dataset("valid", os.path.join(wikisql_raw_data_dir, "data", "dev.jsonl"),
                                  processed_wikisql_data_dir)
    build_wikisql_fariseq_dataset("test", os.path.join(wikisql_raw_data_dir, "data", "test.jsonl"),
                                  processed_wikisql_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to BPE and build the dataset binaries in {}/bin".format(processed_wikisql_data_dir))
    preprocess_wikisql_dataset(processed_wikisql_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to build the HuggingFace dataset version in {}".format(processed_wikisql_data_dir))
    build_wikisql_huggingface_dataset(processed_wikisql_data_dir)

    logger.info("*" * 80)
    logger.info("Now you can train models using {} as the <data_dir> argument. "
                "More details in `run_model.py`.".format(os.path.join(processed_wikisql_data_dir, MODEL_NAME)))
