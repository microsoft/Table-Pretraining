# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import zipfile
import logging
from tqdm import tqdm
import pandas as pd
from tapex.common.download import download_file
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.format_converter import convert_fairseq_to_hf

RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PATH = os.path.join(RAW_DATASET_FOLDER, "sqa")
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large, tapex.base, tapex.large
MODEL_NAME = "tapex.base"
logger = logging.getLogger(__name__)


def download_sqa():
    """
    Download WikiSQL dataset and unzip the files
    """
    sqa_url = "https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/SQA%20Release%201.0.zip"
    sqa_raw_path = os.path.join(RAW_DATASET_FOLDER, "sqa")
    sqa_zip_file = download_file(sqa_url)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(sqa_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "SQA Release 1.0")
    shutil.move(unzip_wtq_path, sqa_raw_path)
    # remove the original file
    os.remove(sqa_zip_file)
    return sqa_raw_path


def build_sqa_fairseq_dataset(out_prefix, src_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def _read_table_from_file(_sqa_table_name: str):
        rows = []
        assert ".csv" in _sqa_table_name
        table_data = pd.read_csv(os.path.join(TABLE_PATH, _sqa_table_name))
        # the first line is header
        header = list(table_data.columns)
        for row_data in table_data.values:
            rows.append([str(_) for _ in list(row_data)])

        return {
            "header": header,
            "rows": rows
        }

    input_f = open("{}/{}.src".format(data_dir, out_prefix), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(data_dir, out_prefix), "w", encoding="utf8")

    examples = open(src_file, "r", encoding="utf8").readlines()
    history = ""
    idx = 1
    for example in tqdm(examples[1:]):
        try:
            anno_id, _, position, question, table_file, _, answer_text = example.strip("\n").split("\t")
            answer_text = answer_text.replace("\"\"", "\"").strip("\"'")
            # must contain rows and header keys
            if position == "0":
                # reset history
                history = ""
            question = question.lower()
            if history:
                question = history + " " + question
            answer = eval(answer_text)
            table_content = _read_table_from_file(table_file)
            if out_prefix == "train":
                # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
                # otherwise, we cannot utilize answer information
                input_source = TABLE_PROCESSOR.process_input(table_content, question, answer).lower()
            else:
                input_source = TABLE_PROCESSOR.process_input(table_content, question, []).lower()
            output_target = TABLE_PROCESSOR.process_output(answer).lower()
            input_f.write(input_source + "\n")
            output_f.write(output_target + "\n")
            # reset the history
            history = question
            idx += 1
        except:
            logger.error("Error case on Line: {}, {}".format(idx, question))
    input_f.close()
    output_f.close()


def build_sqa_huggingface_dataset(fairseq_data_dir):
    convert_fairseq_to_hf(fairseq_data_dir, "train")
    convert_fairseq_to_hf(fairseq_data_dir, "valid")
    convert_fairseq_to_hf(fairseq_data_dir, "test")


def preprocess_sqa_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME)
    fairseq_binary_translation(processed_data_dir, resource_name=MODEL_NAME)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    logger.info("*" * 80)
    logger.info("Prepare to download SQA dataset from the official link...")
    sqa_raw_data_dir = download_sqa()
    logger.info("Download finished! The original WikiTableQuestions dataset is saved in {}".format(sqa_raw_data_dir))
    processed_sqa_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, "sqa")

    logger.info("*" * 80)
    logger.info("Process the dataset and save the processed dataset in {}".format(processed_sqa_data_dir))
    build_sqa_fairseq_dataset("train", os.path.join(sqa_raw_data_dir, "random-split-1-train.tsv"),
                              processed_sqa_data_dir)
    build_sqa_fairseq_dataset("valid", os.path.join(sqa_raw_data_dir, "random-split-1-dev.tsv"),
                              processed_sqa_data_dir)
    build_sqa_fairseq_dataset("test", os.path.join(sqa_raw_data_dir, "test.tsv"),
                              processed_sqa_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to BPE and build the dataset binaries in {}/bin".format(processed_sqa_data_dir))
    preprocess_sqa_dataset(processed_sqa_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to build the HuggingFace dataset version in {}".format(processed_sqa_data_dir))
    build_sqa_huggingface_dataset(processed_sqa_data_dir)

    logger.info("*" * 80)
    logger.info("Now you can train models using {} as the <data_dir> argument. "
                "More details in `run_model.py`.".format(os.path.join(processed_sqa_data_dir, MODEL_NAME)))
