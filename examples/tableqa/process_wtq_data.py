# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import zipfile
import logging
from tqdm import tqdm

from tapex.common.download import download_file
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.format_converter import convert_fairseq_to_hf

RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PATH = os.path.join(RAW_DATASET_FOLDER, "wtq")
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large, tapex.base, tapex.large
MODEL_NAME = "tapex.base"
logger = logging.getLogger(__name__)


def download_wikitablequestions():
    """
    Download WikiTableQuestion dataset and unzip the files
    """
    wtq_url = "https://github.com/ppasupat/WikiTableQuestions/releases/" \
              "download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    wtq_raw_path = os.path.join(RAW_DATASET_FOLDER, "wtq")
    wtq_zip_file = download_file(wtq_url)
    # unzip and move it into raw_dataset folder
    with zipfile.ZipFile(wtq_zip_file) as zf:
        zf.extractall(RAW_DATASET_FOLDER)
    unzip_wtq_path = os.path.join(RAW_DATASET_FOLDER, "WikiTableQuestions")
    shutil.move(unzip_wtq_path, wtq_raw_path)
    # remove the original file
    os.remove(wtq_zip_file)
    return wtq_raw_path


def build_wtq_fairseq_dataset(out_prefix, src_file, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def _extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        return _vals

    def _read_table_from_file(_wtq_table_name: str):
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

    input_f = open("{}/{}.src".format(data_dir, out_prefix), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(data_dir, out_prefix), "w", encoding="utf8")

    examples = open(src_file, "r", encoding="utf8").readlines()
    for example in tqdm(examples[1:]):
        _, question, table_name, answer = example.strip("\n").split("\t")
        answer = answer.split("|")
        # must contain rows and header keys
        table_content = _read_table_from_file(table_name)
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


def build_wtq_huggingface_dataset(fairseq_data_dir):
    convert_fairseq_to_hf(fairseq_data_dir, "train")
    convert_fairseq_to_hf(fairseq_data_dir, "valid")
    convert_fairseq_to_hf(fairseq_data_dir, "test")


def preprocess_wtq_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME)
    fairseq_binary_translation(processed_data_dir, resource_name=MODEL_NAME)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    logger.info("*" * 80)
    logger.info("Prepare to download WikiTableQuestions from the official link...")
    wtq_raw_data_dir = download_wikitablequestions()
    logger.info("Download finished! The original WikiTableQuestions dataset is saved in {}".format(wtq_raw_data_dir))
    processed_wtq_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, "wtq")

    logger.info("*" * 80)
    logger.info("Process the dataset and save the processed dataset in {}".format(processed_wtq_data_dir))
    build_wtq_fairseq_dataset("train", os.path.join(wtq_raw_data_dir, "data", "random-split-1-train.tsv"),
                              processed_wtq_data_dir)
    build_wtq_fairseq_dataset("valid", os.path.join(wtq_raw_data_dir, "data", "random-split-1-dev.tsv"),
                              processed_wtq_data_dir)
    build_wtq_fairseq_dataset("test", os.path.join(wtq_raw_data_dir, "data", "pristine-unseen-tables.tsv"),
                              processed_wtq_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to BPE and build the dataset binaries in {}/bin".format(processed_wtq_data_dir))
    preprocess_wtq_dataset(processed_wtq_data_dir)

    logger.info("*" * 80)
    logger.info("Begin to build the HuggingFace dataset version in {}".format(processed_wtq_data_dir))
    build_wtq_huggingface_dataset(processed_wtq_data_dir)

    logger.info("*" * 80)
    logger.info("Now you can train models using {} as the <data_dir> argument. "
                "More details in `run_model.py`.".format(os.path.join(processed_wtq_data_dir, MODEL_NAME)))
