# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import tarfile

from tapex.common.download import download_file
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.processor import get_default_processor
from random import shuffle

PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large (you do not need to further pre-train your models on tapex.base or tapex.large)
MODEL_NAME = "bart.base"
logger = logging.getLogger(__name__)


def download_tapex_pretraining_data():
    """
    Download WikiTableQuestion dataset and unzip the files
    """
    pretrain_url = "https://github.com/microsoft/Table-Pretraining/releases/" \
                   "download/pretraining-corpus/tapex_pretrain.tar.gz"
    pretrain_path = os.path.join(PROCESSED_DATASET_FOLDER, "pretrain")
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    pretrain_gz_file = download_file(pretrain_url)
    # unzip and move it into raw_dataset folder
    tar = tarfile.open(pretrain_gz_file, "r:gz")
    names = tar.getnames()
    for name in names:
        read_f = tar.extractfile(name)
        # if is a file
        if read_f:
            # open a file with the same name
            file_name = os.path.split(name)[-1]
            write_f = open(os.path.join(pretrain_path, file_name), "wb")
            write_f.write(read_f.read())
    tar.close()
    # remove the original file
    os.remove(pretrain_gz_file)
    return pretrain_path


def split_train_valid(data_dir):
    if os.path.exists(os.path.join(data_dir, "valid.src")):
        print("No need to split train/valid on this dataset!")
        return
    # split train/valid if no validation set
    with open(os.path.join(data_dir, "train.src"), "r") as train_src, open(os.path.join(data_dir, "train.tgt"), "r") as train_tgt:
        all_input_lines = train_src.readlines()
        all_output_lines = train_tgt.readlines()
    # process split
    all_zipped_lines = list(zip(all_input_lines, all_output_lines))
    shuffle(all_zipped_lines)
    # we take 20,000 examples to perform validation
    train_lines, valid_lines = all_zipped_lines[:-20000], all_zipped_lines[-20000:]

    # process files
    train_src_out = open(os.path.join(data_dir, "train.src"), "w", encoding="utf8")
    train_tgt_out = open(os.path.join(data_dir, "train.tgt"), "w", encoding="utf8")
    valid_src_out = open(os.path.join(data_dir, "valid.src"), "w", encoding="utf8")
    valid_tgt_out = open(os.path.join(data_dir, "valid.tgt"), "w", encoding="utf8")
    for train_src_line, train_tgt_line in train_lines:
        train_src_out.write(train_src_line.strip() + "\n")
        train_tgt_out.write(train_tgt_line.strip() + "\n")

    for valid_src_line, valid_tgt_line in valid_lines:
        valid_src_out.write(valid_src_line.strip() + "\n")
        valid_tgt_out.write(valid_tgt_line.strip() + "\n")

    train_src_out.close()
    train_tgt_out.close()
    valid_src_out.close()
    valid_tgt_out.close()


def preprocess_pretrain_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME, with_test_set=False)
    fairseq_binary_translation(processed_data_dir, resource_name=MODEL_NAME, with_test_set=False)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    logger.info("*" * 80)
    logger.info("Prepare to download TAPEX Pre-training Corpus from the official link...")
    pretrain_path = download_tapex_pretraining_data()
    logger.info("Download finished! The processed pre-training corpus is saved in {}".format(pretrain_path))

    # split train/valid set
    split_train_valid(pretrain_path)

    logger.info("*" * 80)
    logger.info("Begin to BPE and build the dataset binaries in {}/bin".format(pretrain_path))
    preprocess_pretrain_dataset(pretrain_path)

    logger.info("*" * 80)
    logger.info("Now you can pre-train any generative model using {} as the <data_dir> argument. "
                "More details in `run_model.py`.".format(os.path.join(pretrain_path, MODEL_NAME)))
