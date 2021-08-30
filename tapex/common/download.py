# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tarfile
import requests
import shutil
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Resources are obtained and modified from https://github.com/pytorch/fairseq/tree/master/examples/bart
RESOURCE_DICT = {
    "bart.large": "https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
    "tapex.large": "https://github.com/microsoft/Table-Pretraining/releases/download/pretrained-model/tapex.large.tar.gz",
    "bart.base": "https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
    "tapex.base": "https://github.com/microsoft/Table-Pretraining/releases/download/pretrained-model/tapex.base.tar.gz"
}

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


def download_file(url, download_dir=None):
    """
    Download file into local file system from url
    """
    local_filename = url.split('/')[-1]
    if download_dir is None:
        download_dir = os.curdir
    elif not os.path.exists(download_dir):
        os.makedirs(download_dir)
    with requests.get(url, stream=True) as r:
        file_name = os.path.join(download_dir, local_filename)
        if os.path.exists(file_name):
            os.remove(file_name)
        write_f = open(file_name, "wb")
        for data in tqdm(r.iter_content()):
            write_f.write(data)
        write_f.close()

    return os.path.abspath(file_name)


def download_model_weights(resource_dir, resource_name):
    abs_resource_dir = os.path.abspath(resource_dir)
    logger.info("Downloading `model.pt` and `dict.txt` to `{}` ...".format(abs_resource_dir))
    download_url = RESOURCE_DICT[resource_name]
    # download file into resource folder, the file ends with .tar.gz
    file_path = download_file(download_url, resource_dir)
    # unzip files into resource_folder
    tar = tarfile.open(file_path, "r:gz")
    names = tar.getnames()
    for name in names:
        read_f = tar.extractfile(name)
        # if is a file
        if read_f:
            # open a file with the same name
            file_name = os.path.split(name)[-1]
            write_f = open(os.path.join(resource_dir, file_name), "wb")
            write_f.write(read_f.read())
    tar.close()
    logger.info("Copying `dict.txt` to `dict.src.txt` and `dict.tgt.txt` ...")
    # copy dict.txt into dict.src.txt and dict.
    shutil.copy(os.path.join(resource_dir, "dict.txt"),
                os.path.join(resource_dir, "dict.src.txt"))
    shutil.copy(os.path.join(resource_dir, "dict.txt"),
                os.path.join(resource_dir, "dict.tgt.txt"))


def download_bpe_files(resource_dir):
    abs_resource_dir = os.path.abspath(resource_dir)
    logger.info("Downloading `vocab.bpe` and `encoder.json` to `{}` ...".format(abs_resource_dir))
    download_file(DEFAULT_VOCAB_BPE, resource_dir)
    download_file(DEFAULT_ENCODER_JSON, resource_dir)
