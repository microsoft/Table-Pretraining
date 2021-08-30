# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A wrapper to wrap the BPE preprocessing procedure for different tasks:
1. TableQA tasks - Translation setting (WikiSQL, WikiTableQuestion, SQA)
2. TableFT tasks - Class setting (TabFact)
"""
import argparse
import logging
import os
import sys

from fairseq.examples.roberta.multiprocessing_bpe_encoder import main as bpe_main

from tapex.common.download import download_model_weights, download_bpe_files

logger = logging.getLogger(__name__)


def get_bpe_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        help="path to save encoded outputs",
    )
    return parser


def setup_translation_bpe_arguments(args, data_dir, resource_dir, prefix, language):
    encoder_json = args.encoder_json if getattr(args, "encoder_json") else os.path.join(resource_dir, "encoder.json")
    vocab_bpe = args.vocab_bpe if getattr(args, "vocab_bpe") else os.path.join(resource_dir, "vocab.bpe")
    inputs = args.inputs if getattr(args, "inputs") else os.path.join(data_dir, "{}.{}".format(prefix, language))
    outputs = args.outputs if getattr(args, "outputs") else os.path.join(data_dir, "{}.bpe.{}".format(prefix, language))
    return {
        "--encoder-json": encoder_json,
        "--vocab-bpe": vocab_bpe,
        "--inputs": inputs,
        "--outputs": outputs
    }


def setup_class_bpe_arguments(args, data_dir, resource_dir, prefix):
    encoder_json = args.encoder_json if getattr(args, "encoder_json") else os.path.join(resource_dir, "encoder.json")
    vocab_bpe = args.vocab_bpe if getattr(args, "vocab_bpe") else os.path.join(resource_dir, "vocab.bpe")
    inputs = args.inputs if getattr(args, "inputs") else os.path.join(data_dir, "{}.raw.input0".format(prefix))
    outputs = args.outputs if getattr(args, "outputs") else os.path.join(data_dir, "{}.input0".format(prefix))
    return {
        "--encoder-json": encoder_json,
        "--vocab-bpe": vocab_bpe,
        "--inputs": inputs,
        "--outputs": outputs
    }


def fairseq_bpe_translation(data_dir, resource_name, resource_dir=None, with_test_set=True):
    """
    BPE script which wrapped the original fairseq BPE for translation tasks.
    The function is equivalent to calling the following script:
    ```
    python -m examples.roberta.multiprocessing_bpe_encoder \
                --encoder-json "$resource_dir/encoder.json" \
                --vocab-bpe "$resource_dir/vocab.bpe" \
                --special-token "$resource_dir/special.txt" \
                --inputs "$data_folder/train.src" \
                --outputs "$data_folder/train.bpe.src" \
                --workers 20 \
                --keep-empty
    ```
    If resource_folder does not exist or contains nothing, we will automatically download BPE files for GPT2.
    :param data_dir: the directory which stores the dataset files, including `train.src`, `train.tgt` and so on.
    :param resource_dir: the cached directory for `resource_name`.
    :param resource_name: corresponding resource files will be automatically downloaded by specifying this parameter.
    You must select one from the choices of `bart.base`, `bart.large`, `tapex.base` and `tapex.large`.
    :param with_test_set: if true, process the test set; otherwise not.
    """
    if resource_dir is None:
        resource_dir = os.path.abspath(resource_name)

    assert resource_name in ["bart.base", "bart.large", "tapex.base", "tapex.large"],\
        "You must specify `download_resource_from` in " \
        "`bart.base`, `bart.large`, `tapex.base` and `tapex.large`."

    if not os.path.exists(os.path.join(resource_dir, "model.pt")):
        # download file into resource folder
        download_model_weights(resource_dir, resource_name)

    if not os.path.exists(os.path.join(resource_dir, "vocab.bpe")):
        download_bpe_files(resource_dir)

    data_files = os.listdir(data_dir)
    assert "train.src" in data_files, "You should prepare your dataset and guarantee a file named `train.src`."
    assert "train.tgt" in data_files, "You should prepare your dataset and guarantee a file named `train.tgt`."
    assert "valid.src" in data_files, "You should prepare your dataset and guarantee a file named `valid.src`."
    assert "valid.tgt" in data_files, "You should prepare your dataset and guarantee a file named `valid.tgt`."
    dataset_prefix = ["train", "valid"]

    if with_test_set:
        assert "test.src" in data_files, "You should prepare your dataset and guarantee a file named `test.src`."
        assert "test.tgt" in data_files, "You should prepare your dataset and guarantee a file named `test.tgt`."
        dataset_prefix.append("test")

    # bpe on files of data_folder
    bpe_parser = get_bpe_parser()
    args = bpe_parser.parse_args()
    for prefix in dataset_prefix:
        for language in ["src", "tgt"]:
            args_command = setup_translation_bpe_arguments(args, data_dir, resource_dir, prefix, language)
            arguments = []
            for key, value in args_command.items():
                arguments.append(key)
                arguments.append(value)
            sys.argv = ["call"] + arguments
            logging.info("BPE files by calling `python -m fairseq.examples.roberta.multiprocessing_bpe_encoder "
                         "{}`".format(" ".join(arguments)))
            bpe_main()
            sys.argv = ["call"]


def fairseq_bpe_classification(data_dir, resource_name, resource_dir=None, with_test_set=True):
    """
    BPE script which wrapped the original fairseq BPE for sentence prediction tasks (i.e., classification).
    The function is equivalent to calling the following script:
    ```
    python -m examples.roberta.multiprocessing_bpe_encoder \
                --encoder-json "$resource_dir/encoder.json" \
                --vocab-bpe "$resource_dir/vocab.bpe" \
                --special-token "$resource_dir/special.txt" \
                --inputs "$data_folder/train.raw.input0" \
                --inputs "$data_folder/train.input0" \
                --workers 20 \
                --keep-empty
    ```
    If resource_folder does not exist or contains nothing, we will automatically download BPE files for GPT2.
    :param data_dir: the directory which stores the dataset files, including `train.src`, `train.tgt` and so on.
    :param resource_dir: the cached folder for `resource_name`.
    :param resource_name: corresponding resource files will be automatically downloaded by specifying this parameter.
    You must select one from the choices of `bart.base`, `bart.large`, `tapex.base` and `tapex.large`.
    :param with_test_set: if true, process the test set; otherwise not.
    """
    if resource_dir is None:
        resource_dir = os.path.abspath(resource_name)

    assert resource_name in ["bart.base", "bart.large", "tapex.base", "tapex.large"], \
        "You must specify `download_resource_from` in " \
        "`bart.base`, `bart.large`, `tapex.base` and `tapex.large`."

    if not os.path.exists(os.path.join(resource_dir, "model.pt")):
        # download file into resource folder
        download_model_weights(resource_dir, resource_name)

    if not os.path.exists(os.path.join(resource_dir, "vocab.bpe")):
        download_bpe_files(resource_dir)

    data_files = os.listdir(data_dir)
    assert "train.raw.input0" in data_files, "You should prepare your dataset and guarantee a file named `train.raw.input0`."
    assert "valid.raw.input0" in data_files, "You should prepare your dataset and guarantee a file named `valid.raw.input0`."
    dataset_prefix = ["train", "valid"]

    if with_test_set:
        assert "test.raw.input0" in data_files, "You should prepare your dataset and guarantee a file named `test.raw.input0`."
        dataset_prefix.append("test")

    # bpe on files of data_folder
    bpe_parser = get_bpe_parser()
    args = bpe_parser.parse_args()
    for prefix in dataset_prefix:
        args_command = setup_class_bpe_arguments(args, data_dir, resource_dir, prefix)
        arguments = []
        for key, value in args_command.items():
            arguments.append(key)
            arguments.append(value)
        sys.argv = ["call"] + arguments
        bpe_main()
        sys.argv = ["call"]
