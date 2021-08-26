# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A wrapper to wrap the fairseq preprocessing procedure.
"""
import os

from fairseq import options
from fairseq_cli import preprocess


def setup_translation_binary_arguments(args, data_dir, resource_dir, with_test_set):
    args.source_lang = args.source_lang if getattr(args, "source_lang") else "src"
    args.target_lang = args.target_lang if getattr(args, "target_lang") else "tgt"
    args.trainpref = args.trainpref if getattr(args, "trainpref") else os.path.join(data_dir, "train.bpe")
    args.validpref = args.validpref if getattr(args, "validpref") else os.path.join(data_dir, "valid.bpe")
    args.destdir = args.destdir if getattr(args, "destdir") not in [None, "data-bin"] else os.path.join(data_dir, "bin")
    args.srcdict = args.srcdict if getattr(args, "srcdict") else os.path.join(resource_dir, "dict.src.txt")
    args.tgtdict = args.tgtdict if getattr(args, "tgtdict") else os.path.join(resource_dir, "dict.tgt.txt")
    args.workers = args.workers if getattr(args, "workers") else 20

    if with_test_set:
        args.testpref = args.testpref if getattr(args, "testpref") else os.path.join(data_dir, "test.bpe")


def setup_class_input_binary_arguments(args, data_dir, resource_dir, with_test_set):
    args.only_source = args.only_source if getattr(args, "only_source") else True
    args.trainpref = args.trainpref if getattr(args, "trainpref") else os.path.join(data_dir, "train.input0")
    args.validpref = args.validpref if getattr(args, "validpref") else os.path.join(data_dir, "valid.input0")
    args.destdir = args.destdir if getattr(args, "destdir") not in [None, "data-bin"] else os.path.join(data_dir, "input0")
    args.srcdict = args.srcdict if getattr(args, "srcdict") else os.path.join(resource_dir, "dict.src.txt")
    args.tgtdict = args.tgtdict if getattr(args, "tgtdict") else os.path.join(resource_dir, "dict.tgt.txt")
    args.workers = args.workers if getattr(args, "workers") else 20

    if with_test_set:
        args.testpref = args.testpref if getattr(args, "testpref") else os.path.join(data_dir, "test.input0")


def setup_class_label_binary_arguments(args, data_dir, resource_dir, with_test_set):
    args.only_source = args.only_source if getattr(args, "only_source") else True
    args.trainpref = args.trainpref if getattr(args, "trainpref") else os.path.join(data_dir, "train.label")
    args.validpref = args.validpref if getattr(args, "validpref") else os.path.join(data_dir, "valid.label")
    args.destdir = args.destdir if getattr(args, "destdir") else os.path.join(data_dir, "label")
    args.srcdict = args.srcdict if getattr(args, "srcdict") else os.path.join(resource_dir, "dict.src.txt")
    args.tgtdict = args.tgtdict if getattr(args, "tgtdict") else os.path.join(resource_dir, "dict.tgt.txt")
    args.workers = args.workers if getattr(args, "workers") else 20

    if with_test_set:
        args.testpref = args.testpref if getattr(args, "testpref") else os.path.join(data_dir, "test.input0")


def fairseq_binary_translation(data_dir, resource_name, resource_dir=None, with_test_set=True):
    """
    Execute fairseq using default arguments, more arguments can be seen at
     https://fairseq.readthedocs.io/en/latest/
    :return:
    """
    if resource_dir is None:
        resource_dir = resource_name

    # preprocess data_folder
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    setup_translation_binary_arguments(args, data_dir, resource_dir, with_test_set)
    # pass by args and preprocess the dataset
    preprocess.main(args)


def fairseq_binary_classification(data_dir, resource_name, resource_dir=None, with_test_set=True):
    """
    Execute fairseq using default arguments, more arguments can be seen at
    https://fairseq.readthedocs.io/en/latest/
    :return:
    """
    if resource_dir is None:
        resource_dir = resource_name

    # preprocess data_folder
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    setup_class_input_binary_arguments(args, data_dir, resource_dir, with_test_set)
    preprocess.main(args)
    setup_class_label_binary_arguments(args, data_dir, resource_dir, with_test_set)
    preprocess.main(args)
