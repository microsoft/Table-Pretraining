# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import sys
from argparse import ArgumentParser
from fairseq_cli.train import cli_main as fairseq_train
from fairseq_cli.generate import cli_main as fairseq_generate
import logging
import shlex
import re
from tapex.model_interface import TAPEXModelInterface
from fairseq.models.bart import BARTModel
from tapex.model_eval import evaluate_generate_file
import os
from tqdm import tqdm
import torch
import pathlib

logger = logging.getLogger(__name__)


def set_train_parser(parser_group):
    train_parser = parser_group.add_parser("train")
    train_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                              help="dataset directory where train.src is located in")
    train_parser.add_argument("--exp-dir", type=str, default="checkpoints",
                              help="experiment directory which stores the checkpoint weights")
    train_parser.add_argument("--model-path", type=str, default="tapex.base/model.pt",
                              help="the directory of pre-trained model path")
    train_parser.add_argument("--model-arch", type=str, default="bart_base", choices=["bart_large", "bart_base"],
                              help="tapex large should correspond to bart_large, and tapex base should be bart_base")
    train_parser.add_argument("--max-tokens", type=int, default=1800,
                              help="if you train a large model on 16GB memory, max-tokens should be empirically "
                                   "set as 1536, and can be near-linearly increased according to your GPU memory.")
    train_parser.add_argument("--gradient-accumulation", type=int, default=8,
                              help="the accumulation steps to arrive a equal batch size, the default value can be used"
                                   "to reproduce our results. And you can also reduce it to a proper value for you.")
    train_parser.add_argument("--total-num-update", type=int, default=20000,
                              help="the total optimization training steps")
    train_parser.add_argument("--learning-rate", type=float, default=3e-5,
                              help="the peak learning rate for model training")


def set_eval_parser(parser_group):
    eval_parser = parser_group.add_parser("eval")
    eval_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                             help="dataset directory where train.src is located in")
    eval_parser.add_argument("--model-dir", type=str, default="tapex.base.tabfact",
                             help="the directory of fine-tuned model path such as tapex.base.tabfact")
    eval_parser.add_argument("--sub-dir", type=str, default="valid", choices=["train", "valid", "test",
                                                                              "test_complex", "test_simple",
                                                                              "test_small"],
                             help="the directory of pre-trained model path, and the default should be in"
                                  "{bart.base, bart.large, tapex.base, tapex.large}.")


def train_fairseq_model(args):
    cmd = f"""
        fairseq-train {args.dataset_dir} \
        --save-dir {args.exp_dir} \
        --restore-file {args.model_path} \
        --arch {args.model_arch}  \
        --memory-efficient-fp16	\
        --task sentence_prediction \
        --num-classes 2 \
        --add-prev-output-tokens \
        --criterion sentence_prediction \
        --find-unused-parameters \
        --init-token 0 \
        --best-checkpoint-metric accuracy \
        --maximize-best-checkpoint-metric \
        --max-tokens {args.max_tokens}  \
        --update-freq {args.gradient_accumulation} \
        --max-update {args.total_num_update}  \
        --required-batch-size-multiple 1  \
        --dropout 0.1  \
        --attention-dropout 0.1  \
        --relu-dropout 0.0  \
        --weight-decay 0.01  \
        --optimizer adam  \
        --adam-eps 1e-08  \
        --clip-norm 0.1  \
        --lr-scheduler polynomial_decay  \
        --lr {args.learning_rate}  \
        --total-num-update {args.total_num_update}  \
        --warmup-updates 5000  \
        --ddp-backend no_c10d  \
        --num-workers 20  \
        --reset-meters  \
        --reset-optimizer \
        --reset-dataloader \
        --share-all-embeddings \
        --layernorm-embedding \
        --share-decoder-input-output-embed  \
        --skip-invalid-size-inputs-valid-test  \
        --log-format json  \
        --log-interval 10  \
        --save-interval-updates	100 \
        --validate-interval	50 \
        --save-interval	50 \
        --patience 200
    """
    sys.argv = shlex.split(cmd)
    logger.info("Begin to train model for dataset {}".format(args.dataset_dir))
    logger.info("Running command {}".format(re.sub("\s+", " ", cmd.replace("\n", " "))))
    fairseq_train()


def evaluate_fairseq_model(args):
    data_path = pathlib.Path(args.dataset_dir).parent
    bart = BARTModel.from_pretrained(
        args.model_dir,
        data_name_or_path=args.dataset_dir
    )
    bart.eval()

    if torch.cuda.is_available():
        cuda_device = list(range(torch.cuda.device_count()))
        bart = bart.cuda(cuda_device[0])

    call_back_label = lambda label: bart.task.label_dictionary.string(
        [label + bart.task.label_dictionary.nspecial]
    )
    split = args.sub_dir
    input_file, label_file = os.path.join(data_path, "%s.raw.input0" % split), \
                             os.path.join(data_path, "%s.label" % split)
    with open(input_file, 'r', encoding="utf8") as f:
        inputs = f.readlines()
    with open(label_file, 'r', encoding="utf8") as f:
        labels = f.readlines()
    assert len(inputs) == len(labels)
    total, correct = 0, 0
    for input, gold_label in tqdm(zip(inputs, labels)):
        total += 1
        tokens = bart.encode(input)
        pred = call_back_label(bart.predict('sentence_classification_head', tokens).argmax().item())
        if pred == gold_label.strip():
            correct += 1
    logger.info("=" * 20 + "evaluate on {}".format(split) + "=" * 20)
    logger.info(json.dumps({
        "total": total,
        "correct": correct,
        "acc": correct / total
    }))


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    set_train_parser(subparsers)
    set_eval_parser(subparsers)

    args = parser.parse_args()
    if args.subcommand == "train":
        train_fairseq_model(args)
    elif args.subcommand == "eval":
        evaluate_fairseq_model(args)
