# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTModel
from omegaconf import open_dict
from tapex.processor import get_default_processor

logger = logging.getLogger(__name__)


class TAPEXModelInterface:
    """
    A simple model interface to tapex for online prediction
    """

    def __init__(self, tapex_model_path, table_processor=None):
        self.model = BARTModel.from_pretrained(tapex_model_path)
        self.model.eval()
        if table_processor is not None:
            self.tab_processor = table_processor
        else:
            self.tab_processor = get_default_processor(max_cell_length=15, max_input_length=1024)

    def predict(self, question: str, table_context: Dict) -> List[str]:
        # process input
        model_input = self.tab_processor.process_input(table_context, question, []).lower()
        model_output = self.model.translate(
            sentences=[model_input],
            beam=5
        )
        return model_output