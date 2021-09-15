# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List

import torch.cuda
from fairseq.models.bart import BARTModel

from tapex.processor import get_default_processor

logger = logging.getLogger(__name__)


class TAPEXModelInterface:
    """
    A simple model interface to tapex for online prediction
    """

    def __init__(self, resource_dir, checkpoint_name, table_processor=None):
        self.model = BARTModel.from_pretrained(model_name_or_path=resource_dir,
                                               checkpoint_file=checkpoint_name)
        if torch.cuda.is_available():
            self.model.cuda()
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
        # the result should be a list of answers, and we only care about the answer itself instead of score
        return model_output[0][0]
