# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from .grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from .alpaca_dataset import InstructionDataset as get_alpaca_dataset
from .custom_dataset import get_custom_dataset,get_data_collator
from .samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from .toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from .multifaceted_dataset import InstructionDataset_MultiFaceted_Skill_Collection as get_multifaceted_skill_collection_dataset

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "multifaceted_skill_collection_dataset": get_multifaceted_skill_collection_dataset
}

DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}
