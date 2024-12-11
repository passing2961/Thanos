from datasets import load_dataset

from benchmarks_zoo.wrapppers.base import BaseDataset


class PersonaChatDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        return