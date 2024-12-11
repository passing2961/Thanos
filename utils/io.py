import os
import yaml
import json

import torch


def read_yaml(path):
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            return None

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_feat(path):
    return torch.load(path)