import os
import gc
import re
import json
import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Check for duplicated questions, items
def remove_duplicate(benchmark, inputs, generations):
    if benchmark == "mme": 
        return inputs, generations
    elif benchmark == "pope": 
        questions = set()
        new_inputs, new_answers = [], []
        for i, a in zip(inputs, generations):
            dup = i['id'], i['category']
            if dup in questions:
                continue
            questions.add(dup)
            new_inputs.append(i)
            new_answers.append(a)
    else:
        questions = set()
        new_inputs, new_answers = [], []
        for i, a in zip(inputs, generations):
            if i['id'] in questions:
                continue
            questions.add(i['id'])
            new_inputs.append(i)
            new_answers.append(a)
    return new_inputs, new_answers

def memory_optimization():
    # memory deallocation
    gc.collect()

    # removing cache
    torch.cuda.empty_cache()

