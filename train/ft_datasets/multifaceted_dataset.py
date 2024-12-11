# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import os
import copy
import json
import random

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import datasets


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

USER_PROMPT_TEMPLATE = """### Task Description:
A dialogue and social context containing the speaker's demographics, preferences, persona, current situation/narrative, past dialogue summaries, episodic memory, or other relevant details are provided. During this dialogue, image-sharing moments may occur, represented by the format "[Sharing Image] <image_description>", where "<image_description>" represents the description of the shared image. Your task is to imagine yourself as the actual speaker who needs to respond in the next conversational turn. You will first generate the internal thought process behind selecting the appropriate conversational skill, and then generate the most appropriate conversational skill itself. The output format should be as follows: "### Explanation: (write an explanation for why the chosen skill is selected.) [RESULT SKILL] (A conversational skill that fits the situation.)"

### Social Context:
{social_context}

### Dialogue:
{conversation}

### Explanation: """


SYSTEM_MESSAGE = """You are an excellent skill predictor that generates the most appropriate conversational skill for the next turn response in the given dialogue and social context. Before generating the skill, please think about which skill is appropriate and then generate the skill."""

        


def create_output_answer(annotation):
    num_skills = len(annotation)
    num_text = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four'}.get(num_skills, str(num_skills))
    
    if num_skills > 1:
        random.shuffle(annotation)
        skill = annotation[0]['skill']
        explanation = annotation[0]['explanation']
    elif num_skills:
        skill = annotation[0]['skill']
        explanation = annotation[0]['explanation']

    output_message = '{} [RESULT SKILL] {}'.format(explanation, skill)
    
    return output_message


def create_message(instance):
    dialogue = ['{}: {}'.format(item['speaker'], item['utter']) for item in instance['dialogue'][:-1]]
    skill_annotation = instance['parsed_generation']
    
    messages = []
    system = {
        "content": SYSTEM_MESSAGE,
        "role": "system"
    }
    messages.append(system)
    user = {
        "content": USER_PROMPT_TEMPLATE.format(
            social_context=instance['social_context_prompt'],
            conversation='\n'.join(dialogue),
            #instruction=INSTRUCTION[0]
        ),
        "role": "user"
    }
    

    messages.append(user)
    
    assistant = {
        "content": create_output_answer(skill_annotation),
        "role": "assistant"
    }
    messages.append(assistant)
    
    #return {'message': messages}
    return messages


def create_prompt(message):
    prompt = TOKENIZER.apply_chat_template(message, add_generation_prompt=False, tokenize=False)

    #return {'text': prompt}
    return prompt

def load_multifaceted_skill_collection_dataset(data_path: str):
    
    merged_dataset = []
    for dataset_name in ['stark', 'syn-personachat', 'wizard_of_wikipedia', 'cactus', 'pearl', 'empathetic_dialogues', 'SODA', 'ConversationChronicles', 'Casino', 'MULTIWOZ2_2', 'Prosocial', 'persuasion']:
        #dataset = load_dataset("json", data_files=f'/home/work/workspace/NAACL2025/main_codes/build_skill_annotation/annotation_results/multifaceted/{dataset_name}/train_data.json', split='train')
        path = os.path.join('/home/work/workspace/NAACL2025/main_codes/build_skill_annotation/annotation_results/multifaceted', dataset_name, 'train_data.json')
        with open(path, 'r') as f:
            dataset = json.load(f)
        print(f"[ {dataset_name} ] dataset size: {len(dataset)}")
        
        merged_dataset.extend(dataset)
    
    print(f"[ Total ] dataset size: {len(merged_dataset)}")
    
    return merged_dataset

class InstructionDataset_MultiFaceted_Skill_Collection(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        global TOKENIZER
        TOKENIZER = tokenizer

        self.ann = load_multifaceted_skill_collection_dataset(dataset_config.data_path)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]

        message = create_message(ann)
        prompt_text = create_prompt(message)

        prompt = prompt_text.split('<|start_header_id|>assistant<|end_header_id|>')[0] + '<|start_header_id|>assistant<|end_header_id|>'
        output_str = prompt_text.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        
        example = prompt + output_str
        
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
