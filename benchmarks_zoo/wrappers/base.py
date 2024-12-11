from abc import abstractmethod
import os

import torch
import numpy as np
import evaluate

from utils.prompt_templates import *


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(BaseDataset, self).__init__()

        self.config = config
        self.prompt_template = self.set_prompt_template(config.task_type)
        self.system_message = self.set_system_message(config.task_type)

        self.dataset = self.load_dataset()
        if self.config.debug:
            #import random
            #random.shuffle(self.dataset)
            self.dataset = self.dataset[:1000]

        if self.config.for_human_eval:
            self.dataset = self.dataset[:100]

    def set_prompt_template(self, task_type):

        template_dict = {
            'base_skill': OUR_BASE_SKILL_TEMPLATE,
            'base_next_resp': OUR_BASE_NEXT_RESP_TEMPLATE,
            'thanos_skill': THANOS_SKILL_TEMPLATE,
            'thanos_next_resp_skill': THANOS_NEXT_RESP_TEMPLATE_SKILL,
            'thanos_next_resp_explanation': THANOS_NEXT_RESP_TEMPLATE_RATIONALE,
            'thanos_next_resp_both': THANOS_NEXT_RESP_TEMPLATE_BOTH,
            'doctor': '',
            'doctor_next_resp': DOCTOR_CHATGPT_TEMPLATE
        }
        return template_dict[task_type]

    def set_system_message(self, task_type):
        system_message_dict = {
            'base_skill': OUR_BASE_SKILL_SYSTEM_MESSAGE,
            'base_next_resp': OUR_BASE_NEXT_RESP_SYSTEM_MESSAGE,
            'thanos_skill': THANOS_SKILL_SYSTEM_MESSAGE,
            'thanos_next_resp_skill': THANOS_NEXT_RESP_SYSTEM_MESSAGE,
            'thanos_next_resp_explanation': THANOS_NEXT_RESP_SYSTEM_MESSAGE,
            'thanos_next_resp_both': THANOS_NEXT_RESP_SYSTEM_MESSAGE,
            'doctor': '',
            'doctor_next_resp': 'You are a helpful assistant.'
        }
        return system_message_dict[task_type]

    def load_dataset(self):
        pre_data = self.preload_dataset()
        if 'thanos_next_resp' in self.config.task_type:
            return self.filtering_skill_annot(pre_data)
        elif 'doctor_next_resp' in self.config.task_type:
            return self.filtering_doctor(pre_data)
        else:
            return self.filtering(pre_data)
    
    @abstractmethod
    def filtering(self):
        raise NotImplementedError

    @abstractmethod
    def preload_dataset(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def evaluate_response(self, results):

        all_predictions = [ele['model_response'] for ele in results]
        golden_responses = [ele['golden_response'] for ele in results]

        report = dict()

        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(
            predictions=all_predictions,
            references=golden_responses,
            use_aggregator=False
        )
        for k, v in rouge_score.items():
            report[k] = str(round(np.mean(v), 4))

        bleu = evaluate.load("bleu")
        for i in range(1, 5):
            bleu_score = bleu.compute(
                predictions=all_predictions,
                references=golden_responses,
                max_order=i
            )['bleu']
            report[f'bleu{i}'] = str(round(bleu_score, 4))

        bertscore = evaluate.load("bertscore")
        bertscore_score = round(
            np.mean(
                bertscore.compute(
                    predictions=all_predictions,
                    references=golden_responses,
                    lang="en"
                )["f1"]
            ), 4
        )
        report['bertscore'] = str(bertscore_score)

        return report



