import copy
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
import evaluate
import numpy as np

from benchmarks_zoo.wrappers.base import BaseDataset
from utils.constant import EVAL_OUTPUT_ROOT
from utils.common import load_json


class DailyDialogDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        if 'thanos_next_resp' in self.config.task_type:
            if self.config.for_human_eval:
                path = EVAL_OUTPUT_ROOT / 'thanos_skill' / 'dailydialog' / 'seed:0' / 'for_human_eval' / f'thanos_skill_thanos_{self.config.thanos_model_size}_dailydialog_results.json'
            elif self.config.debug:
                path = EVAL_OUTPUT_ROOT / 'thanos_skill' / 'dailydialog' / 'seed:0' / 'debug' / f'thanos_skill_thanos_{self.config.thanos_model_size}_dailydialog_results.json'
            else:
                path = EVAL_OUTPUT_ROOT / 'thanos_skill' / 'dailydialog' / 'seed:0' / f'thanos_skill_thanos_{self.config.thanos_model_size}_dailydialog_results.json'
            return load_json(path)
        elif 'doctor_next_resp' in self.config.task_type:
            if self.config.debug:
                path = EVAL_OUTPUT_ROOT / 'doctor' / 'dailydialog' / 'seed:0' / 'debug' / f'doctor_doctor_dailydialog_results.json'
            return load_json(path)
        else:
            return load_dataset("li2017dailydialog/daily_dialog", split='test', trust_remote_code=True)
    
    def filtering_skill_annot(self, pre_data):
        data = []
        for instance in tqdm(pre_data, desc="Processing Data"):
            social_context = instance['social_context']
            explanation = instance['rationale']
            skill = instance['skill']
            dialogue = instance['flatten_dialogue']

            if self.config.task_type == 'thanos_next_resp_skill':
                input_prompt = self.prompt_template.format(
                    dialogue=dialogue,
                    social_context=social_context,
                    next_speaker='Speaker B:',
                    skill=skill
                )
            elif self.config.task_type == 'thanos_next_resp_explanation':
                input_prompt = self.prompt_template.format(
                    dialogue=dialogue,
                    social_context=social_context,
                    next_speaker='Speaker B:',
                    explanation=explanation,
                )
            elif self.config.task_type == 'thanos_next_resp_both':
                input_prompt = self.prompt_template.format(
                    dialogue=dialogue,
                    social_context=social_context,
                    next_speaker='Speaker B:',
                    explanation=explanation,
                    skill=skill
                )
            cp_instance = copy.deepcopy(instance)
            cp_instance['prompt_input'] = input_prompt
            cp_instance['system_message'] = self.system_message
            data.append(cp_instance)
        
        print(f"Total conversations processed: {len(data)}")
        return data

    def filtering(self, pre_data):
        data = []

        for idx, instance in enumerate(tqdm(pre_data, desc="Processing Data")):
            
            conv = instance['dialog']
            if len(conv) % 2 != 0:
                continue

            act = instance['act']
            emotion = instance['emotion']

            dialogue_lines = [
                f"{'Speaker A' if i % 2 == 0 else 'Speaker B'}: {item}"
                for i, item in enumerate(conv[:-1])
            ]
            #dialogue_lines.append('Speaker B:')
            flatten_dialogue = '\n'.join(dialogue_lines)

            social_context_prompt = self.construct_social_context_prompt()
            
            input_prompt = self.prompt_template.format(
                dialogue=flatten_dialogue,
                social_context=social_context_prompt,
                next_speaker='Speaker B:'
            )

            data.append({
                'id': idx,
                'prompt_input': input_prompt,
                'golden_response': conv[-1],
                'dialogue': conv,
                'flatten_dialogue': flatten_dialogue,
                'emotion': emotion,
                'act': act,
                'system_message': self.system_message,
                'social_context': social_context_prompt,
            })

        print(f"Total conversations processed: {len(data)}")
        return data

    def __getitem__(self, index):
        return self.dataset[index]
    
    def construct_social_context_prompt(self):
        return 'Two speakers are communicate with each other.'

    def evaluate_response(self, results, task_type=None):
        
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

        # bertscore = evaluate.load("bertscore")
        # bertscore_score = round(
        #     np.mean(
        #         bertscore.compute(
        #             predictions=all_predictions,
        #             references=golden_responses,
        #             lang="en"
        #         )["f1"]
        #     ), 4
        # )
        # report['bertscore'] = str(bertscore_score)

        return report