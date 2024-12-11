import os
import re
import copy
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
import numpy as np

from benchmarks_zoo.wrappers.base import BaseDataset
from utils.common import load_json


class OursDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        merged_dataset = []
        for dataset_name in ['stark', 'syn-personachat', 'wizard_of_wikipedia', 'cactus', 'pearl', 'empathetic_dialogues', 'SODA', 'ConversationChronicles', 'Casino', 'MULTIWOZ2_2', 'Prosocial', 'persuasion']:
            path = os.path.join('build_skill_annotation/annotation_results/multifaceted', dataset_name, 'test_data.json')
            dataset = load_json(path)
            merged_dataset.extend(dataset)

        return merged_dataset

    def filtering(self, pre_data):
        data = []
        for idx, instance in enumerate(tqdm(pre_data, desc="Processing Data")):
            social_context = instance['social_context_prompt']
            dialogue = instance['dialogue']
            annotation = instance['parsed_generation']
            if len(annotation) > 1:
                continue
            flatten_dialogue = ['{}: {}'.format(ele['speaker'], ele['utter']) for ele in dialogue]
            flatten_dialogue = '\n'.join(flatten_dialogue)
            explanation = annotation[0]['explanation']
            skill = annotation[0]['skill']

            input_prompt = self.prompt_template.format(
                dialogue=flatten_dialogue,
                social_context=social_context
            )

            cp_instance = copy.deepcopy(instance)
            cp_instance['id'] = instance['index']
            cp_instance['prompt_input'] = input_prompt
            cp_instance['system_message'] = self.system_message
            cp_instance['gt_skill'] = skill
            cp_instance['gt_explanation'] = explanation
            data.append(cp_instance)
        
        print(f"Total conversations processed: {len(data)}")
        return data
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def evaluate_response(self, results, task_type=None):
        
        total_correct_cnt = 0
        fine_correct_cnt = defaultdict(list)
        pred_explanations = []
        gt_explanations = []
        for instance in tqdm(results):
            golden_skill = instance['gt_skill']
            pred_skill = instance['pred_skill']

            golden_explanation = instance['gt_explanation']
            pred_explanation = instance['pred_explanation']
            
            pred_explanations.append(pred_explanation)
            gt_explanations.append(golden_explanation)

            if golden_skill.lower() == pred_skill.lower():
                total_correct_cnt += 1
                fine_correct_cnt[golden_skill].append(1)
            else:
                fine_correct_cnt[golden_skill].append(0)
        report = dict()

        report['all:acc'] = (100 * total_correct_cnt) / len(results)

        for k, v in fine_correct_cnt.items():
            report[f'{k}:acc'] = (100 * sum(v)) / len(v)
        
        import evaluate

        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(
            predictions=pred_explanations,
            references=gt_explanations,
            use_aggregator=False
        )
        for k, v in rouge_score.items():
            report[k] = str(round(np.mean(v), 4))

        bleu = evaluate.load("bleu")
        for i in range(1, 5):
            bleu_score = bleu.compute(
                predictions=pred_explanations,
                references=gt_explanations,
                max_order=i
            )['bleu']
            report[f'bleu{i}'] = str(round(bleu_score, 4))

        # bertscore = evaluate.load("bertscore")
        # bertscore_score = round(
        #     np.mean(
        #         bertscore.compute(
        #             predictions=pred_explanations,
        #             references=gt_explanations,
        #             lang="en"
        #         )["f1"]
        #     ), 4
        # )
        # report['bertscore'] = str(bertscore_score)

        return report