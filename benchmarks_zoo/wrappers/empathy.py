import copy
import random
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
import evaluate
import numpy as np

from benchmarks_zoo.wrappers.base import BaseDataset
from utils.common import load_json
from utils.constant import EVAL_OUTPUT_ROOT


class EmpathyDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        if 'thanos_next_resp' in self.config.task_type:
            if self.config.debug:
                path = EVAL_OUTPUT_ROOT / 'thanos_skill' / 'empathy' / 'seed:0' / 'debug' / f'thanos_skill_thanos_{self.config.thanos_model_size}_empathy_results.json'
            else:
                path = EVAL_OUTPUT_ROOT / 'thanos_skill' / 'empathy' / 'seed:0' / f'thanos_skill_thanos_{self.config.thanos_model_size}_empathy_results.json'
            return load_json(path)
        elif 'doctor_next_resp' in self.config.task_type:
            if self.config.debug:
                path = EVAL_OUTPUT_ROOT / 'doctor' / 'empathy' / 'seed:0' / 'debug' / f'doctor_doctor_empathy_results.json'
            return load_json(path)
        else:    
            return load_dataset("facebook/empathetic_dialogues", split='test', trust_remote_code=True)
        
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
        data = defaultdict(list)

        for instance in tqdm(pre_data, desc="Processing Data"):
            conv_id = instance['conv_id']
            utter_id = instance['utterance_idx']
            emotion = instance['context']
            situation = instance['prompt']
            utterance = instance['utterance'].replace('_comma_', ', ')
            
            data[conv_id].append({
                'utterance_idx': utter_id,
                'utterance': utterance,
                'emotion': emotion,
                'situation': situation
            })

        final_data = []
        for conv_index, conv in data.items():
            if len(conv) % 2 == 0:
                dialogue_lines = [
                    f"{'Speaker A' if i % 2 == 0 else 'Speaker B'}: {item['utterance']}"
                    for i, item in enumerate(conv[:-1])
                ]
                flatten_dialogue = '\n'.join(dialogue_lines)

                social_context_prompt = self.construct_social_context_prompt(conv[-1]['emotion'], conv[-1]['situation'])
                input_prompt = self.prompt_template.format(
                    dialogue=flatten_dialogue,
                    social_context=social_context_prompt,
                    next_speaker='Speaker B:'
                )
                final_data.append({
                    'id': conv_index,
                    'prompt_input': input_prompt,
                    'golden_response': conv[-1]['utterance'],
                    'flatten_dialogue': flatten_dialogue,
                    'situation': conv[-1]['situation'],
                    'emotion': conv[-1]['emotion'],
                    'system_message': self.system_message,
                    'social_context': social_context_prompt,
                })

        print(f"Total conversations processed: {len(final_data)}")
        return final_data

    def construct_social_context_prompt(self, emotion: str, situation: str):

        EMPATHY_TEMPLATES = [
            "Speaker A is feeling {emotion} because {situation}.",
            "Due to {situation}, Speaker A's emotion is {emotion}.",
            "Speaker A's emotional state: {emotion}; Situation: {situation}.",
            "Because of {situation}, Speaker A is in a {emotion} mood.",
            "The situation is {situation}, so Speaker A feels {emotion}."
        ]
        random.shuffle(EMPATHY_TEMPLATES)
        selected_template = EMPATHY_TEMPLATES[0]
        social_context_prompt = selected_template.format(emotion=emotion, situation=situation)

        return social_context_prompt

    def __getitem__(self, index):
        return self.dataset[index]

    
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