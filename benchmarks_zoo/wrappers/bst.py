import re
import copy
from tqdm import tqdm
from datasets import load_dataset

from benchmarks_zoo.wrappers.base import BaseDataset


SKILL_MAP = {
    'convai2': 'personal background',
    'empathetic_dialogues': 'empathy',
    'wizard_of_wikipedia': 'knowledge'
}

SKILL_MAP_THANOS = {
    'Empathy': 'empathy',
    'Personal Background': 'personal background',
    'Persona Recall': 'personal background',
    'Preference Elicitation': 'personal background',
    'Knowledge Sharing': 'knowledge',
    'Knowledge Acquisition': 'knowledge',
    'Knowledge Searching': 'knowledge',
    'Active Listening': 'empathy',
    'Self-disclosure': 'personal background'
}

class BSTDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        return load_dataset("ParlAI/blended_skill_talk", split='test')

    def filtering(self, pre_data):
        data = []
        for idx, instance in enumerate(tqdm(pre_data, desc="Processing Data")):
            prev_utters = instance['previous_utterance']
            free_utters = instance['free_messages']
            guide_utters = instance['guided_messages']
            assert len(free_utters) == len(guide_utters)

            chosen_suggestion = instance['guided_chosen_suggestions']
            if all([ele == '' for ele in chosen_suggestion]):
                continue
            
            dialogue = [
                f'Speaker A: {prev_utters[0]}',
                f'Speaker B: {prev_utters[1]}'
            ]
            final_dialogue = []
            for i, item in enumerate(chosen_suggestion):
                if item != '':
                    cp_dialog = copy.deepcopy(dialogue)
                    final_dialogue.append({
                        'dialogue': cp_dialog,
                        'gt_skill': SKILL_MAP[item]
                    })
                
                if i % 2 == 0:
                    dialogue.append(f'Speaker A: {free_utters[i]}')
                else:
                    dialogue.append(f'Speaker B: {guide_utters[i]}')
                

            for j, item in enumerate(final_dialogue):
                
                input_prompt = self.prompt_template.format(
                    dialogue='\n'.join(cp_dialog),
                    social_context=self.construct_social_context_prompt(),
                    #skill_collection=list(SKILL_MAP.values())
                )
                data.append({
                    'id': f'{idx}:{j}',
                    'prompt_input': input_prompt,
                    'dialogue': item['dialogue'],
                    'gt_skill': item['gt_skill'],
                    'system_message': self.system_message
                })

        print(f"Total conversations processed: {len(data)}")
        return data

    def __getitem__(self, index):
        return self.dataset[index]
    
    def construct_social_context_prompt(self):
        return 'Two speakers are communicate with each other.'

    def evaluate_skill(self, results):
        correct_cnt = 0
        for instance in tqdm(results):
            golden_skill = instance['gt_skill']
            try:
                pred_skill = SKILL_MAP_THANOS[instance['skill']]

                match = re.search(r"'(.*?)'", pred_skill)
                if match:
                    pred_skill = match.group(1)
            except:
                pred_skill = ''
                
            if golden_skill == pred_skill.lower():
                correct_cnt += 1

        acc = (100 * correct_cnt) / len(results)
        return {'acc': acc}

    def evaluate_response(self, results, task_type=None):
        if 'skill' in task_type:
            return self.evaluate_skill(results)
