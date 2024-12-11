from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from benchmarks_zoo.wrappers.base import BaseDataset


class PhotoChatDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

    def preload_dataset(self):
        return load_dataset("passing2961/photochat_plus", split='train', trust_remote_code=True)
    
    def filtering(self, pre_data):
        data = []

        for idx, instance in enumerate(tqdm(pre_data, desc="Processing Data")):
            
            dialogue = []
            for i, item in enumerate(instance['dialogue']):
                msg = item['message']
                share_photo = item['share_photo']
                #user_id = item['user_id']
                if share_photo:
                    break

                if i % 2 == 0:
                    dialogue.append(f'Speaker A: {msg}')
                else:
                    dialogue.append(f'Speaker B: {msg}')
            
            input_prompt = self.prompt_template.format(
                dialogue='\n'.join(dialogue),
                social_context=self.construct_social_context_prompt(),
            )
            
            data.append({
                'id': idx,
                'dialogue': dialogue,
                'prompt_input': input_prompt,
                'system_message': self.system_message,
                'gt_skill': 'Image-Sharing'
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
            pred_skill = instance['pred_skill']
            #print(golden_skill, pred_skill)    
            if golden_skill.lower() == pred_skill.lower():
                correct_cnt += 1

        acc = (100 * correct_cnt) / len(results)
        return {'acc': acc}

    def evaluate_response(self, results, task_type=None):
        if 'skill' in task_type:
            return self.evaluate_skill(results)
