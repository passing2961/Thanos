from abc import ABC, abstractmethod
import copy
from tqdm import tqdm

from rich.console import Console

from utils.common import generate_incremental_sub_lists


class BaseDataset(ABC):
    def __init__(self, args):
        self.args = args
    
    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def process_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def construct_social_context_prompt(self):
        raise NotImplementedError
    
    def prepare_data(self):
        processed_data = self.process_data()
        
        final_dataset = []
        for instance in tqdm(processed_data, total=len(processed_data)):
            if self.args.dataset_name == 'Janus':
                dialogue = [instance['dialogue']]
            elif self.args.dataset_name == 'stark':
                temp_dialogue = []
                current_dialogue = []

                for item in instance['dialogue']:
                    utt = item['utter']
                    if '[Sharing Image' in utt:
                        cp_dialog = copy.deepcopy(current_dialogue)
                        cp_dialog.append(item)
                        temp_dialogue.append(cp_dialog)
                            
                    current_dialogue.append(item)

                if current_dialogue:
                    temp_dialogue.append(current_dialogue)
                
                dialogue = []
                for ele in temp_dialogue:
                    if len(ele) % 2 == 0:
                        dialogue.append(ele)
                
            else:
                dialogue = generate_incremental_sub_lists(instance['dialogue'])
            original_index = instance['index']
            
            for idx, item in enumerate(dialogue):
                if self.args.dataset_name != 'Janus':
                    if self.args.multi_turn_filter:
                        if len(item) < self.args.turn_num_threshold:
                            continue
                cp_instance = copy.deepcopy(instance)
                cp_instance['index'] = f'{idx}:{original_index}'
                cp_instance['dialogue'] = item
                final_dataset.append(cp_instance)
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] # of dataset: {len(final_dataset)}"
        )
        return final_dataset