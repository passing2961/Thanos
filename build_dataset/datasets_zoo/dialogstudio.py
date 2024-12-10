import json
import random
from tqdm import tqdm

from datasets import load_dataset
from rich.console import Console

from .base import BaseDataset
    

class DialogStudioDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.load_data()  
    
    def load_data(self):
        self.dataset = load_dataset('Salesforce/dialogstudio', self.args.dataset_name, split=self.args.split)
        
        Console().print(
            f"[ {self.args.dataset_name} ] # of {self.args.split} dataset: {len(self.dataset)}"
        )

class ConvAI2Dataset(DialogStudioDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.speaker_dict = {'Bot': 'Speaker A', 'Human': 'Speaker B'}
        
    def construct_social_context_prompt(self, user_profile, bot_profile):
        return "Speaker A's Persona Information: {}\n\nSpeaker B's Persona Information: {}".format(' '.join([ele.capitalize() for ele in bot_profile]), ' '.join([ele.capitalize() for ele in user_profile]))
    
    def process_data(self):
        final_dataset = []
        from collections import defaultdict
        len_count = defaultdict(int)
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            original_info = json.loads(instance['original dialog info'])

            bot_profile = original_info['bot_profile']
            user_profile = original_info['user_profile']
            
            dialogue = [
                utterance
                for item in instance['log']
                for utterance in [item['user utterance'], item['system response']]
            ]
            if dialogue[-1] == '':
                dialogue = dialogue[:-1]
            
            if len(dialogue) > 12:
                continue

            speaker_list = []
            for item in instance['log']:
                user_info = json.loads(item['original user side information'])['sender_class']
                speaker_list.append(self.speaker_dict[user_info])
                try:
                    system_info = json.loads(item['original system side information'])['sender_class']
                    speaker_list.append(self.speaker_dict[system_info])
                except KeyError:
                    pass
                
            if len(speaker_list) != len(dialogue):
                print(dialogue)
                print(speaker_list)
                assert False
                continue
            
            dialogue_dict = [
                {'speaker': spk, 'utter': utt} 
                for spk, utt in zip(speaker_list, dialogue)
            ]
            len_count[len(dialogue_dict)] += 1
            final_dataset.append({
                'index': instance['new dialog id'],
                'dialogue': dialogue_dict,
                'social_context_prompt': self.construct_social_context_prompt(user_profile, bot_profile)
            })

        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset
    

class WoWDataset(DialogStudioDataset):
    def __init__(self, args):
        super().__init__(args)

        self.speaker_dict = {'Wizard': 'Speaker B', 'Apprentice': 'Speaker A'}

    def construct_social_context_prompt(self, original_info):
        chosen_topic = original_info['chosen_topic']
        persona = original_info['persona']
        if persona == '':
            assert False

        TEMPLATE = [
            "The chosen topic is {chosen_topic}. Speaker A is curious, and Speaker B is eager to discuss it with them.",
            "Speaker A has selected {chosen_topic} as the topic. Speaker A is curious, and Speaker B is enthusiastic about the discussion.",
            "Speaker A chose {chosen_topic} to talk about. Speaker A shows curiosity, and Speaker B is keen to engage in conversation.",
            "The chosen topic is {chosen_topic}. Speaker A is curious, and Speaker B looks forward to discussing it.",
            "Speaker A has picked {chosen_topic} as the topic. Speaker A is curious, and Speaker B is eager to converse with them."
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(persona=persona, chosen_topic=chosen_topic)

        return social_context_prompt
    
    def get_speaker_name(self, speaker_name: str):
        if 'Wizard' in speaker_name:
            return 'Wizard'
        elif 'Apprentice' in speaker_name:
            return 'Apprentice'
        else:
            raise ValueError(f'{speaker_name}')

    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            original_info = json.loads(instance['original dialog info'])
            
            dialogue = []
            for item in instance['log']:
                user_info = json.loads(item['original user side information'])
                system_info = json.loads(item['original system side information'])
                user_utter = item['user utterance']
                system_utter = item['system response']

                if user_utter != '':
                    dialogue.append({
                        'speaker': self.speaker_dict[self.get_speaker_name(user_info['speaker'])],
                        'utter': item['user utterance']
                    })
                
                if system_utter != '':
                    dialogue.append({
                        'speaker': self.speaker_dict[self.get_speaker_name(system_info['speaker'])],
                        'utter': item['system response']
                    })

            if dialogue[-1]['utter'] == '':
                assert False

            final_dataset.append({
                'index': instance['new dialog id'],
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(original_info)
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset


class MultiWoZDataset(DialogStudioDataset):
    def __init__(self, args):
        super().__init__(args)

    def construct_social_context_prompt(self, original_info):
        service = original_info['services']
        if not service:
            return ''
        
        if len(service) > 1:
            service = ', '.join(service[:-1]) + ' and ' + service[-1]
        else:
            service = service[0]

        TEMPLATE = [
            "The conversation between two speakers is about {service}.",
            "Two speakers are discussing the domain of {service}.",
            "Both speakers engage in a conversation on {service}.",
            "The topic of conversation for the two speakers is {service}.",
            "Two speakers have a discussion focused on {service}."
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(service=service)
        
        return social_context_prompt
    
    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            original_info = json.loads(instance['original dialog info'])
            
            dialogue = []
            for item in instance['log']:
                user_utter = item['user utterance']
                system_utter = item['system response']

                if user_utter != '':
                    dialogue.append({
                        'speaker': 'Speaker A',
                        'utter': item['user utterance']
                    })
                
                if system_utter != '':
                    dialogue.append({
                        'speaker': 'Speaker B',
                        'utter': item['system response']
                    })

            if dialogue[-1]['utter'] == '':
                assert False

            social_context_prompt = self.construct_social_context_prompt(original_info)
            if social_context_prompt == '':
                continue

            final_dataset.append({
                'index': instance['new dialog id'],
                'dialogue': dialogue,
                'social_context_prompt': social_context_prompt
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset


class SodaDataset(DialogStudioDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def construct_social_context_prompt(self, narrative):
        return narrative
    
    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            original_info = json.loads(instance['original dialog info'])
            speaker_list = original_info['speakers']
            
            dialogue = [
                utterance 
                for item in instance['log']
                for utterance in [item['user utterance'], item['system response']]
            ]
            
            if len(speaker_list) != len(dialogue):
                continue
            
            dialogue_dict = [
                {'speaker': spk, 'utter': utt} 
                for spk, utt in zip(speaker_list, dialogue)
            ]
            
            final_dataset.append({
                'index': instance['new dialog id'],
                'dialogue': dialogue_dict,
                'social_context_prompt': self.construct_social_context_prompt(original_info['narrative'])
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset


class ProsocialDataset(DialogStudioDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def construct_social_context_prompt(self, user_info):
        user_info = json.loads(user_info)
        rots = user_info['rots']
        if len(rots) == 0:
            assert False
        rots_formatted = '\n- '.join(rots)

        TEMPLATE = [
            "Speaker B should foster prosocial behavior by providing constructive feedback based on these Rule-of-Thumbs:\n- {rots_formatted}",
            "Speaker B should encourage prosocial behavior by giving constructive feedback based on these Rule-of-Thumbs:\n- {rots_formatted}",
            "To promote positive behavior, Speaker B should offer constructive feedback following these Rule-of-Thumbs:\n- {rots_formatted}",
            "Guided by these Rule-of-Thumbs, Speaker B should encourage prosocial behavior through constructive feedback:\n- {rots_formatted}",
            "Speaker B is expected to provide constructive feedback to encourage positive interactions, using these Rule-of-Thumbs:\n- {rots_formatted}",
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(rots_formatted=rots_formatted)
        
        return social_context_prompt
    
    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            original_info = json.loads(instance['original dialog info'])
            
            dialogue = []
            for item in instance['log']:
                user_utter = item['user utterance']
                system_utter = item['system response']

                user_info = item["original user side information"]
                if user_utter != '':
                    dialogue.append({
                        'speaker': 'Speaker A',
                        'utter': item['user utterance']
                    })
                
                if system_utter != '':
                    dialogue.append({
                        'speaker': 'Speaker B',
                        'utter': item['system response']
                    })

            if dialogue[-1]['utter'] == '':
                assert False
            
            final_dataset.append({
                'index': instance['new dialog id'],
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(user_info)
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset
