import os
import json
import random
from tqdm import tqdm

from collections import defaultdict
from datasets import load_dataset
from rich.console import Console

from .base import BaseDataset
from utils.common import generate_incremental_sub_lists
from utils.prompt_template import CASINO_TEMPLATE_STRUCT, CASINO_TEMPLATE_SENTENCE
    

class HuggingFaceDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        
    def load_data(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name, split=self.args.split)
        
        Console().print(
            f"[ {self.args.dataset_name} ] # of {self.args.split} dataset: {len(self.dataset)}"
        )


class ConversationChroniclesDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("jihyoung/ConversationChronicles")
    
    def construct_social_context_prompt(self, relationship: str, time_interval: str, prev_summary: str):
        if time_interval == 'Start':
            social_context_prompt = f'Two speakers have {relationship} relationship.'
        else:
            if 'after' not in time_interval:
                time_interval = time_interval + ' before'
                social_context_prompt = "Two speakers have {} relationship. {}, {}".format(relationship, time_interval, prev_summary)
            else:
                social_context_prompt = "Two speakers have {} relationship. {}, {}".format(relationship, time_interval.replace('after', 'before'), prev_summary)

        return social_context_prompt
        
    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            
            # Verify that the length of `time_interval` and `summary` are equal and equal to 5
            self._validate_lengths(instance['time_interval'], instance['summary'])
            
            # Extract session data
            for session_num in range(1, 6):
                session_data = self._extract_session_data(instance, session_num)
            
                data = {
                    'index': '{}:session{}'.format(instance['dataID'], session_num),
                    'relationship': instance['relationship'],
                    'time_interval': instance['time_interval'][session_num-1],
                    'current_summary': instance['summary'][session_num-1],
                    'prev_summary': instance['summary'][session_num-2] if session_num >= 2 else ''
                }
                social_context_prompt = self.construct_social_context_prompt(data['relationship'], data['time_interval'], data['prev_summary'])
                
                data.update({'social_context_prompt': social_context_prompt})
                data.update(session_data)

                final_dataset.append(data)
            
        Console().print(
            f"[ {self.args.dataset_name} ] # of {self.args.split} dataset: {len(final_dataset)}"
        )
        return final_dataset
    
    def _validate_lengths(self, time_interval, summary):
        """Validate that `time_interval` and `summary` have a length of 5."""
        assert len(time_interval) == len(summary) == 5, "Length of time_interval and summary must be 5"
    
    def _extract_session_data(self, instance, session_num):
        """Extract dialogue and speaker data for a given session."""
        session_data = {}
        session_prefix = f"{session_num}_session"
        
        session_data[f"{session_prefix}_dialogue"] = instance[f'{self._ordinal(session_num)}_session_dialogue']
        session_data[f"{session_prefix}_speakers"] = instance[f'{self._ordinal(session_num)}_session_speakers']
        
        session_data["dialogue"] = [
            {'speaker': spk, 'utter': utt} 
            for spk, utt in zip(session_data[f"{session_prefix}_speakers"], session_data[f"{session_prefix}_dialogue"])
        ]
        
        return session_data
    
    @staticmethod
    def _ordinal(n):
        """Return the ordinal string (e.g., 'first', 'second') for a given number."""
        return ["first", "second", "third", "fourth", 'fifth'][n-1]
         


class StarkDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        # Since we already have the STARK dataset in our local directory and are using it here,
        # we implemented a custom dataset loader specific to our setup.
        # For others who want to use this dataset, a different dataset loader using Hugging Face Hub should be implemented.
        # In the future, we plan to update this to include support for more flexible loading methods.
        
        stark_root_dir = '<LOCAL-STARK-DIR>'

        self.dataset = []
        path = os.path.join(stark_root_dir, 'stark.json')
        with open(path, 'r') as f:
            self.dataset = json.load(f)

        ###########################################################
        Console().print(
            f"[ {self.args.dataset_name} ] # of {self.args.split} dataset: {len(self.dataset)}"
        )

    def construct_social_context_prompt(self, session_num, name, age, gender, birthplace, residence, experience, time_interval, event):
        
        if session_num == 1:
            TEMPLATE = [
                '{name} is {age} years old, born in {birthplace}, and currently lives in {residence}. {event}',
                '{name}, aged {age}, was born in {birthplace} and resides in {residence}. {event}',
                '{name}, who is {age}, was born in {birthplace} and now lives in {residence}. {event}',
                '{name} is {age}, originally from {birthplace}, and now living in {residence}. {event}',
                '{name} is {age} years old, born in {birthplace}, and resides in {residence}. {event}'
            ]
            random.shuffle(TEMPLATE)
            selected_template = TEMPLATE[0]

            social_context_prompt = selected_template.format(name=name, age=age, birthplace=birthplace, residence=residence, event=event)
            
        else:
            TEMPLATE = [
                '{name} is {age} years old, born in {birthplace}, and currently lives in {residence}. After {time_interval}, {name} has gone through {experience}, and now {event}',
                '{name}, aged {age}, was born in {birthplace} and now resides in {residence}. Following {time_interval}, {name} experienced {experience}, and {event}',
                '{name}, who is {age} years old, originally from {birthplace} and living in {residence}, went through {experience} after {time_interval}, and now {event}',
                '{name} is {age}, born in {birthplace}, and currently resides in {residence}. After {time_interval} of {experience}, {name} has now {event}',
                '{name}, {age} years old, from {birthplace} and residing in {residence}, has experienced {experience} over {time_interval}, and as a result, {event}'
            ]
            random.shuffle(TEMPLATE)
            selected_template = TEMPLATE[0]

            social_context_prompt = selected_template.format(name=name, age=age, birthplace=birthplace, residence=residence, event=event, time_interval=time_interval, experience=experience)
            
        return social_context_prompt
        
    def process_data(self):
        final_dataset = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            
            if instance['number_of_session'] != 5:
                continue
            
            # Extract session data
            for session_num in range(1, 6):
                session_data, share_flag = self._extract_session_data(instance, session_num)
                if not share_flag:
                    continue

                data = {
                    'index': '{}:session{}'.format(instance['unique_id'], session_num),
                    'time_interval': instance[f'session{session_num}:time_interval'],
                }
                social_context_prompt = self.construct_social_context_prompt(session_num, instance['name'], instance['age'], instance['gender'], instance['birthplace'],
                    instance['residence'], instance[f'session{session_num}:experience'], instance[f'session{session_num}:time_interval'], instance[f'session{session_num}:event'])
                
                data.update({'social_context_prompt': social_context_prompt})
                data.update(session_data)

                final_dataset.append(data)
            
        Console().print(
            f"[ {self.args.dataset_name} ] # of {self.args.split} dataset: {len(final_dataset)}"
        )
        return final_dataset
    
    def _validate_lengths(self, time_interval, summary):
        """Validate that `time_interval` and `summary` have a length of 5."""
        assert len(time_interval) == len(summary) == 5, "Length of time_interval and summary must be 5"
    
    def _extract_session_data(self, instance, session_num):
        """Extract dialogue and speaker data for a given session."""
        session_data = {}
        session_prefix = f"session{session_num}"
        
        session_data[f"{session_prefix}_utterance"] = instance[f'session{session_num}:utterance']
        session_data[f"{session_prefix}_speakers"] = instance[f'session{session_num}:speaker']
        session_data[f"{session_prefix}_image_description"] = instance[f'session{session_num}:image_description']

        dialog = []
        share_flag = False
        for spk, utt, img in zip(
            instance[f'session{session_num}:speaker'], instance[f'session{session_num}:utterance'], instance[f'session{session_num}:image_description']
        ):
            new_utt = ''
            if img != '':
                if utt != '':
                    new_utt = f'{utt} [Sharing Image: {img}]'
                else:
                    new_utt = f'[Sharing Image: {img}]'
                share_flag = True
            else:
                new_utt = utt

            dialog.append({
                'speaker': spk,
                'utter': new_utt
            })

        session_data["dialogue"] = dialog
     
        return session_data, share_flag
    

class CactusDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("DLI-Lab/cactus")

    def process_data(self):
        final_dataset = []

        for conv_idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            
            dialogue = []
            for item in instance['dialogue'].split('\n'):
                if 'Client: ' in item:
                    speaker = 'Client'
                    utter = item.split(f'{speaker}: ')[-1]
                elif 'Client (sighs): ' in item:
                    speaker = 'Client'
                    utter = item.split(f'{speaker} (sighs): ')[-1]
                elif 'Client (sarcastically): ' in item:
                    speaker = 'Client'
                    utter = item.split(f'{speaker} (sarcastically): ')[-1]
                elif 'Counselor: ' in item:
                    speaker = 'Counselor'
                    utter = item.split(f'{speaker}: ')[-1]
                elif 'Counselor:' in item:
                    speaker = 'Counselor'
                    utter = item.split(f'{speaker}:')[-1]
                else:
                    raise ValueError(f'{item}')
                
                dialogue.append({
                    'speaker': speaker, 'utter': utter
                })
            
            client_intake_form = instance['intake_form']
            client_attitude = instance['attitude']

            final_dataset.append({
                'index': conv_idx,
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(client_intake_form, client_attitude)
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset

    def construct_social_context_prompt(self, client_intake_form, client_attitude):
        TEMPLATE = [
            "Client's attitude is {client_attitude}. The client's intake form is as follows:\n{client_intake_form}.",
            "The client has an attitude of {client_attitude}. Below is the client's intake form:\n{client_intake_form}.",
            "With an attitude of {client_attitude}, the client's intake form details are:\n{client_intake_form}.",
            "Client's attitude: {client_attitude}. Intake form information:\n{client_intake_form}.",
            "The client's attitude is {client_attitude}. Here is their intake form:\n{client_intake_form}."
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(client_attitude=client_attitude, client_intake_form=client_intake_form)

        return social_context_prompt

class PersuasionForGoodDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.args.split = 'FullDialog'
        self.load_data("spawn99/PersuasionForGood")

        self.speaker_dict = {0: 'Speaker A', 1: 'Speaker B'}
    
    def process_data(self):
        data = defaultdict(list)
        cnt = 0
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            
            conv_id = instance['B2']
            utterance = instance['Unit']
            speaker = self.speaker_dict[instance['B4']]

            data[conv_id].append({
                'utterance': utterance,
                'speaker': speaker
            })
        
        final_dataset = []
        for conv_index, conv in data.items():
            
            dialogue = []
            for i, item in enumerate(conv):
                dialogue.append({
                    'speaker': item['speaker'],
                    'utter': item['utterance']
                })

            social_context_prompt = self.construct_social_context_prompt()
            final_dataset.append({
                'index': conv_index,
                'dialogue': dialogue,
                'social_context_prompt': social_context_prompt
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )

        return final_dataset

    def construct_social_context_prompt(self):
        PERSUASION_TEMPLATE = [
            "Speaker A is attempting to persuade Speaker B.",
            "In this scenario, Speaker A is the Persuader and Speaker B is the Persuadee.",
            "Speaker A acts as Persuader, while Speaker B plays the role of Persuadee.",
            "In the conversation, Speaker A is persuading Speaker B.",
            "Speaker A aims to convince Speaker B."
        ]

        random.shuffle(PERSUASION_TEMPLATE)
        return PERSUASION_TEMPLATE[0]

class EmpathyDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("facebook/empathetic_dialogues")
    
    def get_speaker_dict(self):
        speaker_indices = list(set([item['speaker_idx'] for item in self.dataset]))
        Console().print(
            f'[ {self.args.dataset_name} | {self.args.split} ] # of speaker indices: {len(speaker_indices)}'
        )
    
    def process_data(self):
        data = defaultdict(list)
        cnt = 0
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            
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
        
        final_dataset = []
        for conv_index, conv in data.items():
            
            dialogue = []
            for i, item in enumerate(conv):
                dialogue.append({
                    'speaker': 'Speaker A' if i % 2 == 0 else 'Speaker B',
                    'utter': item['utterance']
                })

            social_context_prompt = self.construct_social_context_prompt(conv[0]['emotion'], conv[0]['situation'])
            final_dataset.append({
                'index': conv_index,
                'dialogue': dialogue,
                'social_context_prompt': social_context_prompt
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )

        return final_dataset

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


class SynPersonaChatDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("google/Synthetic-Persona-Chat")

    def process_data(self):
        final_dataset = []
        cnt = 0
        for idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            dialogue = []
            for item in instance['Best Generated Conversation'].split('\n'):
                if 'User 1: ' in item:
                    speaker = 'User 1'
                    utter = item.split(f'{speaker}: ')[-1]
                elif 'User 2: ' in item:
                    speaker = 'User 2'
                    utter = item.split(f'{speaker}: ')[-1]
                else:
                    continue
                
                dialogue.append({
                    'speaker': speaker, 'utter': utter
                })

            final_dataset.append({
                'index': idx,
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(instance['user 1 personas'], instance['user 2 personas'])
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset
        
    def construct_social_context_prompt(self, user1_persona, user2_persona):
        user1_persona = '\n- '.join(user1_persona.split('\n'))
        user2_persona = '\n- '.join(user2_persona.split('\n'))
        TEMPLATE = [
            "User 1's Persona Information:\n- {user1_persona}\n\nUser 2's Persona Information:\n- {user2_persona}",
            "User 1's Profile:\n- {user1_persona}\n\nUser 2's Profile:\n- {user2_persona}",
            "Details of User 1's Persona:\n- {user1_persona}\n\nDetails of User 2's Persona:\n- {user2_persona}",
            "Persona for User 1:\n- {user1_persona}\n\nPersona for User 2:\n- {user2_persona}",
            "Information about User 1's Persona:\n- {user1_persona}\n\nInformation about User 2's Persona:\n- {user2_persona}"
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(user1_persona=user1_persona, user2_persona=user2_persona)

        return social_context_prompt


class PearlDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("DLI-Lab/pearl")

    def process_data(self):
        final_dataset = []
        cnt = 0
        for idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            dialogue = []
            for item in instance['dialogue']:
                if item == '':
                    continue

                if 'Seeker: ' in item:
                    speaker = 'Seeker'
                    utter = item.split(f'{speaker}: ')[-1]
                elif 'Recommender: ' in item:
                    speaker = 'Recommender'
                    utter = item.split(f'{speaker}: ')[-1]
                else:
                    raise ValueError(f'{item}')

                dialogue.append({
                    'speaker': speaker, 'utter': utter
                })

            final_dataset.append({
                'index': idx,
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(instance['user_persona'])
            })
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset
        
    def construct_social_context_prompt(self, user_persona):
        TEMPLATE = [
            "Seeker's overall movie preferences are represented as follows:\n{user_persona}",
            "Here is the seeker's complete movie profile:\n{user_persona}",
            "The seeker's general movie state is described below:\n{user_persona}",
            "Representation of seeker's overall movie interests:\n{user_persona}",
            "Below is the seeker's overall movie persona:\n{user_persona}"
        ]
        random.shuffle(TEMPLATE)
        selected_template = TEMPLATE[0]
        social_context_prompt = selected_template.format(user_persona=user_persona)
        
        return social_context_prompt

class CasinoDataset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("kchawla123/casino")

        self.speaker_dict = {
            'mturk_agent_1': 'Speaker A',
            'mturk_agent_2': 'Speaker B'
        }

    def process_data(self):
        final_dataset = []
        cnt = 0
        for idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            logs = instance['chat_logs']

            dialogue = []
            for item in instance['chat_logs']:
                dialogue.append({
                    'speaker': self.speaker_dict[item['id']],
                    'utter': item['text']
                })

            final_dataset.append({
                'index': idx,
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(instance['participant_info'])
            })

        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )

        return final_dataset

    def extract_agent_info(self, agent_info, prefix):
        demographic = agent_info['demographics']
        personality = agent_info['personality']
        value2issue = agent_info['value2issue']
        value2reason = agent_info['value2reason']

        return {
            f'{prefix}_age': demographic['age'],
            f'{prefix}_gender': demographic['gender'],
            f'{prefix}_ethnicity': demographic['ethnicity'],
            f'{prefix}_education': demographic['education'],
            f'{prefix}_svo': personality['svo'],
            f'{prefix}_extraversion': personality['big-five']['extraversion'],
            f'{prefix}_agreeableness': personality['big-five']['agreeableness'],
            f'{prefix}_conscientiousness': personality['big-five']['conscientiousness'],
            f'{prefix}_emotional_stability': personality['big-five']['emotional-stability'],
            f'{prefix}_openness_to_experiences': personality['big-five']['openness-to-experiences'],
            f'{prefix}_value2issue_high': value2issue['High'],
            f'{prefix}_value2issue_medium': value2issue['Medium'],
            f'{prefix}_value2issue_low': value2issue['Low'],
            f'{prefix}_value2reason_high': value2reason['High'],
            f'{prefix}_value2reason_medium': value2reason['Medium'],
            f'{prefix}_value2reason_low': value2reason['Low'],
        }

    def construct_social_context_prompt(self, participant_info):

        agent_1_info = self.extract_agent_info(participant_info['mturk_agent_1'], 'speaker_a')
        agent_2_info = self.extract_agent_info(participant_info['mturk_agent_2'], 'speaker_b')

        all_agent_info = {**agent_1_info, **agent_2_info}

        template_pool = [CASINO_TEMPLATE_STRUCT, CASINO_TEMPLATE_SENTENCE]
        random.shuffle(template_pool)
        selected_template = template_pool[0]

        social_context_prompt = selected_template.format(**all_agent_info)

        return social_context_prompt

    
class WildChatDatset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("allenai/WildChat-1M")
    
    def process_data(self):
        final_dataset = []
        
        for idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            
            dialogue = [
                {'speaker': item['role'].capitalize(), 'utter': item['content']}
                for item in instance['conversation']
            ]
            
            if not instance['country'] or not instance['state']:
                continue
            
            if instance['language'] != self.args.target_language:
                continue
            
            unique_indices = []
            for item in instance['conversation']:
                unique_indices.append(item['turn_identifier'])
                
            social_context_prompt = self.construct_social_context_prompt(instance['country'], instance['state'])
            
            data = dict()
            data['index'] = str(idx) 
            data['dialogue'] = dialogue
            data['social_context_prompt'] = social_context_prompt
            
            final_dataset.append(data)
            
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        return final_dataset
            
    def construct_social_context_prompt(self, country, state):
        return f"User currently lives in {state}, {country}."
    
class MultifacetedCollectionDatset(HuggingFaceDataset):
    def __init__(self, args):
        super().__init__(args)
        
        self.load_data("kaist-ai/Multifaceted-Collection")
        
    def process_data(self):
        final_dataset = []
        
        for idx, instance in enumerate(tqdm(self.dataset, total=len(self.dataset))):
            index = '{}:{}:{}'.format(
                idx, instance['main_source'], instance['original_source']
            )

            dialogue = [
                {'speaker': 'Human', 'utter': instance['prompt']},
                {'speaker': 'AI Assistant', 'utter': instance['output']}
            ]

            final_dataset.append({
                'index': index,
                'dialogue': dialogue,
                'social_context_prompt': self.construct_social_context_prompt(instance['system'])
            })

        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] "
            f"{len(self.dataset) - len(final_dataset)} dialogue samples removed..."
        )
        
        return final_dataset
    
    def construct_social_context_prompt(self, system_message):
        return f"Here is the system message that conveys individual preferences: {system_message}"