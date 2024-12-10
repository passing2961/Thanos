import os
import copy
import json
import time
from tqdm import tqdm

from rich.console import Console

from agents_zoo import *
from utils.common import *
from utils.prompt_template import SYSTEM_PROMPT, SKILL_ANNOTATION_PROMPT_TEMPLATE


SAMPLING_PARAMS_OPENAI = {
    "max_tokens": 1024, 
    "temperature": 0.9, 
    "top_p": 0.95
}

class BaseAnnotator:
    
    def __init__(self, args):
        self.args = args
    
    def load_template(self, path: str):
        with open(path, 'r') as f:
            return f.read()
    
    def run(self):
        pass
    
    
class OpenAIAnnotator(BaseAnnotator):
    def __init__(self, args, dataset=None):
        super().__init__(args)
        
        self.client = OpenAIBatchClient()
        self.dataset = dataset
        os.makedirs(os.path.join(self.args.batch_file_dir, self.args.dataset_name), exist_ok=True)
        
        if self.args.mode == 'execute':
            self.dataset_dict = {item['index']: item for item in self.dataset}
        
    def prepare_input_prompts(self):
        input_prompts = []
        
        for instance in tqdm(self.dataset, total=len(self.dataset)):
            dialogue = ['{}: {}'.format(item['speaker'], item['utter']) for item in instance['dialogue']]
            response = dialogue[-1]
            dialogue = '\n'.join(dialogue[:-1])
            
            user_prompt = SKILL_ANNOTATION_PROMPT_TEMPLATE.format(
                response=response,
                dialogue=dialogue, 
                social_context=instance['social_context_prompt'],
                word_num=len(response.split())
            )
            
            messages = format_messages(SYSTEM_PROMPT, user_prompt)

            input_prompts.append(
                {
                    "custom_id": instance['index'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.args.model_name,
                        "messages": messages,
                        **SAMPLING_PARAMS_OPENAI,
                    },
                }
            )
        
        return input_prompts
    
    def create_batch(self):
        inputs = self.prepare_input_prompts()
        if self.args.debug:
            import random
            random.shuffle(inputs)
            inputs = inputs[:self.args.sub_sample_num]
        Console().print(f'# of inputs: {len(inputs)}')
        
        batch_file_path = os.path.join(self.args.batch_file_dir, self.args.dataset_name, f'{self.args.split}_batch.jsonl')
        with open(batch_file_path, "w") as f:
            for input in inputs:
                f.write(json.dumps(input) + "\n")
                
        batch = self.client.create_batch(batch_file_path, f'Create batch of {self.args.dataset_name} {self.args.split} dataset...')
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] Batch created: {batch}"
        )
        return batch.id
    
    def check_batch(self, batch_id):
        while True:
            status, batch_output_file_id = self.client.check_batch(batch_id)
            Console().print(
                f"[ {self.args.dataset_name} | {self.args.split} ] Current status: {status}"
            )

            if status == "completed":
                break
            elif status in ["failed", "cancelling", "cancelled", "expired"]:
                raise Exception(f"Batch failed with status: {status}")

            time.sleep(10)  # Wait for 30 seconds before checking again
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] Batch completed. Output file ID: {batch_output_file_id}"
        )
        return batch_output_file_id
    
    def run(self):
        batch_id = self.create_batch()
        batch_output_file_id = self.check_batch(batch_id)
        
        outputs = self.client.retrieve_batch(batch_output_file_id)
        output_path = os.path.join(self.args.batch_file_dir, self.args.dataset_name, f'{self.args.split}_batch_output.jsonl')
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] Retrieved results saved to {output_path}"
        )
        
        final_outputs = []
        batch_output_writer = open(output_path, 'w')
        for output in outputs.iter_lines():
            batch_output_writer.write(output + "\n")
            
            output = json.loads(output)
            
            custom_id = output["custom_id"]
            generation = output["response"]["body"]["choices"][0]["message"]["content"]
            
            input_token_cnt = output["response"]["body"]["usage"]["prompt_tokens"]
            output_token_cnt = output["response"]["body"]["usage"]["completion_tokens"]
            
            dataset_dict = self.dataset_dict[custom_id]
            dataset_dict.update({
                "generation": generation,
                "input_token_cnt": input_token_cnt,
                "output_token_cnt": output_token_cnt
            })
            final_outputs.append(dataset_dict)
            
        batch_output_writer.close()
        
        with open(output_path.replace('.jsonl', '.json'), 'w') as f:
            json.dump(final_outputs, f, indent='\t', ensure_ascii=False)
            
    def parse_generation(self, output_path: str):
        with open(output_path, 'r') as f:
            outputs = json.load(f)
        
        final_outputs = []
        input_token_cnt, output_token_cnt = 0, 0
        for output in tqdm(outputs, total=len(outputs)):
            generation = output['generation']
            input_token_cnt += output['input_token_cnt']
            output_token_cnt += output['output_token_cnt']
            
            try:
                generation = generation.replace("```json", "").replace("```", "")
                parsed_generation = json.loads(generation)
            except:
                print(generation, type(generation))
                
                continue
            
            
            cp_output = copy.deepcopy(output)
            cp_output['parsed_generation'] = parsed_generation
            final_outputs.append(cp_output)
        
        with open(output_path.replace('_output.json', '_parsed_output.json'), 'w') as f:
            json.dump(final_outputs, f, ensure_ascii=False, indent='\t')
        
        with open(output_path.replace('_output.json', '_usage_token.json'), 'w') as f:
            json.dump({"input_token_cnt": input_token_cnt, "output_token_cnt": output_token_cnt}, f, ensure_ascii=False, indent='\t')
            
        
        Console().print(
            f"[ {self.args.dataset_name} | {self.args.split} ] Completed parsing..."
        )