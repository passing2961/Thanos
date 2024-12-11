import os

import torch

from models_zoo.wrappers.base import BaseModel


class Cosmo(BaseModel):
    def __init__(self, tokenizer, model, config):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.model = model
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
        self.top_p = config.top_p

    def construct_prompt(self, dialogue, social_context, rationale=None, skill=None):
        #input_text = " <turn> ".join([ele.split('Speaker A: | Speaker B: ') for ele in dialogue])

        input_text = []
        for ele in dialogue:
            if 'Speaker A:' in ele:
                input_text.append(ele.split('Speaker A: ')[-1].strip())
            elif 'Speaker B:' in ele:
                input_text.append(ele.split('Speaker B: ')[-1].strip())
            

        input_text = " <turn> ".join(input_text)

        if rationale != None:
            skill_of_mind = f'{rationale} Thus, the most appropriate conversational skill for the next response is {skill}.'
            input_text = "{} <sep> {} <sep> {}".format(social_context, skill_of_mind, input_text)
        else:
            input_text = "{} <sep> {}".format(social_context, input_text)
        return input_text

    def parse_skill(self, output):
        rationale, skill = output.split(' Thus, "')
        skill = skill.split('"')[0]

        return skill, rationale

    def generate(self, inputs=None, device=None):
        
        if 'thanos_next_resp' in self.config.task_type:
            prompt = self.construct_prompt(inputs[0]['flatten_dialogue'], inputs[0]['social_context'], inputs[0]['rationale'], inputs[0]['skill'])
        else:
            prompt = self.construct_prompt(inputs[0]['flatten_dialogue'], inputs[0]['social_context'])
        
        input_ids = self.tokenizer(
            [prompt], return_tensors="pt").to(device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids['input_ids'],
                do_sample=True if self.temperature > 0 else False,
                temperature=1.0,
                max_new_tokens=128,
                top_p = .95,
                #use_cache=True,
                #num_beams=1,
                #repetition_penalty=1.03
            )
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #outputs = self.postprocess_output(outputs)
        #skill, rationale = self.parse_skill(outputs)
        
        if isinstance(outputs, str):
            outputs = [outputs]
        return outputs