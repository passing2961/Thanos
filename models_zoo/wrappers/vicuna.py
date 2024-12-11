import os

import torch

from models_zoo.wrappers.base import BaseModel


class Vicuna(BaseModel):
    def __init__(self, tokenizer, model, config):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.model = model
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens

    def construct_prompt(self, text: str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def generate(self, inputs=None, device=None):
        prompt = self.construct_prompt(inputs[0]['text'])
        input_ids = self.tokenizer(
            prompt, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **input_ids,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p = 0.0,
                use_cache=True,
                num_beams=1,
            )

        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.input_ids.shape[1]:], skip_special_tokens=True)[0]    
        
        if isinstance(outputs, str):
            outputs = [outputs]
        return outputs