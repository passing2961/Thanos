import os

import torch

from models_zoo.wrappers.base import BaseModel


class Doctor(BaseModel):
    def __init__(self, tokenizer, model, config):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.model = model
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
        self.top_p = config.top_p

    def construct_prompt(self, text: str):
        return text

    def generate(self, inputs=None, device=None):
        prompt = self.construct_prompt(inputs[0]['prompt_input'])
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **input_ids,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p = self.top_p,
                use_cache=True,
                num_beams=1,
            )

        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.input_ids.shape[1]:], skip_special_tokens=True)[0]    

        if isinstance(outputs, str):
            outputs = [outputs]
        return outputs