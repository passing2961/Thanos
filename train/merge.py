from peft import AutoPeftModelForCausalLM    
from transformers import AutoTokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# base_model = LlamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# adapters_name = 'peft_model'
# lora_model = PeftModel.from_pretrained(base_model, adapters_name)
# model = lora_model.merge_and_unload()

import torch
import os
new_model = AutoPeftModelForCausalLM.from_pretrained(
    'peft_model/thanos_8b',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16, #torch.float16,
    trust_remote_code=True,
    device_map='auto',
)

merged_model = new_model.merge_and_unload()
merged_model.save_pretrained(
    os.path.join('peft_model/thanos_8b', "merged_model"), trust_remote_code=True, safe_serialization=True)
tokenizer.save_pretrained(os.path.join('peft_model/thanos_8b', "merged_model"))
