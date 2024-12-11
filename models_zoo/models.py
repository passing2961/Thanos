import os

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSeq2SeqLM

from models_zoo import register_model
from models_zoo.wrappers import *


def load_llm(config, accel):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return LLM(tokenizer, model, config)


def load_llama(config, accel):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()
    
    return LLaMA(tokenizer, model, config)

@register_model("llama_3_1_8b")
def llama_3_1_8b(config, accel):
    return load_llama(config, accel)

@register_model("llama_3_2_1b")
def llama_3_2_1b(config, accel):
    return load_llama(config, accel)

@register_model("llama_3_2_3b")
def llama_3_2_3b(config, accel):
    return load_llama(config, accel)

def load_gemma(config, accel):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(accel.device)
    model = model.eval()

    return Gemma(tokenizer, model, config)

@register_model("gemma_2_it_2b")
def gemma_2_it_2b(config, accel):
    return load_gemma(config, accel)

@register_model("gemma_2_it_9b")
def gemma_2_it_9b(config, accel):
    return load_gemma(config, accel)


def load_qwen(config, accel):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return Qwen(tokenizer, model, config)

@register_model("qwen_2_5_it_1_5b")
def qwen_2_5_it_1_5b(config, accel):
    return load_qwen(config, accel)

def load_phi(config, accel):
    model = AutoModelForCausalLM.from_pretrained( 
        config.model_path,  
        trust_remote_code=True,  
        torch_dtype="auto",  
        attn_implementation='flash_attention_2'
    ) 

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return Phi(tokenizer, model, config)

@register_model("phi_3_mini_it_3_8b")
def phi_3_mini_it_3_8b(config, accel):
    return load_phi(config, accel) 


def load_mistral(config, accel):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return Mistral(tokenizer, model, config)

@register_model("mistral_it_0_2_7b")
def mistral_it_0_2_7b(config, accel):
    return load_mistral(config, accel)

@register_model("doctor")
def doctor(config, accel):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return Doctor(tokenizer, model, config)


def load_thanos(config, accel):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True, torch_dtype="auto",
    )
    model = model.to(accel.device)
    model = model.eval()

    return Thanos(tokenizer, model, config)

@register_model("thanos_1b")
def thanos_1b(config, accel):
    return load_thanos(config, accel)

@register_model("thanos_3b")
def thanos_3b(config, accel):
    return load_thanos(config, accel)

@register_model("thanos_8b")
def thanos_8b(config, accel):
    return load_thanos(config, accel)

def load_cosmo(config, accel):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
    model = model.to(accel.device)
    model = model.eval()

    return Cosmo(tokenizer, model, config)    

@register_model("cosmo_xl")
def cosmo_xl(config, accel):
    return load_cosmo(config, accel)