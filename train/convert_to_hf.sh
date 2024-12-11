#!/bin/bash


 python -m inference.checkpoint_converter_fsdp_hf \
    --fsdp_checkpoint_path  model_checkpoints/thanos_3b_ft-meta-llama/Llama-3.2-3B-Instruct \
    --consolidated_model_path save_models/thanos_3b \
    --HF_model_path_or_name meta-llama/Llama-3.2-3B-Instruct

 # --HF_model_path_or_name specifies the HF Llama model name or path where it has config.json and tokenizer.json


