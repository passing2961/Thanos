#!/bin/bash


torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    finetuning.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --enable_fsdp \
    --dataset multifaceted_skill_collection_dataset \
    --num_epochs 3 \
    --batch_size_training 8 \
    --use_wandb True \
    --batching_strategy padding \
    --lr 1e-5 \
    --use_peft \
    --peft_method lora \
    --output_dir peft_model/thanos_1b 

torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    finetuning.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --enable_fsdp \
    --dataset multifaceted_skill_collection_dataset \
    --num_epochs 3 \
    --batch_size_training 8 \
    --use_wandb True \
    --batching_strategy padding \
    --lr 1e-5 \
    --use_peft \
    --peft_method lora \
    --output_dir peft_model/thanos_3b


torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    finetuning.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --enable_fsdp \
    --dataset multifaceted_skill_collection_dataset \
    --num_epochs 3 \
    --batch_size_training 8 \
    --use_wandb True \
    --batching_strategy padding \
    --lr 1e-5 \
    --use_peft \
    --peft_method lora \
    --output_dir peft_model/thanos_8b
