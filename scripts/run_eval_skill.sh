#!/bin/bash

GPU_DEVICE="0,1,2,3,4,5,6,7"
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))
CACHE_DIR="cache/lvlms"


# # skill generation for response generation
# MODEL_INFOS=(
#     #"passing2961/llama3.1-multifaceted-skill-predictor-8B thanos"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b/merged_model thanos_1b"
#     # "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_v2/merged_model thanos_1b"
#     # "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_3b_v2/merged_model thanos_3b"
#     # "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_8b_v2/merged_model thanos_8b"
#     # #"lmsys/vicuna-7b-v1.5 vicuna_v1_5_7b"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_new_v2/merged_model thanos_1b_new_v2"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_3b_new_v2/merged_model thanos_3b_new_v2"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_8b_new_v2/merged_model thanos_8b_new_v2"
#     "meta-llama/Llama-3.1-8B llama_3_2_1b"
#     "meta-llama/Llama-3.1-8B llama_3_1_8b"
#     #"DLI-Lab/DOCTOR doctor"
#     "google/gemma-2-2b-it gemma_2_it_2b"
#     "google/gemma-2-9b-it gemma_2_it_9b"
#     "Qwen/Qwen2.5-1.5B-Instruct qwen_2_5_it_1_5b"
#     "microsoft/Phi-3-mini-4k-instruct phi_3_mini_it_3_8b"
#     "mistralai/Mistral-7B-Instruct-v0.2 mistral_it_0_2_7b"
#     #"allenai/cosmo-xl cosmo_xl"
# )
# DATASET_NAMES=("ours" "photochat" "bst" "prosocial") # "dailydialog" "empathy")

# # Loop through the model infos
# for dataset_name in "${DATASET_NAMES[@]}"; do
#     for model_info in "${MODEL_INFOS[@]}"; do
#         # Split the model_info string into model path and name
#         model_path=$(echo $model_info | awk '{print $1}')
#         model_name=$(echo $model_info | awk '{print $2}')

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
#         accelerate launch --config_file utils/ddp_accel.yaml \
#             --num_processes=$n_gpu \
#             evaluate_response.py \
#             --batch-size 1 \
#             --dataset $dataset_name \
#             --model-path $model_path \
#             --model-name $model_name \
#             --cache-dir $CACHE_DIR \
#             --top-p 0.9 \
#             --temperature 1.0 \
#             --max-new-tokens 256 \
#             --task-type thanos_skill \
#             #--debug
#             #--for-human-eval \
#             #--thanos-model-size 8b
#             #--use-skill-annot \
#             #--debug

#     done
# done


# MODEL_INFOS=(
#     "meta-llama/Llama-3.1-8B llama_3_2_1b"
#     "meta-llama/Llama-3.1-8B llama_3_1_8b"
#     #"DLI-Lab/DOCTOR doctor"
#     "google/gemma-2-2b-it gemma_2_it_2b"
#     "google/gemma-2-9b-it gemma_2_it_9b"
#     "Qwen/Qwen2.5-1.5B-Instruct qwen_2_5_it_1_5b"
#     "microsoft/Phi-3-mini-4k-instruct phi_3_mini_it_3_8b"
#     "mistralai/Mistral-7B-Instruct-v0.2 mistral_it_0_2_7b"
#     #"allenai/cosmo-xl cosmo_xl"
# )
# DATASET_NAMES=("ours" "photochat" "bst" "prosocial") # "dailydialog" "empathy")

# # Loop through the model infos
# for dataset_name in "${DATASET_NAMES[@]}"; do
#     for model_info in "${MODEL_INFOS[@]}"; do
#         # Split the model_info string into model path and name
#         model_path=$(echo $model_info | awk '{print $1}')
#         model_name=$(echo $model_info | awk '{print $2}')

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
#         accelerate launch --config_file utils/ddp_accel.yaml \
#             --num_processes=$n_gpu \
#             evaluate_response.py \
#             --batch-size 1 \
#             --dataset $dataset_name \
#             --model-path $model_path \
#             --model-name $model_name \
#             --cache-dir $CACHE_DIR \
#             --top-p 0.9 \
#             --temperature 1.0 \
#             --max-new-tokens 256 \
#             --task-type base_skill \
#             #--debug
#             #--for-human-eval \
#             #--thanos-model-size 8b
#             #--use-skill-annot \
#             #--debug

#     done
# done

# # prosocial skill classificaiton
# MODEL_INFOS=(
#     #"meta-llama/Llama-3.1-8B llama_3_2_1b"
#     #"meta-llama/Llama-3.1-8B llama_3_1_8b"
#     "DLI-Lab/DOCTOR doctor"
#     #"google/gemma-2-2b-it gemma_2_it_2b"
#     #"google/gemma-2-9b-it gemma_2_it_9b"
#     #"Qwen/Qwen2.5-1.5B-Instruct qwen_2_5_it_1_5b"
#     #"microsoft/Phi-3-mini-4k-instruct phi_3_mini_it_3_8b"
#     #"mistralai/Mistral-7B-Instruct-v0.2 mistral_it_0_2_7b"
#     #"allenai/cosmo-xl cosmo_xl"
# )
# DATASET_NAMES=("prosocial" "dailydialog" "empathy")

# # Loop through the model infos
# for dataset_name in "${DATASET_NAMES[@]}"; do
#     for model_info in "${MODEL_INFOS[@]}"; do
#         # Split the model_info string into model path and name
#         model_path=$(echo $model_info | awk '{print $1}')
#         model_name=$(echo $model_info | awk '{print $2}')

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
#         accelerate launch --config_file utils/ddp_accel.yaml \
#             --num_processes=$n_gpu \
#             evaluate_response.py \
#             --batch-size 1 \
#             --dataset $dataset_name \
#             --model-path $model_path \
#             --model-name $model_name \
#             --cache-dir $CACHE_DIR \
#             --top-p 0.9 \
#             --temperature 1.0 \
#             --max-new-tokens 256 \
#             --task-type doctor \
#             --debug
#             #--for-human-eval \
#             #--thanos-model-size 8b
#             #--use-skill-annot \
#             #--debug

#     done
# done

# #######

# using generated skill for response generation

# MODEL_INFOS=(
#     #"passing2961/llama3.1-multifaceted-skill-predictor-8B thanos"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b/merged_model thanos_1b"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_v2/merged_model thanos_1b"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_3b_v2/merged_model thanos_3b"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_8b_v2/merged_model thanos_8b"
#     #"/home/work/workspace/NAACL2025/main_codes/train/save_models/thanos_3b thanos_3b_ft"
#     #"/home/work/workspace/NAACL2025/main_codes/train/save_models/thanos_3b_v2 thanos_3b_ft_v2"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_new_v1/merged_model thanos_1b_new_v1"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_new_v2/merged_model thanos_1b"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_3b_new_v2/merged_model thanos_3b"
#     "/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_8b_new_v2/merged_model thanos_8b"
#     #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_new_v2/merged_model thanos_1b_new_v2"
#     #"lmsys/vicuna-7b-v1.5 vicuna_v1_5_7b"
#     #"meta-llama/Llama-3.1-8B llama_3_2_1b"
#     #"meta-llama/Llama-3.1-8B llama_3_1_8b"
#     #"DLI-Lab/DOCTOR doctor"
#     #"google/gemma-2-2b-it gemma_2_it_2b"
#     #"google/gemma-2-9b-it gemma_2_it_9b"
#     #"Qwen/Qwen2.5-1.5B-Instruct qwen_2_5_it_1_5b"
#     #"microsoft/Phi-3-mini-4k-instruct phi_3_mini_it_3_8b"
#     #"mistralai/Mistral-7B-Instruct-v0.2 mistral_it_0_2_7b"
#     #"allenai/cosmo-xl cosmo_xl"
# )
DATASET_NAMES=("empathy" "dailydialog")

# # Loop through the model infos
# for dataset_name in "${DATASET_NAMES[@]}"; do
#     for model_info in "${MODEL_INFOS[@]}"; do
#         # Split the model_info string into model path and name
#         model_path=$(echo $model_info | awk '{print $1}')
#         model_name=$(echo $model_info | awk '{print $2}')

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
#         accelerate launch --config_file utils/ddp_accel.yaml \
#             --num_processes=$n_gpu \
#             evaluate_response.py \
#             --batch-size 1 \
#             --dataset $dataset_name \
#             --model-path $model_path \
#             --model-name $model_name \
#             --cache-dir $CACHE_DIR \
#             --top-p 0.9 \
#             --temperature 1.0 \
#             --max-new-tokens 256 \
#             --task-type thanos_skill \
#             #--debug
#             #--for-human-eval \
#             #--thanos-model-size 8b
#             #--use-skill-annot \
#             #--debug

#     done
# done
# doctor_next_resp


MODEL_INFOS=(
    #"passing2961/llama3.1-multifaceted-skill-predictor-8B thanos"
    #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b/merged_model thanos_1b"
    #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_1b_v2/merged_model thanos_1b"
    #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_3b_v2/merged_model thanos_3b"
    #"/home/work/workspace/NAACL2025/main_codes/train/peft_model/thanos_8b_v2/merged_model thanos_8b"
    #"lmsys/vicuna-7b-v1.5 vicuna_v1_5_7b"
    "meta-llama/Llama-3.1-8B llama_3_2_1b"
    "meta-llama/Llama-3.1-8B llama_3_1_8b"
    #"DLI-Lab/DOCTOR doctor"
    "google/gemma-2-2b-it gemma_2_it_2b"
    #"google/gemma-2-9b-it gemma_2_it_9b"
    "Qwen/Qwen2.5-1.5B-Instruct qwen_2_5_it_1_5b"
    #"microsoft/Phi-3-mini-4k-instruct phi_3_mini_it_3_8b"
    #"mistralai/Mistral-7B-Instruct-v0.2 mistral_it_0_2_7b"
    "allenai/cosmo-xl cosmo_xl"
)
# DATASET_NAMES=("prosocial") # "empathy" "dailydialog")

# # Loop through the model infos
for dataset_name in "${DATASET_NAMES[@]}"; do
    for model_info in "${MODEL_INFOS[@]}"; do
        # Split the model_info string into model path and name
        model_path=$(echo $model_info | awk '{print $1}')
        model_name=$(echo $model_info | awk '{print $2}')

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
        accelerate launch --config_file utils/ddp_accel.yaml \
            --num_processes=$n_gpu \
            evaluate_response.py \
            --batch-size 1 \
            --dataset $dataset_name \
            --model-path $model_path \
            --model-name $model_name \
            --cache-dir $CACHE_DIR \
            --top-p 0.9 \
            --temperature 1.0 \
            --max-new-tokens 256 \
            --task-type thanos_next_resp_both \
            --thanos-model-size 1b \
            #--debug
            #--for-human-eval \
            #--thanos-model-size 8b
            #--use-skill-annot \
            #--debug

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
        accelerate launch --config_file utils/ddp_accel.yaml \
            --num_processes=$n_gpu \
            evaluate_response.py \
            --batch-size 1 \
            --dataset $dataset_name \
            --model-path $model_path \
            --model-name $model_name \
            --cache-dir $CACHE_DIR \
            --top-p 0.9 \
            --temperature 1.0 \
            --max-new-tokens 256 \
            --task-type thanos_next_resp_both \
            --thanos-model-size 3b \
            #--debug

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
        accelerate launch --config_file utils/ddp_accel.yaml \
            --num_processes=$n_gpu \
            evaluate_response.py \
            --batch-size 1 \
            --dataset $dataset_name \
            --model-path $model_path \
            --model-name $model_name \
            --cache-dir $CACHE_DIR \
            --top-p 0.9 \
            --temperature 1.0 \
            --max-new-tokens 256 \
            --task-type thanos_next_resp_both \
            --thanos-model-size 8b \
            #--debug
    done
done


# # Loop through the model infos
for dataset_name in "${DATASET_NAMES[@]}"; do
    for model_info in "${MODEL_INFOS[@]}"; do
        # Split the model_info string into model path and name
        model_path=$(echo $model_info | awk '{print $1}')
        model_name=$(echo $model_info | awk '{print $2}')

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
        accelerate launch --config_file utils/ddp_accel.yaml \
            --num_processes=$n_gpu \
            evaluate_response.py \
            --batch-size 1 \
            --dataset $dataset_name \
            --model-path $model_path \
            --model-name $model_name \
            --cache-dir $CACHE_DIR \
            --top-p 0.9 \
            --temperature 1.0 \
            --max-new-tokens 256 \
            --task-type base_next_resp \
            #--debug
            #--for-human-eval \
            #--thanos-model-size 8b
            #--use-skill-annot \
            #--debug

    done
done

# # # thanos_next_resp_both