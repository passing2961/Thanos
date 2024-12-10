#!/bin/bash



# dataset_names=(
#     "Casino"
#     "empathetic_dialogues"
#     "persuasion"
#     "wizard_of_wikipedia" X
#     "cactus"
#     "MULTIWOZ2_2".  X
#     "Prosocial".  X
#     "Janus"
#     "pearl"
#     "syn-personachat"
# )



python build_dataset.py \
    --dataset-name stark \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name stark \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name syn-personachat \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name syn-personachat \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name wizard_of_wikipedia \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name wizard_of_wikipedia \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name Prosocial \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 20000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name Prosocial \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 20000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4




python build_dataset.py \
    --dataset-name MULTIWOZ2_2 \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 10000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name MULTIWOZ2_2 \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 10000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name Casino \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name Casino \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name empathetic_dialogues \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 10 \
    --batch-file-dir annotation_results/test3 \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name persuasion \
    --split FullDialog \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name persuasion \
    --split FullDialog \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4



python build_dataset.py \
    --dataset-name pearl \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name pearl \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 5000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4




python build_dataset.py \
    --dataset-name empathetic_dialogues \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name empathetic_dialogues \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4


python build_dataset.py \
    --dataset-name cactus \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode execute \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name cactus \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 2500 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4






python build_dataset.py \
    --dataset-name ConversationChronicles \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 20000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

python build_dataset.py \
    --dataset-name SODA \
    --split train \
    --debug \
    --model-name gpt-4-turbo \
    --mode parse \
    --sub-sample-num 20000 \
    --batch-file-dir annotation_results/multifaceted \
    --multi-turn-filter \
    --turn-num-threshold 4

# python build_dataset.py \
#     --dataset-name WildChat \
#     --split train \
#     --debug \
#     --model-name gpt-4-turbo \
#     --mode execute \
#     --sub-sample-num 10000 \
#     --batch-file-dir annotation_results/multifaceted \
#     --multi-turn-filter \
#     --turn-num-threshold 2

# python build_dataset.py \
#     --dataset-name Janus \
#     --split train \
#     --debug \
#     --model-name gpt-4-turbo \
#     --mode execute \
#     --sub-sample-num 10 \
#     --batch-file-dir annotation_results/multifaceted \