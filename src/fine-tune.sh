#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

# Fine-tuning on AsciiTE dataset
python asciite.py \
    --mode finetune \
    --huggingface_cache $huggingface_path \
    --model \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-finetune/ \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 3e-5

python asciite.py \
    --mode finetune \
    --huggingface_cache $huggingface_path \
    --model  \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-finetune/ \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 3e-5

python asciite.py \
    --mode finetune \
    --huggingface_cache $huggingface_path \
    --model microsoft/models \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-finetune/ \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 3e-5

