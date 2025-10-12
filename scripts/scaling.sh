#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

# Scaling experiments with different dataset sizes
for proportion in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python asciite.py \
        --mode finetune \
        --huggingface_cache $huggingface_path \
        --model bert-base-uncased \
        --dataset ../AsciiTE.csv \
        --output_dir ../benchmark_data/results/TE-scaling/ \
        --data_proportion $proportion \
        --epochs 2 \
        --batch_size 16

    python asciite.py \
        --mode finetune \
        --huggingface_cache $huggingface_path \
        --model roberta-base \
        --dataset ../AsciiTE.csv \
        --output_dir ../benchmark_data/results/TE-scaling/ \
        --data_proportion $proportion \
        --epochs 2 \
        --batch_size 16

    python asciite.py \
        --mode finetune \
        --huggingface_cache $huggingface_path \
        --model microsoft/deberta-v3-base \
        --dataset ../AsciiTE.csv \
        --output_dir ../benchmark_data/results/TE-scaling/ \
        --data_proportion $proportion \
        --epochs 2 \
        --batch_size 16
done

