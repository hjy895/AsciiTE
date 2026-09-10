#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH

# Unsupervised evaluation on AsciiTE (zero-shot)
python asciite.py \
    --mode unsupervised \
    --huggingface_cache $huggingface_path \
    --model name mentioned \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-unsup/

python asciite.py \
    --mode unsupervised \
    --huggingface_cache $huggingface_path \
    --model name mentioned \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-unsup/

python asciite.py \
    --mode unsupervised \
    --huggingface_cache $huggingface_path \
    --model name mentioned \
    --dataset ../AsciiTE.csv \
    --output_dir ../benchmark_data/results/TE-unsup/

