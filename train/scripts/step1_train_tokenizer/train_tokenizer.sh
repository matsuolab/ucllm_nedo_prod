#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=tokenizer
#SBATCH --output=tokenizer.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=12

# Command(s) goes here
python ./train_sentencepiece_tokenizer.py \
    --input /persistentshare/storage/team_nakamura/member/horie/dataset/jawiki_newline_mecab.txt \
    --model_prefix JINIAC_V0_2 \
    --vocab_size 32000