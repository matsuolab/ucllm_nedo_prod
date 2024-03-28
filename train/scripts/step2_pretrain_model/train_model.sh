#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=train_model
#SBATCH --output=train_model.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12

# Command(s) goes here
cd ~/ucllm_nedo_dev/train/scripts/step2_pretrain_model

bash ./gcp_node-1_gpu/dataset-arxiv_tokenizer-sentencepiece_model-gpt_0.125B/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
    --input_tokenizer_file /persistentshare/storage/team_nakamura/member/horie/tokenizer/JINIAC_V0_1.model \
    --output_model_dir /persistentshare/storage/team_nakamura/member/horie/output/step2_pretrain_model/wiki_1.9b_0.084b \
    --save_interval 200
