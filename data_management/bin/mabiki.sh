#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=mabiki
#SBATCH --output=mabiki.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12

# Command(s) goes here

python -m preprocessing.mabiki