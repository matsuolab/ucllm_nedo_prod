#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=prefilter
#SBATCH --output=prefilter.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12

# Command(s) goes here

python -m wikiextractor.WikiExtractor -o /persistentshare/storage/team_nakamura/member/horie/dataset/prefilter/ \
--no-templates /persistentshare/storage/team_nakamura/member/horie/dataset/tmp/wikipedia/20240301/ja/jawiki-20240301-pages-articles-multistream.xml.bz2