#!/bin/bash
#SBATCH --job-name=CAMELYON_heatmap      # Job name
#SBATCH --time=0-3:00:00               # Time limit hrs:min:sec
#SBATCH --output=./logs/heatmap_%j.log    # Standard output and error log, plik pojawia się w miejscu odpalenia skryptu
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=20G 


source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSCon

python create_heatmaps.py --config config_template.yaml