#!/bin/bash
#SBATCH --job-name=heatmap      
#SBATCH --time=0-3:00:00              
#SBATCH --output=./logs/heatmap_%j.log    
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=20G 


source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSCon

python ../CLAM/create_heatmaps.py --config config_template.yaml