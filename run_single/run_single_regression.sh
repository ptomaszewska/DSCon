#!/bin/bash
#SBATCH --job-name=regression      
#SBATCH --time=0-24:00:00               
#SBATCH --output=./logs/reg_%j.log    
#SBATCH --error=./logs/reg_%j.err
#SBATCH --partition=short

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DSCon

python ../spatial_regression.py --model_type $1 --model_size $2 --lesion_type $3 --fold $4 --window_size $5 --k $6
