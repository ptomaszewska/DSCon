#!/bin/bash
#SBATCH --job-name=features      
#SBATCH --time=1-00:00:00               
#SBATCH --output=./logs/feat_%j.log    
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=40G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSCon


python ../CLAM/extract_features_fp.py --data_h5_dir $1 --data_slide_dir $2 --csv_path $3 --feat_dir $4 --batch_size 28 --slide_ext .tif --model_type $5 --swin_model_name $6 --gray $7