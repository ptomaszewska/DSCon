#!/bin/bash
#SBATCH --job-name=classification     
#SBATCH --time=0-24:00:00               
#SBATCH --output=./logs/class_%j.log    
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=30G 
#SBATCH -A cause-lab


source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSCon


WANDB__SERVICE_WAIT=300 python ../CLAM/main.py --drop_out --dropout_rate $1 --early_stopping_patience $2 --early_stopping_stop_epoch $3 --lr ${11} --k 5 --label_frac 1 --exp_code $4 --bag_loss ce --inst_loss ce --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir $5  --bag_weight $6 --weighted_sample $7 --gray $8 --opt ${13} --reg ${12} --max_epochs 50 --layer1 $9 --layer2 ${10}
