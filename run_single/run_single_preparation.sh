#!/bin/bash
#SBATCH --job-name=prep      
#SBATCH --time=1-00:00:00               
#SBATCH --output=./logs/prep_%j.log    
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=40G

# fill in below according to the experiment design
splits=('training') # 'testing'


data="CAMELYON16"
raw_data_path="../../CLAM_files/${data}/${split}/images"
patch_size=224  #256
patches_path="../../CLAM_files/${data}_patches_${patch_size}/${split}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DSCon

python ../CLAM/create_patches_fp.py --source ${raw_data_path} --save_dir ${patches_path}  --patch_size ${patch_size} --step_size ${patch_size} --seg --patch --stitch

python ../CLAM/create_filelist.py --data ${data} --patch_size ${patch_size} --split ${split}

python ../CLAM/create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --label_frac 1 --k 5 --val_frac 0.20 --test_frac 0.20 --data ${data}
