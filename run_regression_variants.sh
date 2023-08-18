model_types=('swin') # 'swinv2' 'vit' 'resnetCLAM'
model_sizes=('small') #'tiny' 'base'
lesion_types=(0 1) 
folds=(0 1 2 3 4) 
window_sizes=(8) #16     ### this is applicable only to swinv2
ks=(99) # 8 24 48  

for m_type in "${model_types[@]}"
do
for m_size in "${model_sizes[@]}"
do
    for l_type in "${lesion_types[@]}"
    do
        for fold_id in "${folds[@]}"
        do
        for window_size in "${window_sizes[@]}"
        do
        for k in "${ks[@]}"
        do
        
        sbatch ./run_single/run_single_regression.sh ${m_type} ${m_size} ${l_type} ${fold_id} ${window_size} ${k}
	sleep 15
                done
              done
          done   
    done
done
done
