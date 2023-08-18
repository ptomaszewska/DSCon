#!/bin/bash


data="CAMELYON16"
raw_data_path="../CLAM_files/${data}/${split}/images"
dropout_rate=("0.25") 
bag_weights=("0.7")
layer1_neurons=512
layer2_neurons=128

learning_rate=0.005
weight_decay=0.0

learning_rate_resnet=2e-4
weight_decays_resnet=1e-5


feature_extractors=("swin")   #"resnetCLAM"  #vit
swin_model_names=("swin_base_patch4_window7_224_22k"
#"swinv2_base_patch4_window16_256"   
#"swinv2_base_patch4_window8_256"
#"swinv2_small_patch4_window16_256"    
#"swinv2_small_patch4_window8_256"
#"swinv2_tiny_patch4_window16_256"       
#"swinv2_tiny_patch4_window8_256"
)



for feature_extractor in "${feature_extractors[@]}"
do
    echo ${feature_extractor}
    for is_gray in "False" 
    do            
        for weighted in "True" 
        do
            for bag_weight in "${bag_weights[@]}"
            do

                if [ ${feature_extractor} == "swin" ]; then
                    for swin_model_name in "${swin_model_names[@]}"
                    do
                        layer1_neurons=512
                        layer2_neurons=128
                        swin_split=(${swin_model_name//_/ })
                        patch_size=${swin_split[4]}
                        patches_path="../CLAM_files/${data}_patches_${patch_size}/${split}"
                        features_path="../${data}_features_${patch_size}_${swin_model_name}/all"
                        exp_code="${data}_${patch_size}_${swin_model_name}"

                         if [ ${is_gray} == "True" ]; then
                             features_path="${features_path}_gray"
                             exp_code="${exp_code}_gray"           
                         fi

                         if [ ${weighted} == "True" ]; then
                              exp_code="${exp_code}_weighted"           
                          fi
                       
                          sbatch run_single_main.sh ${dropout_rate} "10" "50" ${exp_code} ${features_path} ${bag_weight} ${weighted} ${is_gray} ${layer1_neurons} ${layer2_neurons} ${learning_rate} ${weight_decay} "sgd"
                      done
                 else
                       patch_size=224
                       patches_path="../CLAM_files/${data}_patches_${patch_size}/${split}"
                       features_path="../${data}_features_${patch_size}_${feature_extractor}/all"
                       exp_code="${data}_${patch_size}_${feature_extractor}"                

                        if [ ${is_gray} == "True" ]; then
                                    features_path="${features_path}_gray"
                                    exp_code="${exp_code}_gray"  
                        fi
                        if [ ${weighted} == "True" ]; then
                                  exp_code="${exp_code}_weighted"           
                        fi
                        
                        if ${feature_extractor} == "resnetCLAM" ]; then
                            layer1_neurons=512
                            layer2_neurons=256
                            sbatch run_single_main.sh ${dropout_rate} "10" "50" ${exp_code} ${features_path} ${bag_weight} ${weighted} ${is_gray} ${layer1_neurons} ${layer2_neurons} ${learning_rate_resnet} ${weight_decay_resnet} "adam"
                        
                        else
                            layer1_neurons=512
                            layer2_neurons=128
                            sbatch run_single_main.sh ${dropout_rate} "10" "50" ${exp_code} ${features_path} ${bag_weight} ${weighted} ${is_gray} ${layer1_neurons} ${layer2_neurons} ${learning_rate} ${weight_decay} "sgd"
                         fi 
                    fi

                done
          done   
    done
done