# fill in below according to the experiment design

splits=('training') # 'testing'
feature_extractors=("swin")  #"resnetCLAM", "vit" 
swin_model_names=("swin_base_patch4_window7_224_22k"
#"swinv2_base_patch4_window16_256"   
#"swinv2_base_patch4_window8_256"
#"swinv2_small_patch4_window16_256"    
#"swinv2_small_patch4_window8_256"
#"swinv2_tiny_patch4_window16_256"       
#"swinv2_tiny_patch4_window8_256"
)

data="CAMELYON16"

for split in "${splits[@]}"
do
    raw_data_path="../${data}/${split}/images" 

    for feature_extractor in "${feature_extractors[@]}"
    do
        for is_gray in "False" 
        do
               if [ ${feature_extractor} == "swin" ]; then
                        for swin_model_name in "${swin_model_names[@]}"
                        do
                            swin_split=(${swin_model_name//_/ })
                            patch_size=${swin_split[4]}
                            patches_path="../CLAM_files/${data}_patches_${patch_size}/${split}" 
                            features_path="../${data}_features_${patch_size}_${swin_model_name}/all"  
                            exp_code="${data}_${patch_size}_${swin_model_name}"

                            if [ ${is_gray} == True ]; then
                                features_path="${features_path}_gray"
                            fi

                            sbatch ./run_single/run_single_feat.sh  ${patches_path} ${raw_data_path} "${patches_path}/process_list_autogen_modified.csv" ${features_path} ${feature_extractor} ${swin_model_name} ${is_gray} 
                         done

                else
                        patch_size=224
                        patches_path="../CLAM_files/${data}_patches_${patch_size}/${split}"  
                        features_path="../${data}_features_${patch_size}_${feature_extractor}/all"     
                         exp_code="${data}_${patch_size}_${feature_extractor}"

                            if [ ${is_gray} == True ]; then
                                    features_path="${features_path}_gray"
                            fi

                            sbatch ./run_single/run_single_feat.sh  ${patches_path} ${raw_data_path} "${patches_path}/process_list_autogen_modified.csv" ${features_path} ${feature_extractor} "None" ${is_gray} 
                 fi 
          done   
    done
done
