import pingouin
from pingouin import ttest
from scipy.stats import wilcoxon

from utils import *
from plots import *

alpha=0.05
ks=[8, 24, 48, 99]
k_dict={8:0, 24:1, 48:2, 99:3} ## ordering
spatial_regs=['wx', 'lag', 'error']    
n_spatial_reg=len(spatial_regs) 
metric_names=['SCM_features', 'SCM_targets', 'SCM_residuals']
model_combinations=[('swin', 'base', 'window7'), ('swinv2', 'base', 'window8'), 
        ('swinv2', 'base', 'window16'), ('swinv2', 'small', 'window8'), 
        ('swinv2', 'small', 'window16'), ('swinv2', 'tiny', 'window8'), 
        ('swinv2', 'tiny', 'window16'), ('resnetCLAM'), 
        ('vit')
    ]
model_combinations_formatted=[('_').join(model) if 'swin' in model[0] else model for model in model_combinations]
model_combinations_formatted=[model.replace('swin', 's').replace('window', 'w') for model in model_combinations_formatted]


def spatial_context_measures(df, d_values):
    for i, col in enumerate(spatial_regs):
        df[metric_names[i]]=df[col+'_r2'] - df.r2
        d_values[i].extend(df[metric_names[i]])

    return df, d_values 

def aggregate_analyse_results(filelist, patches_per_tumor_224, patches_per_tumor_256, split, agg='concat'):
    """
    agg: mean, concat
    """
    counter=0
    p_vals=[[] for _ in range(n_spatial_reg)]   
    filelist_names=[]
    d_vals=[[] for _ in range(n_spatial_reg)]   
    names_all_single, names_all_models=[], []
    n_tumor_patches, tumor_patches_dist=[], []

    filelist_all_folds=[]
    counter_folds=0

    labels=[]
    dfs=[]

    for i in sorted(filelist):
        if counter_folds==0:
            results_folds=[]

        df=custom_read_csv(i, split=split)   
        d=df[df.Moran_p<alpha]
        df=df.sort_values(by='filename').reset_index(drop=True)
        results_folds.append(df)
        counter_folds+=1

        if counter_folds==5:
            counter_folds=0
            
            if agg=='concat':
                df=pd.concat(results_folds)

            elif agg=='mean':
                argmax_shape=np.argmax([len(results_folds[i].filename) for i in range(5)])
                indexes=list(set(range(5))-set([argmax_shape]))

                set_names=set(results_folds[argmax_shape].filename).intersection(set(results_folds[indexes[0]].filename), 
                                                                                 set(results_folds[indexes[1]].filename), 
                                                                                  set(results_folds[indexes[2]].filename), 
                                                                                 set(results_folds[indexes[3]].filename))
                for j in range(5):
                    results_folds[j]=results_folds[j][results_folds[j].filename.isin(set_names)]
                    results_folds[j]=results_folds[j].sort_values(by='filename').reset_index(drop=True)

                df=pd.concat(results_folds).mean(level=0)
                df['filename']=results_folds[0].filename

            
            model_name= i.split('/')[-1].replace('_4.csv','')
            filelist_all_folds.append(model_name)
            df, d_vals =spatial_context_measures(df, d_vals)

            names_all_single.extend(df.filename.values)
            names_all_models.extend([i]*df.shape[0])

            if "swinv2" in i: 
                n_tumor_patches, tumor_patches_dist=include_tumor_info(df, patches_per_tumor_256, 
                                                                               n_tumor_patches, tumor_patches_dist)
            else:
                n_tumor_patches, tumor_patches_dist=include_tumor_info(df, patches_per_tumor_224, 
                                                                               n_tumor_patches, tumor_patches_dist)

            for reg in spatial_regs:
                plot_distribution(df, "label", model_name, 'concat', split)

            p_vals=t_tests(df, p_vals)  
            dfs.append(df)
            labels.extend(df.label)
 
    df_p=pd.DataFrame({'filename': filelist_all_folds, 'p_val_diff_wx': p_vals[0], 'p_val_diff_lag': p_vals[1],
                       'p_val_diff_error': p_vals[2]
                      })
    d_per_images=pd.DataFrame({'names_all_models': names_all_models, metric_names[0]: d_vals[0], metric_names[1]: d_vals[1], 
                               metric_names[2]: d_vals[2], 'names_all_single': names_all_single, 'n_tumor_patches': n_tumor_patches,
                               'tumor_patches_dist':tumor_patches_dist, 'label': labels
                              })
    return df_p, d_per_images, dfs

def t_tests(df, p_vals): 
    df_0=df[df.label==0]
    df_1=df[df.label==1]
    for i, col in enumerate(metric_names):
        t=ttest( df_1[col], df_0[col], correction = False, alternative='greater')
        p_vals[i].append(t['p-val'][0]) 
    return p_vals 

def pairwise_features_targets(d_per_images, col_base='SCM_features', col_second='SCM_targets', per_k=True):
    values=[]
    
    d_per_images=extract_from_df(d_per_images, 'names_all_models')
    
    d_per_images=d_per_images[d_per_images.label==1]

    df = pd.DataFrame(columns=['model_name', 'k', 'p_value','mean_'+col_base, 'mean_'+col_second,
                               'max_'+col_base, 'max_'+col_second])
    
    model_combinations_full=[model if type(model)==tuple else (model, '', '') for model in model_combinations]
    
    for i, model_n_size in enumerate(model_combinations_full):
        df_p1=d_per_images[(d_per_images['model']==model_n_size[0]) & (d_per_images['size']==model_n_size[1])
                          & (d_per_images['window_size']==model_n_size[2])]
        if per_k:
            for k in ks:
                df_p1_k=df_p1[df_p1.names_all_models.str.split('/').str[-1].str.contains("k_"+str(k))==True]

                w,p = wilcoxon(df_p1_k[col_base], df_p1_k[col_second], alternative='greater' )

                df.loc[len(df)] = [model_combinations_formatted[i], k, p, np.mean(df_p1_k[col_base]), np.mean(df_p1_k[col_second]),
                          np.max(df_p1_k[col_base]), np.max(df_p1_k[col_second])]
        else:
            w,p = wilcoxon(df_p1[col_base], df_p1[col_second], alternative='greater' )
            df.loc[len(df)] = [model_combinations_formatted[i], '-1', p, np.mean(df_p1[col_base]), np.mean(df_p1[col_second]),
                          np.max(df_p1[col_base]), np.max(df_p1[col_second])]

    return df

def mean_train_test(d_per_images_train, d_per_images_test, col='diff_wx'): 
    values=[]

    d_per_images_train=extract_from_df(d_per_images_train, 'names_all_models')
    d_per_images_train=d_per_images_train[d_per_images_train.label==1] 
    
    d_per_images_test=extract_from_df(d_per_images_test, 'names_all_models')
    d_per_images_test=d_per_images_test[d_per_images_test.label==1]
    
    model_combinations1=[model if type(model)==tuple else (model, '', '') for model in model_combinations]
    df = pd.DataFrame(columns=['model_name', 'k', 'p-value'])

    for i, model_n_size1 in enumerate(model_combinations1):    
        df_p1_train=d_per_images_train[(d_per_images_train['model']==model_n_size1[0]) 
                            &            (d_per_images_train['size']==model_n_size1[1])
                          & (d_per_images_train['window_size']==model_n_size1[2])]
        df_p1_test=d_per_images_test[(d_per_images_test['model']==model_n_size1[0])
                           &           (d_per_images_test['size']==model_n_size1[1])
                          & (d_per_images_test['window_size']==model_n_size1[2])]

        for k in ks:
            df_p1_k_train=df_p1_train[df_p1_train.names_all_models.str.split('/').str[-1].str.contains("k_"+str(k))==True]
            df_p1_k_test=df_p1_test[df_p1_test.names_all_models.str.split('/').str[-1].str.contains("k_"+str(k))==True]
            t=ttest(df_p1_k_train[col], df_p1_k_test[col], alternative='greater')
            df.loc[len(df)] = [model_combinations_formatted[i], k, t['p-val'][0]]
    return df

def n_not_correlated(filelist):
    dict_corr={}
    for i in sorted(filelist)[:9*5]:
        df=custom_read_csv(i, split='test')   
        df=df[df.Moran_p<alpha]
        model_name= ('_').join(i.split('/')[-1].split('_')[:-1]).replace('.csv','')
        if model_name not in dict_corr.keys():
            dict_corr[model_name]=list(df.n_correlated)
        else:
            dict_corr[model_name].extend(df.n_correlated.values)

    for k,v in dict_corr.items():
        if "tiny" in k or "small" in k:
            feature_size=768
        else:
            feature_size=1024
        dict_corr[k]=feature_size-np.mean(v)
    return dict_corr

def extract_tumor_images_stats(dfs, patches_per_tumor_224, filelist):
    list_count={}
    list_names={}
    list_k={}
    list_SCM_targets=[{} for i in range(len(ks))] 
    #list_SCM_targets_k8, list_SCM_targets_k24, list_SCM_targets_48, list_SCM_targets_99
    list_SCM_features=[{} for i in range(len(ks))] 
    #list_SCM_features_k8, list_SCM_features_k24, list_SCM_features_48, list_SCM_features_99

    for i, df in enumerate(dfs):
        a=df[df.label==1]      #### take only data with tumor
        for name in a.filename.values:
            k=int(sorted(filelist)[5*i].split('/')[-1].split('_')[1])
            if name not in list_count.keys():
                list_count[name]=1
                list_names[name]=[sorted(filelist)[5*i].split('/')[-1]]
                list_k[name]=[k]

                list_SCM_features, list_SCM_targets=initialize_dicts(name, list_SCM_features, list_SCM_targets)
                extend_dicts(a, name, list_SCM_features, list_SCM_targets, k)

            else:
                list_count[name]+=1
                list_names[name].append(sorted(filelist)[5*i].split('/')[-1])
                list_k[name].append(int(sorted(filelist)[5*i].split('/')[-1].split('_')[1]))
                extend_dicts(a, name, list_SCM_features, list_SCM_targets, k)
                
    dict_tumor_img_stats={key:
                  patches_per_tumor_224[key.split('.')[0]]+
                  [v]+
                  [np.mean(list_k[key])]+
                  mean_list(list_SCM_targets, list_SCM_features, key)
                for key, v in sorted(list_count.items(), reverse=True, key=lambda item: item[1]) 
                  if key.split('.')[0] in patches_per_tumor_224}
                
    dict_tumor_img_stats_sorted={k: v for k, v in sorted(dict_tumor_img_stats.items(), key=lambda item: item[1][1])}   
    return dict_tumor_img_stats_sorted


def most_rarely_spatial(dfs, filelist, patches_per_tumor_224):
    # 'bigger' in names means that the condition that that SCM_feature within particular image is bigger 
    # than 95% percentile of SCM_features within all normal data is met
    list_count_bigger={}
    list_names_bigger={}
    list_k_bigger={}
    list_SCM_targets_bigger=[{} for i in range(len(ks))] 
    #list_SCM_targets_k8, list_SCM_targets_k24, list_SCM_targets_48, list_SCM_targets_99
    list_SCM_features_bigger=[{} for i in range(len(ks))] 
    #list_SCM_features_k8, list_SCM_features_k24, list_SCM_features_48, list_SCM_features_99

    # 'smaller' in names means that the condition that that SCM_feature within particular image is bigger 
    # than 95% percentile of SCM_features within all normal data is not met
    list_count_smaller={}
    list_names_smaller={}
    list_k_smaller={}
    list_SCM_targets_smaller=[{} for i in range(len(ks))] 
    #list_SCM_targets_k8, list_SCM_targets_k24, list_SCM_targets_48, list_SCM_targets_99
    list_SCM_features_smaller=[{} for i in range(len(ks))] 
    #list_SCM_features_k8, list_SCM_features_k24, list_SCM_features_48, list_SCM_features_99

    for i, df in enumerate(dfs):
        percentile_95_1 = np.percentile(df[df.label==0]['SCM_targets'], 95)
        a=df[df.label==1][df.SCM_targets>percentile_95_1]
        a_smaller=df[df.label==1][df.SCM_targets<=percentile_95_1]

        details_img_models=extract_img_details(i, a, list_SCM_features_bigger, list_SCM_targets_bigger, 
                                               list_count_bigger, list_names_bigger, list_k_bigger, filelist)
        list_SCM_features_bigger, list_SCM_targets_bigger, list_count_bigger, list_names_bigger, list_k_bigger=details_img_models

        details_img_models_smaller=extract_img_details(i, a_smaller, list_SCM_features_smaller, list_SCM_targets_smaller, 
                                                    list_count_smaller, list_names_smaller, list_k_smaller, filelist)
        list_SCM_features_smaller, list_SCM_targets_smaller, list_count_smaller, list_names_smaller, list_k_smaller=details_img_models_smaller


    dict_bigger=agg_img_bigger_smaller_stats(patches_per_tumor_224, list_k_bigger, list_SCM_targets_bigger, 
                                             list_SCM_features_bigger, list_count_bigger)
    
    dict_smaller=agg_img_bigger_smaller_stats(patches_per_tumor_224, list_k_smaller, list_SCM_targets_smaller, 
                                             list_SCM_features_smaller, list_count_smaller)
    
    
    ## it is done only to facilitate code understanding (could be done without creating new variables)
    most_spatial = dict_bigger.copy() 
    rarely_spatial = dict_smaller.copy() 
 
    # choose images that were selected as 'bigger' by more than half of model-k combinations and avoid overlap
    for (key,v) in most_spatial.items():
        if v[2] > (9*4/2): 
            if key in rarely_spatial.keys():
                del rarely_spatial[key]  
    for (key,v) in rarely_spatial.items(): 
        if key in most_spatial.keys():
            del most_spatial[key]
    
    assert not most_spatial.keys() & rarely_spatial.keys()    ### check if there is no overlap 
    
    return most_spatial, rarely_spatial

def correlation_metric_k(tumor_images_stats, mean_dist):
    for i, metric in enumerate(metric_names[:2][::-1]):
        for k in ks:
            idx=k_dict[k]
            if i>0:
                idx=idx+len(ks)
            a=np.array([el[4+idx] for el in tumor_images_stats.values()])
            corr=np.corrcoef(mean_dist, a)
            print(f"{metric}, k={k}, corr={corr[1,0]}")