import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

ks=[8, 24, 48, 99]
k_dict={8:0, 24:1, 48:2, 99:3} ## ordering

PATH_tumor_stats='../tumor_stats'

def custom_read_csv(path, split='test'):
    """
    if split is different that 'train' and 'test' then whole file is loaded (train+test)
    """
    df=pd.read_csv(path)
    df=df.drop_duplicates(subset='filename', keep='last') 
    df=df[df.wx_r2!=-100]  # -100 means regression was not perform (fewer features than number of patches (mostly in Wx))
    if split=='test':
        df=df[df['filename'].str.contains(split) ==True]
    elif split=='train':
        df=df[df['filename'].str.contains('test') ==False]
    return df

def include_tumor_info(df, patches_per_tumor, n_tumor_patches, tumor_patches_dist):
    data=[]
    for i in range(2):
        data.append([patches_per_tumor[x][i] if x in patches_per_tumor else 0
                                for x in 
                                df['filename'].str.split('.').str[0] ])
    n_tumor_patches.extend(data[0])
    tumor_patches_dist.extend(data[1])
 
    return n_tumor_patches, tumor_patches_dist


def extract_from_df(df_p, col_name='filename', start_idx=4):
    df_p['model']=df_p[col_name].str.split("/").str[-1].str.split("_").str[start_idx]
    df_p["size"] = df_p[col_name].str.split("/").str[-1].str.split("_").str[start_idx+1]
    df_p["window_size"] = df_p[col_name].str.split("/").str[-1].str.split("_").str[start_idx+3]
    df_p['window_size'][~df_p.model.isin(['swinv2','swin'])]=''
    df_p['size'][~df_p.model.str.contains('swin')]=''
    df_p['k']= df_p[col_name].str.split("/").str[-1].str.split("_").str[1].astype(int)
    return df_p

def get_annot(patch_size):
    annot_train= glob.glob(f"{PATH_tumor_stats}/training/{patch_size}/*.csv")
    annot_test= glob.glob(f"{PATH_tumor_stats}/testing/{patch_size}/*.csv")
    annot=annot_train+annot_test
    return annot

def get_tumor_data(patches_per_tumor, annot):
    for file in annot:
        df=pd.read_csv(file)
        patches_per_tumor[("_").join(file.split("/")[-1].
                                     split('_')[:2])+"_blockmap"] = [np.sum(df.ill_cells_area>0), 
                                                                    compute_average_distance(
                                                                            df.loc[:,['patch_top_left_x', 
                                                                                  'patch_top_left_y']])] 
    return patches_per_tumor

def compute_average_distance(X):
    return np.mean(pdist(X))

def mean_list(list_SCM_targets, list_SCM_features, key):
    resulting_mean_list=[]
    for l in [list_SCM_targets, list_SCM_features]:
        for i in range(len(ks)):
            resulting_mean_list.append(np.mean(l[i][key]))
    return resulting_mean_list

def mean_list_global(list_SCM_targets, list_SCM_features, key):
    resulting_mean_list=[]
    for l in [list_SCM_targets, list_SCM_features]:
        resulting_mean_list_l=[]
        for i in range(len(ks)):
            resulting_mean_list_l.extend(l[i][key])
        resulting_mean_list.append(np.mean(resulting_mean_list_l))
    return resulting_mean_list

def agg_img_bigger_smaller_stats(patches_per_tumor_224, list_k, list_SCM_targets, list_SCM_features, list_count):
    return {key:patches_per_tumor_224[key.split('.')[0]]+
                  [v]+
                  [np.mean(list_k[key])]+
                mean_list_global(list_SCM_targets, list_SCM_features, key)
                for key, v in sorted(list_count.items(), reverse=True, 
                                     key=lambda item: item[1]) if key.split('.')[0] in patches_per_tumor_224}

def extract_img_details(i, df, list_SCM_features, list_SCM_targets, list_count, list_names, list_k, filelist):
    for name in df.filename.values:
        k=int(sorted(filelist)[5*i].split('/')[-1].split('_')[1])
        if name not in list_count.keys():
            list_count[name]=1
            list_names[name]=[sorted(filelist)[5*i].split('/')[-1]]
            list_k[name]=[k]
            list_SCM_features, list_SCM_targets=initialize_dicts(name, list_SCM_features, list_SCM_targets)
            extend_dicts(df, name, list_SCM_features, list_SCM_targets, k)
        else:
            list_count[name]+=1
            list_names[name].append(sorted(filelist)[5*i].split('/')[-1])
            list_k[name].append(int(sorted(filelist)[5*i].split('/')[-1].split('_')[1]))
            extend_dicts(df, name, list_SCM_features, list_SCM_targets, k)
    return [list_SCM_features, list_SCM_targets, list_count, list_names, list_k]

def initialize_dicts(name, list_SCM_features, list_SCM_targets):
    for i in range(len(ks)):
        list_SCM_features[i][name]=[]
        list_SCM_targets[i][name]=[]
    return  list_SCM_features, list_SCM_targets

def extend_dicts(df, name, list_SCM_features, list_SCM_targets, k):
    list_SCM_features[k_dict[k]][name].extend([df[df.filename==name].loc[:,'SCM_features'].values[0]])
    list_SCM_targets[k_dict[k]][name].extend([df[df.filename==name].loc[:,'SCM_targets'].values[0]])
    return  list_SCM_features, list_SCM_targets


def prepare_performance_df(csv_path):
    df_perf = pd.read_csv(csv_path, index_col=False)
    df_perf['classifier_name']=df_perf['results_dir'].str.split('/').str[-1].str[15:-60]
    df_perf=df_perf[~df_perf.classifier_name.str.contains('pcam')]
    df_perf['classifier_name']=[i.replace('patch4_','') if 'patch4_' in i else i for i in df_perf['classifier_name']]
    df_perf['classifier_name']=[i.replace('_256','') if '_256' in i else i for i in df_perf['classifier_name']]
    df_perf['classifier_name']=[i.replace('_224','') if '_224' in i else i for i in df_perf['classifier_name']]
    df_perf['classifier_name']=[i.replace('_22k','') if '_22k' in i else i for i in df_perf['classifier_name']]
    df_perf['classifier_name']=[i.replace('_wei','') if 'wei' in i else i for i in df_perf['classifier_name']]

    df_perf['classifier_name']=[i.replace('swin','s') if 'swin' in i else i for i in df_perf['classifier_name']]
    df_perf['classifier_name']=[i.replace('window','w') if 'window' in i else i for i in df_perf['classifier_name']]
    ## above 2 changes for brevity

    df_perf['accuracy']=df_perf['test_global_acc']
    df_perf['dataset_name']=df_perf['Name'].str.split('_').str[-6]

    return df_perf