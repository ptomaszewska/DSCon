import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

ks=[8, 24, 48, 99]
metric_names=['SCM_features', 'SCM_targets', 'SCM_residuals']
model_combinations=[('swin', 'base', 'window7'), ('swinv2', 'base', 'window8'), 
        ('swinv2', 'base', 'window16'), ('swinv2', 'small', 'window8'), 
        ('swinv2', 'small', 'window16'), ('swinv2', 'tiny', 'window8'), 
        ('swinv2', 'tiny', 'window16'), ('resnetCLAM'), 
        ('vit')
    ]

model_combinations_formatted=[('_').join(model) if 'swin' in model[0] else model for model in model_combinations]
model_combinations_formatted=[model.replace('swin', 's').replace('window', 'w') for model in model_combinations_formatted]

PATH_save_plots='../regression_results_plots/'


def plot_distribution(df, col_hue, model_name, agg='concat', split='test'):
    x_range=[]
    for metric in metric_names:
        df1=df.copy(deep=True)
        df1['label']=df1.label.map({0: 'normal', 1: 'tumor'})
        ax=sns.kdeplot(data=df1, x=metric, hue=col_hue, multiple="fill", common_norm=False, hue_order = ['normal', 'tumor'])
        x_range.append(ax.get_xlim())
        
    xmin=np.min([i[0] for i in x_range])
    xmax=np.max([i[1] for i in x_range])  
    
    for metric in metric_names:   
        df1=df.copy(deep=True)
        df1['label']=df1.label.map({0: 'normal', 1: 'tumor'})
        plt.figure(figsize=(6,2))
        ax=sns.kdeplot(data=df1, x=metric, hue=col_hue, multiple="fill", common_norm=False, hue_order = ['normal', 'tumor'])
        plt.xlabel('')
        ax.set_xlim(xmin, xmax)
        save_path=f"{PATH_save_plots}{split}/{agg}/{metric}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}{model_name}_{metric}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()
        
def k_vs_metric(df_pairwise_features_targets_k, agg='mean'):
    """
    agg: 'mean', 'max'
    """
    n_ks=len(ks)
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=(6.5, 3), sharey=True, sharex=True)
    fig.tight_layout(w_pad=0.001)

    for i, col in enumerate(metric_names[:2]):
        if agg=='max':
            idx=i+5
        else:
            idx=i+3
        ax = ax1 if i ==0 else ax2
        for model_idx in range(0, df_pairwise_features_targets_k.shape[0], n_ks):
            ax.plot(df_pairwise_features_targets_k.iloc[model_idx:(model_idx+n_ks), 1], 
                    df_pairwise_features_targets_k.iloc[model_idx:(model_idx+n_ks),idx], '-o')
        col_parts=col.split('_')
        ax.set_title(f"{agg} {col_parts[0]}" r"$_{{{}}}$".format(col_parts[1]))

        fig.legend(model_combinations_formatted, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3) 

        ax.set_xticks(ks)
        ax.set_xlabel('k')

    plt.savefig(f'{PATH_save_plots}{agg}_diff_vs_k.png', dpi=300, bbox_inches='tight')
    
    
def plot_most_least_spatial(most_spatial, rarely_spatial):
    f=plt.figure(figsize=(7,3))
    scatter=plt.scatter(np.log([el[0] for el in most_spatial.values()]),
                        np.log([el[1] for el in most_spatial.values()]), 
                        c=[el[3] for el in most_spatial.values()],
                        s=200, 
                        alpha=0.7, cmap='seismic',
                        edgecolors='black')
    f.colorbar(scatter)

    scatter=plt.scatter(np.log([el[0] for el in rarely_spatial.values()]),
                        np.log([el[1] for el in rarely_spatial.values()]), 
                        c='black')
    plt.xlabel(r'log ($n_{tumor\_patches})$')
    plt.ylabel(r'log ($dist_{mean}$)')
    plt.savefig(PATH_save_plots+'most_spatial.png', dpi=300, bbox_inches='tight')