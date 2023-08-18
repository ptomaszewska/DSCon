#!/usr/bin/env python
# coding: utf-8

# Graphics
import matplotlib.pyplot as plt
import seaborn
from pysal.viz import splot
from splot.esda import plot_moran
#import contextily

# Analysis
import geopandas as gpd
import pysal
from pysal.explore import esda
from pysal.lib import weights
from numpy.random import seed
from shapely.geometry import Polygon
from pysal.model import spreg

import h5py    
import numpy as np    
import pandas as pd

import os
import glob

import random
random.seed(123)
np.random.seed(123)

import sklearn
import argparse


def corr_drop(df):
    c=df.iloc[:,2:].corr()
    c_abs=c.abs()

    # Select upper triangle of correlation matrix
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]  

    # Drop features 
    df.drop(to_drop, axis=1, inplace=True)
    return df


def create_gdf(df, patch_size):
    polys=[]
    df.x=df.x
    df.y=df.y
    step=int(patch_size)    
    for x, y in zip(df.x, df.y):
       poly = Polygon([(x, y), (x + step, y), (x + step, y + step), (x, y + step)])
       polys.append(poly)

    polys = gpd.GeoSeries(polys)

    df['geometry']=polys
    gdf = gpd.GeoDataFrame(df)
    return gdf



def Wx_regression(gdf, w):
    
    variable_names=gdf.columns.tolist()
    variable_names=variable_names[3:-1]  
    
    wx = (
        gdf.filter(
            regex="\d"
            # Compute the spatial lag of each of those variables
        )
        .apply(
            lambda y: weights.spatial_lag.lag_spatial(w, y)
            # Rename the spatial lag, adding w_ to the original name
        )
        .rename(
            columns=lambda c: "w_"
            + str(c)
            # Remove the lag of the binary variable for apartments
        )
    )
    # Merge original variables with the spatial lags in `wx`
    slx_exog = gdf[variable_names].join(wx)
    # Fit linear model with `spreg`
    wx_regression = spreg.OLS(
        # Dependent variable
        gdf[["att"]].values,
        # Independent variables
        slx_exog.values,
        # Dependent variable name
        name_y="att",
        # Independent variables names
        name_x=slx_exog.columns.tolist(),
        moran=True,
        spat_diag=True,
        w=w ,                        
        white_test=True,
        vm=True
    )
    
    moran_value=wx_regression.moran_res[0]
    moran_value_stand=wx_regression.moran_res[1] 
    moran_p_value=wx_regression.moran_res[2] 
    n_important=sum(map(lambda x : x<0.05 == 1, wx_regression.t_stat[1][1:]))
    mse=sklearn.metrics.mean_squared_error(wx_regression.y, wx_regression.predy)

    return {'wx_r2':wx_regression.r2, 'wx_Moran': moran_value, 'wx_Moran_std': moran_value_stand, 'wx_Moran_p':moran_p_value, 'Wx_n_imp': n_important, 'wx_mse': mse }


def lag_regression(gdf, w):
    
    variable_names=gdf.columns.tolist()
    variable_names=variable_names[3:-1]  

    lag_regression = spreg.GM_Lag(
        # Dependent variable
        gdf[["att"]].values,
        # Independent variables
        gdf[variable_names].values,
        # Spatial weights matrix
        w=w,
        # Dependent variable name
        name_y="att",
        # Independent variables names
        name_x=variable_names,
    )
    
    l=[x[1] for x in lag_regression.z_stat]
    n_important=sum(1 for i in l if i<0.05)
    mse=sklearn.metrics.mean_squared_error(lag_regression.y, lag_regression.predy)
    
    
    return {'lag_r2':lag_regression.pr2, 'lag_n_imp': n_important, 'lag_mse': mse} 

def error_regression(gdf, w):
    
    variable_names=gdf.columns.tolist()
    variable_names=variable_names[3:-1]  

    error_regression = spreg.GM_Error(
        # Dependent variable
        gdf[["att"]].values,
        # Independent variables
        gdf[variable_names].values,
        # Spatial weights matrix
        w=w,
        # Dependent variable name
        name_y="att",
        # Independent variables names
        name_x=variable_names,
    )
    
    l=[x[1] for x in error_regression.z_stat]
    n_important=sum(1 for i in l if i<0.05)
    mse=sklearn.metrics.mean_squared_error(error_regression.y, error_regression.predy)
    
    return {'error_r2':error_regression.pr2,  'error_n_imp': n_important, 'error_mse': mse}  
    
def regression_normal(gdf, w):
    variable_names=gdf.columns.tolist()
    variable_names=variable_names[3:-1]  
    # Fit OLS model
    regression= spreg.OLS(
        # Dependent variable
        gdf['att'].values,            
        # Independent variables
        gdf[variable_names].values,
        # Dependent variable name
        name_y="att",
        # Independent variable name
        name_x=variable_names,
        moran=True,
        spat_diag=True,
        w=w ,                         
        white_test=True,
        vm=True
    )
    
    moran_value=regression.moran_res[0]
    moran_value_stand=regression.moran_res[1] # standarized value
    moran_p_value=regression.moran_res[2] # p-value
    n_important=sum(map(lambda x : x<0.05 == 1, regression.t_stat[1][1:]))
    mse=sklearn.metrics.mean_squared_error(regression.y, regression.predy)
    
    return {'r2':regression.r2, 'Moran': moran_value, 'Moran_std': moran_value_stand, 'Moran_p':moran_p_value, 'n_imp': n_important, 'mse': mse}  
    


def calculate(label, fold):
    c=0
    filenames_list= glob.glob(PATH_att+"/"+label+"/*/*", recursive = True )
    for i, filename_path in enumerate(filenames_list):
      filename = filename_path.split("/")[-1]
      patch_size=int(PATH_att_base.split("/")[-2].split('_')[1])
      lesion_type=filename.split('_')[0]
      print(filename_path, flush=True)
      
      save_path="./regression_results/k_"+str(k)+"_"+PATH_att_base.split('/')[5]+'_'+str(fold)+'.csv'

      coords_att=h5py.File(filename_path, 'r') 
  
      att=coords_att['attention_scores'][:]
  
      coords=coords_att['coords'][:]
  
      d=np.concatenate((coords, att), axis=1) 
  
      df_a=pd.DataFrame(d, columns=['x', 'y', 'att'])
  
      coords_feat=h5py.File(PATH_feat+"_".join(filename.split('_')[:-1])+".h5", 'r')
  
      coords_f=att=coords_feat['coords'][:]
      features=att=coords_feat['features'][:]
  
      n_features=features.shape[1]
  
      d_f=np.concatenate((coords_f, features), axis=1) 
  
      names_feat=list(range(0,n_features))
      names_feat=[str(i)+"_feat" for i in names_feat]
      df_f=pd.DataFrame(d_f, columns=['x', 'y']+names_feat)
  
      d_final=pd.merge(df_a, df_f, on=['x', 'y'])
  
      df=d_final
      df=corr_drop(df)
      n_correlated=n_features-df.shape[1]
      gdf=create_gdf(df, patch_size)
      
      w = weights.KNN.from_dataframe(gdf, k=k)
      w.transform = "R"

      print('regression', flush=True)   
      if (gdf.shape[1]-3)>gdf.shape[0]:  
            print('less patches than features, no regression is run')
            c=c+1 
            continue
      r=regression_normal(gdf, w)
      if ((gdf.shape[1]-3)*2)<gdf.shape[0]:
            r_Wx=Wx_regression(gdf, w)
      else:
            r_Wx={'wx_r2':-100, 'wx_Moran': -100, 'wx_Moran_std': -100, 'wx_Moran_p': -100, 'Wx_n_imp': -100, 'wx_mse': -100}
      r_lag=lag_regression(gdf, w)
      r_error=error_regression(gdf,w)
      
      results = {**r, **r_Wx, **r_lag, **r_error}
      results['filename']=filename
      results['label']=label
      results['n_correlated']=n_correlated
      
      df = pd.DataFrame.from_records([results] )
      df.to_csv(save_path, index=False, mode='a', header = not os.path.exists(save_path))


parser = argparse.ArgumentParser(description='Configurations for regression')
parser.add_argument('--model_type', type=str, default=None, 
                    help='model_type')
parser.add_argument('--model_size', type=str, default=None, 
                    help='model_size')
parser.add_argument('--lesion_type', type=int)
parser.add_argument('--fold', type=int)
parser.add_argument('--window_size', type=int)
parser.add_argument('--k', type=int)
args = parser.parse_args()

if args.model_type=="swinv2":
        PATH_att_base="./heatmaps/heatmap_raw_results/CAMELYON16_256_"+args.model_type+"_"+args.model_size+"_patch4_window"+str(args.window_size)+"_256_weighted_50_0.005_0.0_sgd_0.7_False_0.25_10_50_True_512_128/"         
          PATH_feat="../CAMELYON16_features_256_"+args.model_type+"_"+args.model_size+"_patch4_window"+str(args.window_size)+"_256/all/h5_files/"
elif args.model_type=="vit":
    PATH_att_base="./heatmaps/heatmap_raw_results/CAMELYON16_224_vit_weighted_50_0.005_0.0_sgd_0.7_False_0.25_10_50_True_512_128/"
    PATH_feat="../CAMELYON16_features_224_vit/all/h5_files/"
elif args.model_type=="resnetCLAM":
    PATH_att_base="./heatmaps/heatmap_raw_results/CAMELYON16_224_resnetCLAM_weighted_50_0.0002_1e-05_adam_0.7_False_0.25_10_50_True_512_256/"
    PATH_feat="../CAMELYON16_features_224_resnetCLAM/all/h5_files/"   
elif args.model_type=="swin": 
        PATH_att_base="./heatmaps/heatmap_raw_results/CAMELYON16_224_swin_base_patch4_window7_224_22k_weighted_50_0.005_0.0_sgd_0.7_False_0.25_10_50_True_512_128/"
    PATH_feat="../CAMELYON16_features_224_swin_base_patch4_window7_224_22k/all/h5_files/" 

  

k=args.k
  
PATH_att=PATH_att_base+str(args.fold)
calculate(str(args.lesion_type), args.fold)