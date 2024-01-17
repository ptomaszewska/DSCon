import pandas as pd
import numpy as np
import argparse


def set_label (row):
    if "normal" in row['slide_id']:
        return 0
    elif "tumor" in row['slide_id']:
        return 1
    elif "melanoma" in row['slide_id']:
        return 1
    elif "nevi" in row['slide_id']:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        help="Data source")
    
    parser.add_argument('--patch_size',
                        type=int,
                        help="Patch size")
    parser.add_argument('--split',
                        type=str,
                        help="Split")

    args = parser.parse_args()
    data=args.data
    patch_size=args.patch_size
    split=args.split
    df=pd.read_csv("../"+data+"_patches_"+str(patch_size)+"/"+split+"/process_list_autogen.csv")
    df_new=df['slide_id']
    df_new=df_new.to_frame()
    df_new.columns=['slide_id']
    df_new['label']= df_new.apply (lambda row: set_label(row), axis=1)
    df_new['case_id']=np.arange(df_new.shape[0])
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new['slide_id']=df_new['slide_id'].str.split(".").str[0]
    df_new.to_csv("./dataset_csv/"+data+"_"+"labels_"+split+".csv", index=False)

if __name__ == "__main__":
    main()
