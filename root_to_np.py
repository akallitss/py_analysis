import ROOT
import pandas as pd
import uproot
import numpy as np


def get_tree(run):
    #rootfile = uproot.open(f"/home/evridiki/Downloads/test_run{run}.root");
    rootfile = uproot.open(f"/eos/home-a/akallits/SWAN_projects/pico/Run{run}_parameters.root");
    #rootfile = uproot.open(f"/home/evridiki/Desktop/JULY_RUNS/run{run}.root");
    key = rootfile.keys()
    print(key)

    tree = rootfile["Pico"]
    key = tree.keys()

    print(key)
    #rootfile.close()
    return tree


def get_df(tree, *indexes):
    dataframe = pd.DataFrame() # empty dataframe
    branches =[tree.keys()[i] for i in indexes]

    for df, report in tree.iterate(branches, step_size='1 MB',library='pd', report = True):
        print(report)
        dataframe = pd.concat([dataframe,df], ignore_index=True)

    return dataframe

def rename_tree_branches(df,*col_names):
    l = len(col_names[0])  
    for i in range(l):
        df.rename(columns = {col_names[0][i]:col_names[1][i]}, inplace = True)
    
    return df

def store_new_tree(df):
    newfile = uproot.recreate("./example.root")
    newfile['tree'] = df
    newfile.close()


print('END OF SCRIPT')