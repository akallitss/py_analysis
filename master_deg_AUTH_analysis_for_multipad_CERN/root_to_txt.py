#import ROOT
import numpy as np
import pandas as pd
from root_to_np import get_tree, get_df, rename_tree_branches

run = [636]
pad_name = [37,38,27,28]
color = np.array(['red','blue','purple','green'])
which_mcp = [1,1,2,2]

# get events
data = []
for irun in run:
    tree = get_tree(irun)
    
    var_indexes = np.array([1,4,5,7])
    for index in range(4):
        var_indexes = np.append(var_indexes,[index*10+8,index*10+12,index*10+11]) 
    var_indexes = np.append(var_indexes,[67,65,66])
    
    data_single = get_df(tree,*var_indexes)
    
    old_names = ['mcp_gpeak','mcp_t']
    new_names = ['mcp1_gpeak','mcp1_t']
    names = [old_names,new_names]
    data_single = rename_tree_branches(data_single,*names)
    
    print(len(data_single))
    data.append(data_single)

data = pd.concat(data)
print(data.head())
print(len(data))

#define the cuts

mcp_overflow = np.logical_and((data.mcp1_gpeak>0.3)*(data.mcp1_gpeak<=0.74),
                             (data.mcp2_gpeak>0.3)*(data.mcp2_gpeak<=0.74))
chi2_cut = (data.chi2<=40)*(data.track_flag>0)
mcp_basic_cut = mcp_overflow*chi2_cut


#a subset of our df
#to rearange cols
rearange_cols = ['xy[0]','xy[1]','mm1_gpeak','mm1_qe','mm1_t','mcp1_t',
                 'mm2_gpeak','mm2_qe','mm2_t','mcp1_t',
                 'mm3_gpeak','mm3_qe','mm3_t','mcp2_t',
                 'mm4_gpeak','mm4_qe','mm4_t','mcp2_t']
for_tzam = data[mcp_basic_cut][rearange_cols]
print(len(for_tzam))

#for_tzam.iloc[45000:, :].to_csv(f'/home/evridiki/Desktop/JULY_RUNS/txtdata_run{run[0]}_2.txt', sep=' ',  header=False, index=False)
for_tzam.iloc[45000:, :].to_csv(f'/eos/home-a/akallits/SWAN_projects/pico/txtdata_run{run[0]}_2.txt', sep=' ',  header=False, index=False)