#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on April 08 09:55 2025
Created in PyCharm
Created as py_analysis/scan_read_test.py

@author: Alexandra Kallitsopoulou, akallits
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

from single_pad_analysis.analysis_functions import *


def main():
    # run_path = '/home/akallits/Documents/PicoAnalysis/Saclay_Analysis/data/2023_April_h4/processedTrees/ParameterTrees/Run243-Pool4_treeParam.root'
    run_path = '/home/akallits/Documents/PicoAnalysis/Saclay_Analysis/data/2023_April_h4/processedTrees/ParameterTrees/Run325-Pool3_treeParam.root'
    tree = get_tree(run_path, 'ParameterTree')

    channels = ['C1', 'C2', 'C4']
    global_vars = ['eventNo', 'chi2track']
    channel_vars = ['hitX', 'hitY']
    peak_param_channel_vars = ['ampl', 'dampl', 'charge', 'echarge', 'echargefit', 'echargefixed', 'totcharge',
                               'totchargefixed', 'risetime', 'tfit20', 'tfit20_nb', 'tnaive20', 'sigmoidR[4]']

    branches = []
    for channel in channels:
        for channel_var in channel_vars:
            branches.append(f'{channel_var}_{channel}')
        for var in peak_param_channel_vars:
            branches.append(f'peakparam_{channel}/peakparam_{channel}.{var}')
    for var in global_vars:
        branches.append(var)
    timer_start = time()
    df = get_df_branches(tree, branches, step_size='1000 MB')
    timer_end = time()
    print(f'Time to read the tree: {timer_end - timer_start:.2f} seconds')

    df = df[df['eventNo'] > 350000]

    print(f'Start processing: {df["eventNo"].shape[0]} events')
    timer_start = time()
    make_chi2_cut_tracks(df, channels, 1.0, plot=False)
    timer_end = time()
    print(f'Time to process the tree: {timer_end - timer_start:.2f} seconds')

    timer_start = time()
    make_chi2_cut_tracks(df, channels, 1.0, plot=True)
    timer_end = time()
    print(f'Time to process the tree: {timer_end - timer_start:.2f} seconds')

    # plt.show()

    print('bonzo')


if __name__ == '__main__':
    main()
