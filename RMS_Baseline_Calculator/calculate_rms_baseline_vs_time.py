#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 15 4:32 PM 2025
Created in PyCharm
Created as pico_py_analysis/calculate_rms_baseline_vs_time.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import ROOT


def main():
    base_path = 'C:/Users/Dylan/Desktop/picosec/'
    file_name = 'Run224-Pool2_TESTBEAM_tree.root'
    file_path = base_path + file_name
    tree_name = 'RawDataTree'
    var_names = ['epoch', 'ampl']
    event_start = 0
    event_end = 1000
    # Open ROOT file
    file = ROOT.TFile.Open(file_path, "READ")
    if not file or file.IsZombie():
        print(f"Error: Could not open file {file_path}")
        return

    # Get the TTree
    tree = file.Get(tree_name)
    if not tree:
        print(f"Error: Could not find tree {tree_name} in file {file_path}")
        return

    # Initialize storage for branch data
    data = {var: [] for var in var_names}

    # Loop over events in the specified range
    for i in range(event_start, min(event_end, tree.GetEntries())):
        tree.GetEntry(i)
        for var in var_names:
            data[var].append(getattr(tree, var))

    # Convert lists to NumPy arrays
    for var in var_names:
        data[var] = np.array(data[var])

    print(data)
    print("donzo")

    file.Close()
    print('donzo')


if __name__ == '__main__':
    main()
