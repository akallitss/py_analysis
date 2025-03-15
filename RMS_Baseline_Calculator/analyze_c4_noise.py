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
from scipy.ndimage import uniform_filter1d
import uproot

from TriggerEvaluation.simulation_trigger import moving_average_numpy


def main():
    # base_path = 'C:/Users/Dylan/Desktop/picosec/'
    base_path = '/home/dylan/Desktop/picosec/data/'
    file_name = 'Run224-Pool2_TESTBEAM_tree.root'
    file_path = base_path + file_name
    tree_name = 'RawDataTree'
    channel = 2
    var_names = ['epoch', f'amplC{channel}', 'eventNo']
    event_start = 0
    # event_end = 10000
    event_end = None
    # Open ROOT file
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        events = tree.arrays(var_names, entry_start=event_start, entry_stop=event_end, library='np')

        # Get data from tree
        epochs = events['epoch']
        ampls = events[f'amplC{channel}']
        ampls = np.stack(ampls, axis=0)
        event_num = events['eventNo']

        rmses = np.std(ampls, axis=1)

        # Plot rmses vs event
        plt.figure(figsize=(8, 6))
        plt.scatter(event_num, rmses, alpha=0.5)
        plt.xlabel('Event Number')
        plt.ylabel('RMS')
        plt.tight_layout()

        # Take 50 point moving average of each event ampls
        xs = np.arange(ampls.shape[1])
        xs = np.tile(xs, (ampls.shape[0], 1))
        ampls_ma = []
        for i in range(ampls.shape[0]):
            x_avgs, y_avgs = moving_average_numpy(xs[i], ampls[i], 50)
            ampls_ma.append(y_avgs)
        ampls_ma = np.stack(ampls_ma, axis=0)

        mv_avg_rmses = np.std(ampls_ma, axis=1)
        median_mv_avg_rmses = np.median(mv_avg_rmses)
        std_mv_avg_rmses = np.std(mv_avg_rmses)

        # Plot rmses vs event
        plt.figure(figsize=(8, 6))
        plt.scatter(event_num, mv_avg_rmses, alpha=0.5)
        plt.axhline(median_mv_avg_rmses, color='gray', linestyle='--', label='Median RMS')
        plt.axhline(median_mv_avg_rmses + std_mv_avg_rmses * 3, color='gray', linestyle='--', label='Median + 3 * RMS')
        plt.xlabel('Event Number')
        plt.ylabel('RMS')
        plt.tight_layout()

        plt.show()

    print('donzo')


def moving_average_scipy(x, y, n):
    """
    Calculate the moving average of a waveform in both x and y along axis=0 using numpy.

    Args:
        x: 2D numpy array
        y: 2D numpy array
        n: Window size for moving average

    Returns:
        x_avg: 2D numpy array with moving average applied along axis=0
        y_avg: 2D numpy array with moving average applied along axis=0
    """
    x_avg = uniform_filter1d(x, size=n, axis=0, mode='nearest')
    y_avg = uniform_filter1d(y, size=n, axis=0, mode='nearest')

    return x_avg, y_avg


if __name__ == '__main__':
    main()
