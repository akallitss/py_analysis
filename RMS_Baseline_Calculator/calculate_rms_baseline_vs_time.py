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
import uproot

from TriggerEvaluation.simulation_trigger import moving_average_numpy

# import ROOT


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
        time_from_start = epochs - epochs[0]
        ampls = events[f'amplC{channel}']
        ampls = np.stack(ampls, axis=0)
        event_num = events['eventNo']

        baselines = np.mean(ampls[:, 0:1000], axis=1)

        print(time_from_start)
        print(baselines)
        print(epochs)

        print(time_from_start.shape)
        print(baselines.shape)

        # Plot baseline vs time
        plt.figure(figsize=(8, 6))
        plt.scatter(time_from_start, baselines, alpha=0.5)
        plt.xlabel('Run Time (s)')
        plt.ylabel('Baseline')
        plt.tight_layout()

        # Group by common epoch, plot baseline vs event num as new series for each epoch num
        epoch_nums = np.unique(epochs)
        colors = plt.cm.viridis(np.linspace(0, 1, len(epoch_nums)))
        fig_raw, ax_raw = plt.subplots(figsize=(8, 6))
        fig_mv_avg, ax_mv_avg = plt.subplots(figsize=(8, 6))
        for i, epoch_num in enumerate(epoch_nums):
            epoch_mask = epochs == epoch_num
            epoch_event_num = event_num[epoch_mask] - event_num[epoch_mask][0]
            ax_raw.scatter(epoch_event_num, baselines[epoch_mask], alpha=0.5, color=colors[i], label=f'Epoch {epoch_num}')
            # Calculate 80 point moving average and plot
            event_num_avg, baseline_avg = moving_average_numpy(epoch_event_num, baselines[epoch_mask], 80)
            ax_raw.plot(event_num_avg, baseline_avg, color=colors[i])
            ax_mv_avg.plot(event_num_avg, baseline_avg, color=colors[i], label=f'Epoch {epoch_num}')
        ax_raw.set_xlabel('Event Number')
        ax_raw.set_ylabel('Baseline')
        # ax_raw.legend(loc='best')
        fig_raw.tight_layout()
        ax_mv_avg.set_xlabel('Event Number')
        ax_mv_avg.set_ylabel('Baseline')
        # ax_mv_avg.legend(loc='best')
        fig_mv_avg.tight_layout()

        plt.show()

    print('donzo')



if __name__ == '__main__':
    main()
