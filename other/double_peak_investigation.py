#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 03 12:45 PM 2025
Created in PyCharm
Created as picosec/double_peak_investigation.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import lecroyparser

from Measure import Measure


def main():
    trc_dir = '/media/ucla/picosec/Run358/'
    # plot_waveforms(trc_dir)
    # check_c1_c2_xs(trc_dir)
    check_c1_c2_xs_files(trc_dir)

    print('donzo')


def plot_waveforms(trc_dir):
    trc_file = 'C1--Trace--00153.trc'
    points_per_waveform = 10002
    data = lecroyparser.ScopeData(f'{trc_dir}{trc_file}')
    print(data)
    # Reshape xs and ys by the number of points per waveform
    xs = data.x.reshape(-1, points_per_waveform)
    ys = data.y.reshape(-1, points_per_waveform)
    print(f'xs shape: {xs.shape}')
    print(f'ys shape: {ys.shape}')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(5):
        ax.plot(xs[i], marker='.', ls='none')
    plot_waveform(ys[0], 0.1)
    plt.show()


def check_c1_c2_xs(trc_dir):
    points_per_waveform = 10002
    file_num = '00152'
    trc_c1_file = f'C1--Trace--{file_num}.trc'
    trc_c2_file = f'C2--Trace--{file_num}.trc'
    data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
    data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
    print(f'data_c1.x: {data_c1.x}')
    print(f'data_c2.x: {data_c2.x}')
    print(f'data_c1.diff: {np.diff(data_c1.x)}')
    print(f'data_c2.diff: {np.diff(data_c2.x)}')
    print(f'data_c1.diff hist: {np.histogram(np.diff(data_c1.x), bins=10)}')

    print(f'data_c1.diff mean and std: {np.mean(np.diff(data_c1.x))} +- {np.std(np.diff(data_c1.x))}')

    print(f'data_c2.x - data_c1.x: {data_c2.x - data_c1.x}')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.diff(data_c1.x), bins=100, histtype='step', label='C1')
    ax.hist(np.diff(data_c2.x), bins=100, histtype='step', label='C2')
    ax.set_xlabel('x diff')
    ax.set_ylabel('Counts')
    ax.set_title('C1 and C2 x diff')
    ax.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(data_c2.x - data_c1.x, bins=100, histtype='step', label='C2 - C1')
    ax2.set_xlabel('C2 - C1')
    ax2.set_ylabel('Counts')
    ax2.set_title('C2 - C1')
    ax2.legend()

    plt.show()


def check_c1_c2_xs_files(trc_dir):
    n_files = 265
    file_nums = [str(i).zfill(5) for i in range(0, n_files + 1)]
    c1_diffs, c2_diffs, c2_c1_diffs = [], [], []
    for file_num in file_nums:
        print(f'file_num: {file_num}')
        trc_c1_file = f'C1--Trace--{file_num}.trc'
        trc_c2_file = f'C4--Trace--{file_num}.trc'
        data_c1 = lecroyparser.ScopeData(f'{trc_dir}{trc_c1_file}')
        data_c2 = lecroyparser.ScopeData(f'{trc_dir}{trc_c2_file}')
        c1_diffs_i = np.diff(data_c1.x)
        c2_diffs_i = np.diff(data_c2.x)
        c2_c1_diffs_i = data_c2.x - data_c1.x

        c1_diffs.append(Measure(c1_diffs_i.mean(), c1_diffs_i.std()) * 1e12)  # Convert to ps
        c2_diffs.append(Measure(c2_diffs_i.mean(), c2_diffs_i.std()) * 1e12)  # Convert to ps
        c2_c1_diffs.append(Measure(c2_c1_diffs_i.mean(), c2_c1_diffs_i.std()) * 1e12)  # Convert to ps

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(np.arange(len(c1_diffs)), [c.val for c in c1_diffs], yerr=[c.err for c in c1_diffs], ls='none',
                marker='o', alpha=0.4, label='C1')
    ax.errorbar(np.arange(len(c2_diffs)), [c.val for c in c2_diffs], yerr=[c.err for c in c2_diffs], ls='none',
                marker='o', alpha=0.4, label='C2')
    ax.set_xlabel('File Number')
    ax.set_ylabel('Time Step Between Points (ps)')
    ax.set_title('C1 and C2 x diff')
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(np.arange(len(c2_c1_diffs)), [c.val for c in c2_c1_diffs], yerr=[c.err for c in c2_c1_diffs],
                 marker='o', ls='none', label='C2 - C1')
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Channel Time Offset (ps)')
    ax2.set_title('C2 - C1')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


def plot_waveform(y, dx=0.1, title='Waveform', xlabel='Time (ns)', ylabel='Amplitude (mV)', ax_in=None):
    """ Plot a waveform. """
    if ax_in is None:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax = ax_in
    ax.plot(np.arange(len(y)) * dx, y)

    return ax


def get_lecroy_nsegments(scope_data):
    """
    Get the number of segments in the Lecroy file. --> Number of files
    :param scope_data: ScopeData object
    :return: number of segments
    """
    n_segments = scope_data.parseInt16(144)
    return n_segments


if __name__ == '__main__':
    main()
