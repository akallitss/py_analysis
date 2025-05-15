#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 03 2:06 PM 2025
Created in PyCharm
Created as pico_py_analysis/get_osc_offset_corrections.py

@author: Dylan Neff, dylan
"""

import os
import uproot
import numpy as np
import lecroyparser
from datetime import datetime


def main():
    local_test()
    # run_from_files_directory()

    print('donzo')


def run_from_files_directory():
    # run_periods = ['2024_June_h4', '2023_April_h4', '2023_July_h4', '2023_August_h4', '2024_September_h4']
    run_periods = ['2023_April_h4', '2023_July_h4', '2023_August_h4', '2024_September_h4']
    for run_period in run_periods:
        run_pools_input_dir = f'/eos/project-p/picosec/analysis/Saclay/data/{run_period}/processedTrees'
        out_root_path = f'/eos/project-p/picosec/analysis/Saclay/data/{run_period}/oscilloscope_time_offsets_{run_period}.root'
        log_path = f'/eos/project-p/picosec/analysis/Saclay/data/{run_period}/oscilloscope_time_offsets_{run_period}.log'
        trc_top_dir = f'/eos/project-p/picosec/testbeam/{run_period}/'

        file_list = os.listdir(run_pools_input_dir)
        file_list = [f for f in file_list if f.endswith('.root')]
        run_pool_list = []
        for file_name in file_list:
            run, pool = get_run_pool_from_filename(file_name)
            if run is None or pool is None:
                print_log(f"Skipping input file {file_name} due to parsing error.", log_path)
                continue
            run_pool_list.append((run, pool))

        trees = {}
        for run, pool in run_pool_list:
            print_log(f'Starting run {run}, pool {pool} for {run_period}', log_path)
            pool_dir = f'Pool{pool}/' if pool != 8 else 'LECROY/'
            if not os.path.isdir(f'{trc_top_dir}{pool_dir}'):
                print_log(f"Directory {trc_top_dir}{pool_dir} does not exist. Skipping.", log_path)
                continue

            run_dir = f'Run{run:03d}/'
            if not os.path.isdir(f'{trc_top_dir}{pool_dir}{run_dir}'):
                print_log(f"Directory {trc_top_dir}{pool_dir}{run_dir} does not exist. Skipping.", log_path)
                continue

            trc_dir = f'{trc_top_dir}{pool_dir}{run_dir}'
            try:
                data_per_channel = get_directory_offset_corrections(trc_dir, log_path)
            except Exception as e:
                print_log(f"Error processing directory {trc_dir}: {e}", log_path)
                continue
            trees.update({f'Run{run}_Pool{pool}': data_per_channel})

        with uproot.recreate(out_root_path) as root_file:  # Write to ROOT file
            for tree_name, data_per_channel in trees.items():
                if tree_name in root_file:  # Check if the tree already exists
                    print_log(f"Tree '{tree_name}' already exists. Overwriting.", log_path)
                else:
                    print_log(f"Creating new tree '{tree_name}'.", log_path)
                root_file[tree_name] = data_per_channel  # Write the data to the ROOT file

        print(f"Wrote TTree '{tree_name}' to {out_root_path}")


def local_test():
    # run_num = 358
    # pool_num = 8
    # trc_dir = f'/media/ucla/picosec/Run{run_num}/'
    # out_root_path = f'/media/ucla/picosec/Run{run_num}/oscilloscope_time_offsets.root'
    # tree_name = f"Run{run_num}_Pool{pool_num}"
    #
    # data_per_channel = get_directory_offset_corrections(trc_dir)
    #
    # # Write to ROOT file
    # with uproot.recreate(out_root_path) as root_file:
    #     root_file[tree_name] = data_per_channel
    #
    # print(f"Wrote TTree '{tree_name}' to {out_root_path}")

    run_num = 296
    pool_num = 2
    trc_dir = f'/data/akallits/Saclay_Analysis/data/data/2023_August_h4/Run{run_num}/'
    out_root_path = f'/data/akallits/Saclay_Analysis/data/data/2023_August_h4/Run{run_num}/oscilloscope_time_offsets.root'
    tree_name = f"Run{run_num}_Pool{pool_num}"

    data_per_channel = get_directory_offset_corrections(trc_dir)

    # Write to ROOT file
    with uproot.recreate(out_root_path) as root_file:
        root_file[tree_name] = data_per_channel

    print(f"Wrote TTree '{tree_name}' to {out_root_path}")


def get_directory_offset_corrections(trc_dir, log_path=None):
    """
    Get the time offsets for all trc files for channels in the directory.
    Args:
        trc_dir:
        log_path:

    Returns:

    """
    channel_files = get_trc_files_by_channel(trc_dir)

    data_per_channel = {}  # Dicts to hold data per channel

    for channel, files in channel_files.items():
        print_log(f'Channel {channel}, Files: {len(files)}', log_path)
        event_nums = []
        time_offsets = []
        event_num = 0

        for trc_file in files:
            print(f'Channel {channel}, File: {trc_file}')
            try:
                data = lecroyparser.ScopeData(f'{trc_dir}{trc_file}')
                n_waveforms = get_lecroy_nsegments(data)
                trigger_time, trigger_offset = get_trigger_offset(data)
            except Exception as e:
                print_log(f"Skipping file{trc_file} due to error {e}", log_path)
                continue

            for i in range(n_waveforms):
                event_nums.append(event_num)
                time_offsets.append(trigger_offset[i] * 1e9)  # Convert to ns
                event_num += 1

        data_per_channel[f"{channel}_event_num"] = np.array(event_nums, dtype=np.int32)
        data_per_channel[f"{channel}_time_offset"] = np.array(time_offsets, dtype=np.float32)

    # Ensure all branches are same length
    # branch_lengths = {len(arr) for arr in data_per_channel.values()}
    # if len(branch_lengths) > 1:
    #     message = f"Branches are different lengths: {branch_lengths}"
    #     print_log(message, log_path)
    #     raise ValueError(message)

    return data_per_channel


def get_trc_files_by_channel(trc_dir):
    """
    Get all trc files in the directory and sort them by channel.
    Returns:
        dict: {channel: [trc_files]}
    """
    trc_files = os.listdir(trc_dir)
    trc_files = [f for f in trc_files if f.endswith('.trc')]

    channels = {}
    for f in trc_files:
        channel = f.split('--')[0]
        if channel not in channels:
            channels[channel] = []
        channels[channel].append(f)

    # For each channel, sort the files by the number in the filename
    for channel in channels:
        channels[channel].sort(key=lambda x: int(x.split('--')[2].split('.')[0]))

    return channels


def get_run_pool_from_filename(filename):
    """
    Get the run and pool number from the filename.
    Args:
        filename:

    Returns:

    """
    try:
        # Remove any directory path and extension
        base_name = filename.split('/')[-1].replace('.root', '')

        # Isolate the part like 'Run303-Pool3'
        main_part = base_name.split('_')[0]

        # Split into 'Run303' and 'Pool3'
        run_str, pool_str = main_part.split('-')

        # Extract the numeric parts
        run_number = int(run_str.replace('Run', ''))
        pool_number = int(pool_str.replace('Pool', ''))
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename '{filename}': {e}")
        return None, None

    return run_number, pool_number


def get_lecroy_nsegments(scope_data):
    """
    Get the number of segments in the Lecroy file. --> Number of files
    :param scope_data: ScopeData object
    :return: number of segments
    """
    n_segments = scope_data.parseInt16(144)
    return n_segments


def get_trigger_offset(scope_data):
    """
    Get trigger time and offset from the Lecroy file.
    Trigger time is the time of the current trigger from the first trigger.
    Trigger offset is the time of the first point from the trigger time.
    Args:
        scope_data:

    Returns:

    """
    if scope_data.trigTimeArray == 0:
        raise ValueError("No TRIGTIME_ARRAY section found in the file.")

    pos = scope_data.posWAVEDESC + scope_data.waveDescriptor + scope_data.userText

    n_segments = get_lecroy_nsegments(scope_data)
    trigger_array = np.frombuffer(scope_data.data, dtype=scope_data.endianness + "f8", count=n_segments * 2, offset=pos)

    trigger_time = trigger_array[::2]
    trigger_offset = trigger_array[1::2]
    return trigger_time, trigger_offset


def print_log(message, log_path=None):
    """
    Print message to log file
    :param message:
    :param log_path:
    :return:
    """
    print(message)
    if log_path is not None:
        with open(log_path, 'a') as log_file:
            date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_file.write(f'{date_time_str}: {message}\n')


if __name__ == '__main__':
    main()
