#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on January 20 17:41 2025
Created in PyCharm
Created as py_analysis/MakeTreePicosecProcess.py

@author: akallitss, akallits
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    base_dir = '/home/akallits/Documents/PicoAnalysis/Saclay_Analysis/'  # Laptop ubuntu
    # base_dir = '/eos/project-p/picosec/analysis/Saclay/'  # EOS Server

    logbook_dir = f'{base_dir}data/2023_August_h4/'
    output_dir = f'{base_dir}data/2023_August_h4/processedTrees/'
    # logbook_dir = '/home/akallits/Documents'
    logbook_name = 'OsciloscopeSetup_LogbookAll.txt'
    logbook_path = os.path.join(logbook_dir, logbook_name)
    # logbook = pd.read_csv(logbook_path, sep='\t', header=0)
    # print(logbook)
    expected_line_length = 27
    crash_byte_threshold = 1000  # bytes Output root files smaller than this are considered crashed

    # Read run info from logbook
    log_run_info_dict  = read_logbook(logbook_path, expected_line_length)

    # Read runs already processed
    processed_runs, crashed_runs = check_processed_runs(output_dir, crash_byte_threshold)

    # Process runs which are not already processed
    for run_info in log_run_info_dict:
        if (int(run_info['Run']), int(run_info['Pool'])) not in processed_runs:
            print(f'Run {run_info["Run"]}, Pool {run_info["Pool"]} not processed yet.')
            # Process run
            # process_run(run_info, base_dir)
        else:
            print(f'Run {run_info["Run"]}, Pool {run_info["Pool"]} already processed.')

    print('bonzo')


def read_logbook(logbook_path, expected_line_length):
    """
    Read logbook file and return dataframe
    :param logbook_path:
    :param expected_line_length:
    :return:
    """
    headers, runs = None, []
    with open(logbook_path, 'r') as logbook:
        for i, line in enumerate(logbook):
            line = line.strip().split()
            line = [x.strip() for x in line]
            if len(line) != expected_line_length:
                continue
            if headers is None:
                headers = line
            else:
                runs.append(line)
            # print(f'Line #{i} {len(line)} : {line[0]}')

    # No pandas on server, eliminate dependency :(
    # # Make dataframe with headers and runs
    # df = pd.DataFrame(runs, columns=headers)
    # print(df)
    # return df

    logbook_dict = [{headers[i]: run[i] for i in range(len(headers))} for run in runs]

    return logbook_dict


def check_processed_runs(output_dir, crash_byte_threshold):
    """
    Check which runs have already been processed
    :param output_dir:
    :param crash_byte_threshold:
    :return:
    """
    processed_runs, crashed_runs = [], []
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.root'):
            run_num, pool_num = get_run_pool_from_file_name(file_name)
            file_size = os.path.getsize(os.path.join(output_dir, file_name))
            if file_size >= crash_byte_threshold:
                processed_runs.append((run_num, pool_num))
            else:
                crashed_runs.append((run_num, pool_num))

    return processed_runs, crashed_runs


def get_run_pool_from_file_name(file_name):
    """
    Get run and pool number from file name
    :param file_name:
    :return:
    """
    # Regular expression to extract run number and pool number
    pattern = r"Run(\d+)-Pool(\d+)"
    match = re.search(pattern, file_name)

    if match:
        run_number = int(match.group(1))
        pool_number = int(match.group(2))
    else:
        print(f"No match found in the filename {file_name}.")
        run_number, pool_number = None, None
    return run_number, pool_number


if __name__ == '__main__':
    main()