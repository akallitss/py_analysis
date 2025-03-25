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
from subprocess import Popen, PIPE
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

def main():
    # base_dir = '/home/akallits/Documents/PicoAnalysis/Saclay_Analysis/'  # Laptop ubuntu
    base_dir = '/eos/project-p/picosec/analysis/Saclay/'  # EOS Server
    rerun = True  # If True, rerun runs even if found to be processed. Else only run unprocessed runs
    # test_script(base_dir)
    process_runs(base_dir, rerun)
    print('bonzo')


def process_runs(base_dir, rerun=False):
    """
    Process all runs in the logbook
    :param base_dir:
    :param rerun:
    :return:
    """

    #test_beam_period_dir = '2024_June_h4/'
    #root_macro_name = 'MakeTreefromRawTreePicosecJune24.C+'
    test_beam_period_dir = '2023_April_h4/'
    test_beam_period_code_dir = 'Apr23/'  # 'Aug23'
    root_macro_name = 'MakeTreefromRawTreePicosecApril23.C++'
   
    logbook_dir = f'{base_dir}data/{test_beam_period_dir}/'
    output_dir = f'{base_dir}data/{test_beam_period_dir}/processedTrees/'
    #logbook_name = 'OsciloscopeSetup_LogbookAll.txt'
    logbook_name = 'OsciloscopeSetup_LogbookAll_extra.txt'
    logbook_path = os.path.join(logbook_dir, logbook_name)
    log_path = f'{logbook_dir}MakeTreePicosecProcess.log'
    # logbook = pd.read_csv(logbook_path, sep='\t', header=0)
    # print_log(log_path, logbook)
    expected_line_length = 27
    crash_byte_threshold = 1000  # bytes Output root files smaller than this are considered crashed

    # Read run info from logbook
    log_run_info_dict  = read_logbook(logbook_path, expected_line_length)

    # Read runs already processed
    processed_runs, crashed_runs = check_processed_runs(output_dir, crash_byte_threshold)

    # Process runs which are not already processed
    for run_info in log_run_info_dict:
        if (int(run_info['RunNo']), int(run_info['PoolNo'])) not in processed_runs:
            print_log(f'Run {run_info["RunNo"]}, Pool {run_info["PoolNo"]} not processed yet.', log_path)
            process_run(run_info, base_dir, test_beam_period_dir, test_beam_period_code_dir, root_macro_name)  # Process run
        else:
            print_log(f'Run {run_info["RunNo"]}, Pool {run_info["PoolNo"]} already processed.', log_path)
            if rerun:
                print_log(f'Run {run_info["RunNo"]}, Pool {run_info["PoolNo"]} reprocessing.', log_path)
                process_run(run_info, base_dir, test_beam_period_dir, test_beam_period_code_dir, root_macro_name)  # Process run anyway

    print_log(f'Processed runs: {processed_runs}', log_path)
    print_log(f'Crashed runs: {crashed_runs}', log_path)
    print_log(f'Finished processing runs.', log_path)


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
                #save the run and pool number to a log file
                with open('crashed_runs.log', 'a') as log_file:
                    log_file.write(f'{run_num} {pool_num}\n')

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


def test_script(base_dir):
    """
    Test the process run script
    :return:
    """
    process_run({'RunNo': 163, 'PoolNo': 3}, base_dir)


def process_run(run_info, base_dir, test_beam_period_dir, test_beam_period_code_dir, root_macro_name, log_path=None):
    """
    Process a run
    :param run_info:
    :param base_dir:
    :param test_beam_period_dir:
    :param test_beam_period_code_dir:
    :param root_macro_name:
    :param log_path:
    :return:
    """
    # Remake OscilloscopeSetup.txt with current run info
    osc_path = f'{base_dir}data/{test_beam_period_dir}/OsciloscopeSetup.txt'
    with open(osc_path, 'w') as osc_file:
        for key in run_info:
            osc_file.write(f'{key}\t')
        osc_file.write('\n')
        for key in run_info:
            osc_file.write(f'{run_info[key]}\t')

    script_name = f'code/{test_beam_period_code_dir}{root_macro_name}'
    print_log(f'Processing run {run_info["RunNo"]}, pool {run_info["PoolNo"]}', log_path)
    # Get run and pool number
    run_number = run_info['RunNo']
    pool_number = run_info['PoolNo']

    # Get the file path
    command = f'root -l -q "{base_dir}{script_name}({run_number}, {pool_number})"'
    print_log(f'Running command: {command}', log_path)
    # Run the script
    # Run the command, outputting directly to the screen
    process = Popen(
        command,
        shell=True,
        stdout=None,  # Allow stdout to go to the terminal
        stderr=None,  # Allow stderr to go to the terminal
    )
    process.wait()  # Wait for the process to finish

    # Check for errors
    if process.returncode != 0:
        print_log(f"Error: ROOT script failed with return code {process.returncode}", log_path)
        return False
    return True


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
