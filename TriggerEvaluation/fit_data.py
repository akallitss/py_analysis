#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on February 11 16:41 2025
Created in PyCharm
Created as py_analysis/fit_data.py

@author: Alexandra Kallitsopoulou, akallits
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import pandas as pd

from simulation_trigger import full_fit, integral, full_fit
from Measure import Measure


def main():
    path = '/home/akallits/Documents/PicoAnalysis/Saclay_Analysis/data/2022_October_h4/plots/Run224/Pool2/moreplots/waveform_data.txt'
    event_data =parse_event_data(open(path).read())
    event_data_df = pd.DataFrame(event_data)
    print(event_data_df)

    # Get median dt
    median_dt = np.median(event_data_df['dt'])
    time = np.arange(0, len(event_data_df['data']) * median_dt, median_dt)

    fig, ax = plt.subplots()
    ax.scatter(time, event_data_df['data'], color='black')

    # params = {
    #     'amp': -0.0609 * 10,  # mV, amplitude of the signal
    #     'midpoint_rising': 227.647,  # ns, midpoint of the rising edge
    #     'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
    #     'baseline': 0.00044,  # mV, baseline of the signal
    #     'midpoint_falling': 230.427,  # ns, midpoint of the falling edge
    #     'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
    #     'amp_ion': -0.01308 * 10,  # mV, amplitude of the ion tail
    #     'steepness_ion': 0.0198,  # 1/ns, steepness of the ion tail
    #     'x_sigmoid': 259,  # ns, x value of the sigmoid
    #     'k_sigmoid': 0.058  # 1/ns, steepness of the sigmoid
    # }
    # params = {
    #     'amp': -0.0609,  # mV, amplitude of the signal
    #     'midpoint_rising': 227.647,  # ns, midpoint of the rising edge
    #     'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
    #     'baseline': 0.00089,  # mV, baseline of the signal
    #     'midpoint_falling': 230.427,  # ns, midpoint of the falling edge
    #     'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
    #     'amp_ion': 0.215,  # mV, amplitude of the ion tail
    #     # 'steepness_ion': 0.0198,  # 1/ns, steepness of the ion tail
    #     'x_sigmoid': 259,  # ns, x value of the sigmoid
    #     'k_sigmoid': 0.058  # 1/ns, steepness of the sigmoid
    # }

    params = {
        'amp': -0.309,  # mV, amplitude of the signal
        'midpoint_rising': 213.647,  # ns, midpoint of the rising edge
        'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
        'baseline': 0.00089,  # mV, baseline of the signal
        'midpoint_falling': 217.427,  # ns, midpoint of the falling edge
        'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
        'amp_ion': 0.215,  # mV, amplitude of the ion tail
        'steepness_ion': -0.0198,  # 1/ns, steepness of the ion tail
        'x_sigmoid': 250,  # ns, x value of the sigmoid
        'k_sigmoid': -0.058  # 1/ns, steepness of the sigmoid
    }

    func = full_fit

    p0 = list(params.values())
    x_plot = np.linspace(min(time), max(time), 10000)
    ax.plot(x_plot, func(x_plot, *p0), color='blue', label='Initial Guess')

    popt, pcov = cf(func, time, event_data_df['data'], p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    meases = {key: Measure(val, err) for key, val, err in zip(params.keys(), popt, perr)}
    for key, val in meases.items():
        print(f'{key}: {val}, {val.val}')

    ax.plot(x_plot, func(x_plot, *popt), color='red', ls='--', label='Fitted Curve')

    fig, ax = plt.subplots()
    ax.plot(time, event_data_df['data'], color='black')

    mv_avgs = np.array([1, 20, 50, 100, 150]) * 10
    for mv_avg in mv_avgs:
        x_avg, y_avg = integral(time, event_data_df['data'], mv_avg)
        ax.plot(x_avg, y_avg, label=f'n point: {mv_avg}')

    plt.show()


    print('bonzo')



def parse_event_data(text):
    event_data = {}

    # Extract integer values
    points_match = re.search(r'const int points = (\d+);', text)
    if points_match:
        event_data["points"] = int(points_match.group(1))

    # Extract float values
    dt_match = re.search(r'dt: ([\d\.eE+-]+)', text)
    rms_match = re.search(r'RMS: ([\d\.eE+-]+)', text)
    bsl_match = re.search(r'BSL: ([\d\.eE+-]+)', text)

    if dt_match:
        event_data["dt"] = float(dt_match.group(1))
    if rms_match:
        event_data["RMS"] = float(rms_match.group(1))
    if bsl_match:
        event_data["BSL"] = float(bsl_match.group(1))

    # Extract arrays
    data_match = re.search(r'double data\[\d+\] = \{(.*?)\};', text)
    drv_match = re.search(r'double drv\[\d+\] = \{(.*?)\};', text)

    if data_match:
        event_data["data"] = [float(x) for x in data_match.group(1).split(",")]
    if drv_match:
        event_data["drv"] = [float(x) for x in drv_match.group(1).split(",")]

    return event_data


if __name__ == '__main__':
    main()
