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

from TriggerEvaluation.simulation_trigger import ion_tail_model_simple
from simulation_trigger import full_fit, integral, full_fit, fermi_dirac_sym, full_fit2
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

    # params = {
    #     'amp': -0.309,  # mV, amplitude of the signal
    #     'midpoint_rising': 213.647,  # ns, midpoint of the rising edge
    #     'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
    #     'baseline': 0.00089,  # mV, baseline of the signal
    #     'midpoint_falling': 217.427,  # ns, midpoint of the falling edge
    #     'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
    #     'amp_ion': 0.215,  # mV, amplitude of the ion tail
    #     'steepness_ion': -0.0198,  # 1/ns, steepness of the ion tail
    #     'x_sigmoid': 250,  # ns, x value of the sigmoid
    #     'k_sigmoid': -0.058  # 1/ns, steepness of the sigmoid
    # }
    #
    params = {
        'amp': -0.309,  # mV, amplitude of the signal
        'midpoint_rising': 207.647,  # ns, midpoint of the rising edge
        'steepness_rising': 30.86,  # 1/ns, steepness of the rising edge
        'baseline': 0.00089,  # mV, baseline of the signal
        'midpoint_falling': 213.427,  # ns, midpoint of the falling edge
        'steepness_falling': -0.52,  # 1/ns, steepness of the falling edge
        'amp_ion': -0.041,  # mV, amplitude of the ion tail
        'mid_point_ion_sigmoid': 306.4,  # ns, x value of the sigmoid
        'steepness_ion': -0.03187,  # 1/ns, steepness of the ion tail
        'baseline_ion': 0.0013  # baseline of the ion tail
    }

    func = full_fit

    p0 = list(params.values()) # initial guess

    x_sigmoid_range = [213, 600]
    sigmoid_fit_select = (time > x_sigmoid_range[0]) & (time < x_sigmoid_range[1])
    time_sigmoid_fit = time[sigmoid_fit_select]
    data_sigmoid_fit = event_data_df['data'][sigmoid_fit_select]

    sigmoid_p0 = p0[-4:]
    sigmoid_names = list(params.keys())[-4:]

    sigmoid_popt, sigmoid_pcov = cf(ion_tail_model_simple, time_sigmoid_fit, data_sigmoid_fit, sigmoid_p0, maxfev=10000)
    sigmoid_perr = np.sqrt(np.diag(sigmoid_pcov))
    meases = {key: Measure(val, err) for key, val, err in zip(sigmoid_names, sigmoid_popt, sigmoid_perr)}
    for key, val in meases.items():
        print(f'{key}: {val}')

    x_double_sigmoid_range = [202, 214]
    double_sigmoid_fit_select = (time > x_double_sigmoid_range[0]) & (time < x_double_sigmoid_range[1])
    time_double_sigmoid_fit = time[double_sigmoid_fit_select]
    data_double_sigmoid_fit = event_data_df['data'][double_sigmoid_fit_select]

    double_sigmoid_p0 = p0[:6]
    double_sigmoid_names = list(params.keys())[:6]

    double_sigmoid_popt, double_sigmoid_pcov = cf(fermi_dirac_sym, time_double_sigmoid_fit, data_double_sigmoid_fit, double_sigmoid_p0, maxfev=10000)
    double_sigmoid_perr = np.sqrt(np.diag(double_sigmoid_pcov))
    meases = {key: Measure(val, err) for key, val, err in zip(double_sigmoid_names, double_sigmoid_popt, double_sigmoid_perr)}
    for key, val in meases.items():
        print(f'{key}: {val}')

    full_fit_range = [200, 600]
    full_fit_select = (time > full_fit_range[0]) & (time < full_fit_range[1])
    time_full_fit = time[full_fit_select]
    data_full_fit = event_data_df['data'][full_fit_select]
    p0_full2 = [*double_sigmoid_popt, *sigmoid_popt, 214, -1]
    # transition_fit = lambda x, x_sig, k_sig: full_fit2(x, *(p0_full2[:-2]), x_sig, k_sig)
    # popt_full2, pcov_full2 = cf(full_fit2, time_full_fit, data_full_fit, p0_full2, maxfev=100000)
    # perr_full2 = np.sqrt(np.diag(pcov_full2))
    # meases = {key: Measure(val, err) for key, val, err in zip(params.keys(), popt_full2, perr_full2)}
    # for key, val in meases.items():
    #     print(f'{key}: {val}')

    # popt_transition, pcov_transition = cf(transition_fit, time_full_fit, data_full_fit, [214, -1], maxfev=100000)
    # perr_transition = np.sqrt(np.diag(pcov_transition))
    # meases = {key: Measure(val, err) for key, val, err in zip(['x_sig', 'k_sig'], popt_transition, perr_transition)}
    # for key, val in meases.items():
    #     print(f'{key}: {val}')

    # pop_trans_all = [*p0_full2[:-2], *popt_transition]

    # Plot sigmoid initial guess and fit on the original data. Put vertical lines at the x_sigmoid_range
    fig_sigmoid_fit, ax_sigmoid_fit = plt.subplots()
    ax_sigmoid_fit.scatter(time, event_data_df['data'], color='black')
    ax_sigmoid_fit.plot(time_sigmoid_fit, ion_tail_model_simple(time_sigmoid_fit, *sigmoid_p0), color='blue', label='Initial Guess')
    ax_sigmoid_fit.plot(time_sigmoid_fit, ion_tail_model_simple(time_sigmoid_fit, *sigmoid_popt), color='red', ls='--', label='Fitted Curve')
    ax_sigmoid_fit.plot(time_double_sigmoid_fit, fermi_dirac_sym(time_double_sigmoid_fit, *double_sigmoid_p0), color='blue', label='Initial Guess')
    ax_sigmoid_fit.plot(time_double_sigmoid_fit, fermi_dirac_sym(time_double_sigmoid_fit, *double_sigmoid_popt), color='red', ls='--', label='Fitted Curve')
    ax_sigmoid_fit.axvline(x=x_sigmoid_range[0], color='green', ls='--')
    ax_sigmoid_fit.axvline(x=x_sigmoid_range[1], color='green', ls='--')
    ax_sigmoid_fit.legend()

    fig_full_fit, ax_full_fit = plt.subplots()
    ax_full_fit.scatter(time, event_data_df['data'], color='black')
    # Plot different ks
    ax_full_fit.plot(time_full_fit, full_fit2(time_full_fit, *p0_full2), color='blue', label='Initial Guess')
    ks = np.linspace(-10, -0.001, 100)
    x_switches = np.linspace(212, 216, 10)
    for x_switch in x_switches:
        for k in ks:
            p0_k = [*p0_full2[:-2], x_switch, k]
            ax_full_fit.plot(time_full_fit, full_fit2(time_full_fit, *p0_k), alpha=0.5, color='orange')
    # ax_full_fit.plot(time_full_fit, full_fit2(time_full_fit, *pop_trans_all), color='red', ls='--', label='Fitted Curve')
    ax_full_fit.axvline(x=full_fit_range[0], color='green', ls='--')
    ax_full_fit.axvline(x=full_fit_range[1], color='green', ls='--')
    ax_full_fit.legend()

    plt.show()

    # for each p0, check if it is within the bounds
    # for i, (val, lower, upper) in enumerate(zip(p0, lower_bounds, upper_bounds)):
    #     if val < lower or val > upper:
    #         print(f'Parameter {list(params.keys())[i]} is out of bounds: {val} < {lower} or {val} > {upper}')
    #         # p0[i] = (lower + upper) / 2
    # Define lower and upper bounds (use -np.inf and np.inf for unrestricted parameters)
    # lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.044, 330, -0.0022*0.1]
    # upper_bounds = [ np.inf,  np.inf,  np.inf, np.inf, np.inf, np.inf, np.inf, 0.044, 340, 0.0022]




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
