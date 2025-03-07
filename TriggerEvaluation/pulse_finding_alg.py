#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03 4:17 PM 2025
Created in PyCharm
Created as pico_py_analysis/pulse_finding_alg.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from simulation_trigger import *


def main():
    # Data parameters
    n_time_points = 10000
    start_time = 0  # ns
    end_time = 1000  # ns
    t_step = 0.1  # times step in ns
    x_time = np.arange(start_time, end_time, t_step)

    electron_peak_width = 5  # ns
    ion_tail_width = 150  # ns

    # Background parameters
    baseline = 0.0  # mV baseline of noise
    rms_baseline = 0.005  # mV rms of noise

    signal_func = full_fit

    signal_params = {
        'amp': -10051.0,  # mV, amplitude of the signal, not good representation of the signal
        # 'amp': lambda: np.random.uniform(-0.001, -0.1),  # mV, amplitude of the signal
        'midpoint_rising': 213.9648,  # ns, midpoint of the rising edge
        'steepness_rising': 4.71,  # 1/ns, steepness of the rising edge
        'baseline': 0.000859,  # mV, baseline of the signal
        'midpoint_falling': 167,  # ns, midpoint of the falling edge
        'steepness_falling': -0.2208,  # 1/ns, steepness of the falling edge
        'amp_ion': 7.589e-06,  # fraction of amplitude of electron peak
        'steepness_ion': -0.0565,  # 1/ns, steepness of the ion tail
        'x_sigmoid': 245.10,  # ns, x value of the sigmoid
        'k_sigmoid': -0.0756  # 1/ns, steepness of the sigmoid
    }

    fig_sig, ax_sig = plt.subplots(figsize=(8, 6))
    ax_sig.plot(x_time, generate_signal(x_time, signal_func, list(signal_params.values())), color='blue')
    ax_sig.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_sig.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_sig.set_title('Signal', fontsize=18, fontweight='bold', family='serif')
    ax_sig.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_sig.spines.values():
        spine.set_linewidth(2)
    ax_sig.grid(False)
    fig_sig.tight_layout()

    fig_sig, ax_sig = plt.subplots(figsize=(8, 6))
    ax_sig.plot(x_time, generate_signal(x_time - 500, signal_func, list(signal_params.values())), color='blue')
    ax_sig.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_sig.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_sig.set_title('Signal Shift', fontsize=18, fontweight='bold', family='serif')
    ax_sig.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_sig.spines.values():
        spine.set_linewidth(2)
    ax_sig.grid(False)
    fig_sig.tight_layout()

    param_amp = -0.3  # mV Parametrized signal amplitude
    amp_dist = lambda: np.random.uniform(-0.001, -0.1)  # mV, amplitude of the signal

    # threshold parameters
    threshold = -6 * rms_baseline  # 3 sigma for 0.2% tolerance on the accepted background
    window_finding_integration_points = 50
    secondary_finding_integration_points = 20

    # Number of waveforms
    n_waveforms = 5000
    # n_waveforms = 200
    signal_list = np.array([True] * int(n_waveforms / 2) + [False] * int(n_waveforms / 2))

    # Generate one waveform
    y_signal = generate_signal(x_time, signal_func, list(signal_params.values()))
    y_noise = generate_noise(n_time_points, baseline, rms_baseline)

    # Generate a second waveform shifted
    y_signal_shift = generate_signal(x_time - 500, signal_func, list(signal_params.values()))

    # Generate a third small and slightly shifted waveform
    y_secondary = generate_signal(x_time - 100, signal_func, list(signal_params.values())) * 0.1

    total_waveform = y_signal + y_noise + y_signal_shift + y_secondary

    n_pt_threshold = threshold * np.sqrt(window_finding_integration_points)

    x_int, y_int = integral_numpy(x_time, total_waveform, window_finding_integration_points)
    x_bounds = find_pulse_bounds(x_int, y_int, n_pt_threshold, electron_peak_width, ion_tail_width)

    fig_wave, ax_wave = plt.subplots(figsize=(8, 6))
    ax_wave.plot(x_time, total_waveform, color='blue')
    for (x_left, x_right) in x_bounds:
        ax_wave.axvline(x_left, color='green', linestyle='-')
        ax_wave.axvline(x_right, color='green', linestyle='-')
    ax_wave.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_wave.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_wave.set_title('Waveform', fontsize=18, fontweight='bold', family='serif')
    ax_wave.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_wave.spines.values():
        spine.set_linewidth(2)
    ax_wave.grid(False)
    fig_wave.tight_layout()

    fig_int, ax_int = plt.subplots(figsize=(8, 6))
    ax_int.plot(x_int, y_int, color='blue')
    ax_int.axhline(n_pt_threshold, color='red', linestyle='--')
    for (x_left, x_right) in x_bounds:
        ax_int.axvline(x_left, color='green', linestyle='-')
        ax_int.axvline(x_right, color='green', linestyle='-')
    ax_int.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_int.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_int.set_title('Waveform Integral', fontsize=18, fontweight='bold', family='serif')
    ax_int.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_int.spines.values():
        spine.set_linewidth(2)
    ax_int.grid(False)
    fig_int.tight_layout()

    x_int_sec, y_int_sec = integral_numpy(x_time, total_waveform, secondary_finding_integration_points)
    x_deriv_sec, y_deriv_sec = derivative_numpy(x_int_sec, y_int_sec)
    x_deriv_int_sec, y_deriv_int_sec = integral_numpy(x_deriv_sec, y_deriv_sec, secondary_finding_integration_points)
    # Plot 20 point integral
    fig_int_sec, ax_int_sec = plt.subplots(figsize=(8, 6))
    ax_int_sec.plot(x_int_sec, y_int_sec, color='blue')
    ax_int_sec.plot(x_deriv_sec, y_deriv_sec, color='red')
    ax_int_sec.plot(x_deriv_int_sec, y_deriv_int_sec, color='orange')
    # ax_int_sec.axhline(n_pt_threshold, color='red', linestyle='--')
    for (x_left, x_right) in x_bounds:
        ax_int_sec.axvline(x_left, color='green', linestyle='-')
        ax_int_sec.axvline(x_right, color='green', linestyle='-')
    ax_int_sec.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_int_sec.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_int_sec.set_title('Waveform Integral 20 Points', fontsize=18, fontweight='bold', family='serif')
    ax_int_sec.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_int_sec.spines.values():
        spine.set_linewidth(2)
    ax_int_sec.grid(False)
    fig_int_sec.tight_layout()

    y_cdf = get_cdf(x_int, y_int)
    fig_cdf, ax_cdf = plt.subplots(figsize=(8, 6))
    ax_cdf.plot(x_int, y_cdf, color='blue')
    for (x_left, x_right) in x_bounds:
        ax_cdf.axvline(x_left, color='green', linestyle='-')
        ax_cdf.axvline(x_right, color='green', linestyle='-')
    ax_cdf.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax_cdf.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax_cdf.set_title('Waveform Integral CDF', fontsize=18, fontweight='bold', family='serif')
    ax_cdf.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax_cdf.spines.values():
        spine.set_linewidth(2)
    ax_cdf.grid(False)
    fig_cdf.tight_layout()

    plt.show()

    print('donzo')


def find_pulse_bounds(x_int, y_int, threshold, electron_peak_width=5, ion_tail_width=100, end_frac=0.01):
    """
    Find the bounds of a pulse in a waveform integral. Get the min point and then find the points where the integral
    drops above a fraction of this min on either side of the min point.
    Args:
        x_int:
        y_int:
        threshold:
        electron_peak_width:
        ion_tail_width:
        end_frac:

    Returns:

    """
    signal_bounds = []
    waveform_finished = False
    i_start = 0
    while not waveform_finished:
        # Get first point below threshold
        i_triggers = np.where(y_int[i_start:] < threshold)[0]
        if len(i_triggers) == 0:
            waveform_finished = True
            break
        i_trigger = i_triggers[0] + i_start

        # Get min point within ion tail width
        x_trigger = x_int[i_trigger]
        x_end = x_trigger + ion_tail_width
        i_end = np.where(x_int > x_end)[0][0]

        i_min = np.argmin(y_int[i_trigger:i_end]) + i_trigger
        y_min = y_int[i_min]

        y_end = y_min * end_frac

        print(x_trigger, x_end, y_min, y_end)

        # Get first point to left of minimum above end fraction of min
        i_left = np.where(y_int[i_start:i_min] > y_end)[0][-1] + i_start

        # Get first point to right of minimum above end fraction of min
        i_right = np.where(y_int[i_min:] > y_end)[0][0] + i_min

        signal_bounds.append((x_int[i_left], x_int[i_right]))
        print(signal_bounds)
        i_start = i_right

    return signal_bounds


def identify_secondary_pulses(x_int, y_int, x_left, x_right):
    """
    Identify secondary pulses in a waveform integral. Find the min point in the integral between the bounds of the
    Args:
        x_int:
        y_int:
        x_left:
        x_right:

    Returns:

    """
    pass


def get_cdf(x, y):
    """
    Get the cumulative distribution function of a function.
    Args:
        x:
        y:

    Returns:

    """
    cdf = np.cumsum(y)
    cdf = cdf / cdf[-1]
    return cdf


if __name__ == '__main__':
    main()
