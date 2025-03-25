#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on February 11 11:07 2025
Created in PyCharm
Created as py_analysis/simulation_trigger.py

@author: Alexandra Kallitsopoulou, akallits
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    # Data parameters
    n_time_points = 10000
    start_time = 0  # ns
    end_time = 1000  # ns
    t_step = 0.1 # times step in ns
    # x_time = np.linspace(start_time, end_time, n_time_points)
    x_time = np.arange(start_time, end_time, t_step)
    # print(len(x_time))

    #
    # Background parameters
    baseline = 0.0  # mV baseline of noise
    rms_baseline = 0.005  # mV rms of noise

    # Signal parameters
    # signal_func = gaus_func
    # signal_params = {'amp': -1, 'mu': 200, 'sigma': 6}
    # signal_func = fermi_dirac_sym
    # signal_params = {
    #     'amp': -0.0609 * 10,  # mV, amplitude of the signal
    #     'midpoint_rising': 227.647,  # ns, midpoint of the rising edge
    #     'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
    #     'baseline': 0.00044,  # mV, baseline of the signal
    #     'midpoint_falling': 230.427,  # ns, midpoint of the falling edge
    #     'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
    # }
    signal_func = full_fit
    # signal_params = {
    #     # 'amp': -0.0609 * 1,  # mV, amplitude of the signal
    #     'amp': 0.0,  # mV, amplitude of the signal
    #     # 'amp': lambda: np.random.uniform(-0.001, -0.1),  # mV, amplitude of the signal
    #     'midpoint_rising': 227.647,  # ns, midpoint of the rising edge
    #     'steepness_rising': 3.86,  # 1/ns, steepness of the rising edge
    #     'baseline': 0.00089,  # mV, baseline of the signal
    #     'midpoint_falling': 230.427,  # ns, midpoint of the falling edge
    #     'steepness_falling': -1.52,  # 1/ns, steepness of the falling edge
    #     'amp_ion': 0.215,  # fraction of amplitude of electron peak
    #     'steepness_ion': 0.0198,  # 1/ns, steepness of the ion tail
    #     'x_sigmoid': 259,  # ns, x value of the sigmoid
    #     'k_sigmoid': 0.058  # 1/ns, steepness of the sigmoid
    # }
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

    param_amp = -0.3 # mV Parametrized signal amplitude
    amp_dist = lambda: np.random.uniform(-0.001, -0.1) # mV, amplitude of the signal
        # threshold parameters
    # threshold = -rms_baseline # 1 sigma for 16% tolerance on the accepted background
    # threshold = -3*rms_baseline # 3 sigma for 0.2% tolerance on the accepted background
    threshold = -6*rms_baseline # 3 sigma for 0.2% tolerance on the accepted background
    # threshold = -2*rms_baseline # 2 sigma for 2.5% tolerance on the accepted background
    # Define integration points
    # integration_points = [1400, 1000, 50, 24, 10, 5, 3, 1]
    integration_points = [500, 200, 100, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 5, 3, 1]
    # integration_points = np.arange(20, 1500, 2)

    # Number of waveforms
    n_waveforms = 5000
    # n_waveforms = 200
    signal_list = np.array([True] * int(n_waveforms / 2) + [False] * int(n_waveforms / 2))

    # Run Simulation
    triggers_fired = {int_points: [] for int_points in integration_points}
    amps, amps_triggerd = [], []
    for i in range(n_waveforms):
        print(f'Waveform {i}')
        if signal_list[i]:
            sig_params = {key: signal_params[key]() if callable(signal_params[key]) else signal_params[key]
                            for key in signal_params}
            scale_amp = amp_dist() / param_amp
            # amps.append(sig_params['amp'])
            amps.append(scale_amp * param_amp)
            y_signal = generate_signal(x_time, signal_func, list(sig_params.values())) * scale_amp
        else:
            y_signal = np.zeros(n_time_points)
        y_noise = generate_noise(n_time_points, baseline, rms_baseline)
        total_waveform = combine_signal_noise(x_time, y_signal, y_noise)

        any_trigger_fire = False
        for int_points in integration_points:
            # x_int, y_int = integral(x_time, total_waveform, int_points)
            x_int, y_int = integral_numpy(x_time, total_waveform, int_points)
            # did_trigger_fire = trigger_fired(y_int, threshold * int_points)
            did_trigger_fire = trigger_fired(y_int, threshold * np.sqrt(int_points))
            triggers_fired[int_points].append(did_trigger_fire)
            any_trigger_fire = any_trigger_fire or did_trigger_fire
        if signal_list[i] and any_trigger_fire:
            amps_triggerd.append(scale_amp *
                                 param_amp)

        # plot_waveform_integrals(x_time, total_waveform, integration_points, threshold)
        # plt.show()

    # Plot amp distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    hist, bin_edges, _ = ax.hist(amps, bins=20, color='gray', edgecolor='black', alpha=0.3)
    ax.hist(amps_triggerd, bins=bin_edges, color='blue', edgecolor='black', alpha=0.3)
    ax.set_xlabel('Amplitude (V)', fontsize=16, fontweight='bold', family='serif')
    ax.set_ylabel('Counts', fontsize=16, fontweight='bold', family='serif')
    ax.set_title('Amplitude Distribution', fontsize=18, fontweight='bold', family='serif')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(False)
    fig.tight_layout()
    #save the plot in pdf
    plt.savefig('amplitude_distribution.pdf')


    # Analyze simulation results
    false_positive_fraction = []
    false_negative_fraction = []
    true_positive_fraction = []
    true_negative_fraction = []
    # Initialize lists to store TPR and FPR
    tpr_values = []
    fpr_values = []

    for int_points in integration_points:
        signal_data = np.array(triggers_fired[int_points])[signal_list]
        bkg_data = np.array(triggers_fired[int_points])[~signal_list]

        for data, data_type in zip([signal_data, bkg_data], ['signal', 'background']):
            n_trig_fired = np.sum(data)
            frac_trig_fired = n_trig_fired / len(data)
            print(f'Percentage of triggers fired for {int_points} integration points {data_type}: '
                    f'{frac_trig_fired * 100}%')
            if data_type == 'signal':
                false_negative_fraction.append(1 - frac_trig_fired)
                true_positive_fraction.append(frac_trig_fired)
            else:
                false_positive_fraction.append(frac_trig_fired)
                true_negative_fraction.append(1 - frac_trig_fired)

        #true positives when trigger fires for signal
        true_positives = np.sum(signal_data)
        false_negatives = len(signal_data) - true_positives

        #false positives when trigger fires for background
        false_positives = np.sum(bkg_data)
        true_negatives = len(bkg_data) - false_positives

        #compute TPR and FPR
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        # Append values to lists
        tpr_values.append(tpr)
        fpr_values.append(fpr)


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(integration_points, false_positive_fraction, label='False Positive Fraction', marker='o')
    # ax.plot(integration_points, false_negative_fraction, label='False Negative Fraction', marker='o')
    ax.plot(integration_points, true_positive_fraction, label='True Positive Fraction', marker='o')
    # ax.plot(integration_points, true_negative_fraction, label='True Negative Fraction', marker='o')
    ax.set_xlabel('Integration Points', fontsize=16, fontweight='bold', family='serif')
    ax.set_ylabel('Fraction', fontsize=16, fontweight='bold', family='serif')
    ax.set_title('Trigger Evaluation', fontsize=18, fontweight='bold', family='serif')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    plt.savefig('trigger_evaluation.pdf')

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_values, tpr_values, label='ROC Curve', marker='o')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=16, fontweight='bold', family='serif')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=16, fontweight='bold', family='serif')
    ax.set_title('ROC Curve', fontsize=18, fontweight='bold', family='serif')
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(False)

    # Annotate each point with its corresponding integration point value
    for i, (fpr, tpr) in enumerate(zip(fpr_values, tpr_values)):
        ax.text(fpr, tpr, str(integration_points[i]), fontsize=12, ha='center', va='bottom')

    # Add legend
    ax.legend(frameon=False, fontsize=12)

    # Adjust layout and show plot
    fig.tight_layout()
    plt.savefig('ROC_curve.pdf')


    plt.show()

    print('bonzo')

def generate_noise(n, baseline, rms_baseline):
    """
    Generate noise
    :param n: number of points
    :param baseline: baseline
    :param rms_baseline: rms of the baseline
    :return: y_noise
    """
    y_noise = np.random.normal(baseline, rms_baseline, n)
    return y_noise


def generate_signal(x, signal_func, signal_params, x_shift=0):
    """
    Generate a signal
    :param x: time points
    :param signal_func: signal function
    :param signal_params: signal parameters
    :param x_shift: x shift
    :return: y_signal
    """
    y_signal = signal_func(x + x_shift, *signal_params)
    return y_signal


def plot_waveform_integrals(x, y, integration_points, threshold):
    """
    Plot the waveform and its integrals for different integration points
    :param x:
    :param y:
    :param integration_points:
    :param threshold:
    :return:
    """
    for int_points in integration_points:
        x_int, y_int = integral(x, y, int_points)
        # plot_integral(x, y, x_int, y_int, int_points, threshold * int_points)
        plot_integral(x, y, x_int, y_int, int_points, threshold * np.sqrt(int_points))
        # print_threshold_fraction(y_int, threshold * int_points)
        print_threshold_fraction(y_int, threshold* np.sqrt(int_points))


def combine_signal_noise(x, y_signal, y_noise):
    """
    Combine signal and noise
    :param x:
    :param y_signal:
    :param y_noise:
    :return:
    """
    return y_signal + y_noise


def moving_average(x, y, n):
    """
    Calculate the moving average of a waveform in both x and y without numpy
    :param x: time points
    :param y: voltage points
    :param n: number of points to average
    :return: x, y of the moving average waveform
    """
    x_avg = []
    y_avg = []
    for i in range(len(x) - n + 1):
        x_avg.append(np.mean(x[i:i + n]))
        y_avg.append(np.mean(y[i:i + n]))
    return np.array(x_avg), np.array(y_avg)


def moving_average_numpy(x, y, n):
    """
    Calculate the moving average of a waveform in both x and y with numpy
    Args:
        x:
        y:
        n:

    Returns:

    """
    x_avg = np.convolve(x, np.ones(n), 'valid') / n
    y_avg = np.convolve(y, np.ones(n), 'valid') / n
    return x_avg, y_avg


def integral(x, y, n):
    """
    Calculate the integral of a waveform in both x and y without numpy
    :param x: time points
    :param y: voltage points
    :param n: number of points to integrate
    :return: x, y of the integral waveform
    """
    x_int = []
    y_int = []
    for i in range(len(x) - n + 1):
        x_int.append(np.mean(x[i:i + n]))
        y_int.append(np.sum(y[i:i + n]))
    return np.array(x_int), np.array(y_int)


def integral_numpy(x, y, n):
    """
    Calculate the integral of a waveform in both x and y with numpy
    :param x: time points
    :param y: voltage points
    :param n: number of points to integrate
    :return: x, y of the integral waveform
    """
    x_int = np.convolve(x, np.ones(n), 'valid') / n
    y_int = np.convolve(y, np.ones(n), 'valid')
    return x_int, y_int


def derivative_numpy(x, y):
    """
    Compute the numerical derivative of a waveform.

    Args:
        x (array): X values of the waveform.
        y (array): Y values of the waveform.

    Returns:
        x_avg (array): Midpoints of x values for better alignment.
        dy_dx (array): Numerical derivative dy/dx.
    """
    dy_dx = np.gradient(y, x)  # Compute derivative

    # Compute midpoints of x for better alignment
    x_avg = (x[:-1] + x[1:]) / 2

    return x_avg, dy_dx[:-1]  # Trim dy_dx to match the length of x_avg


def derivate_array(x, y, n):
    # Calculate the derivative of the waveform in both x and y for n points
    x_avg = []
    dy_dx = []
    for i in range(len(x) - n + 1):
        x_avg.append(np.mean(x[i:i + n]))
        dy_dx.append(np.mean(np.diff(y[i:i + n]) / np.diff(x[i:i + n])))
    return np.array(x_avg), np.array(dy_dx)


def derivative_thomas(x, y, n=1):
    """
    Calculate the derivative between nth neighbor points.
    :param x:
    :param y:
    :param n:
    :return:
    """
    dy_dx = (np.array(y)[n:] - np.array(y)[:-n]) / (np.array(x)[n:] - np.array(x)[:-n])
    x_avg = (np.array(x)[n:] + np.array(x)[:-n]) / 2
    return x_avg, dy_dx

def gaus_func(x, a, mu, sigma):
    """
    Gaussian function
    :param x: x points
    :param a: amplitude
    :param mu: mean
    :param sigma: standard deviation
    :return: y points
    """
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fermi_dirac_sym(x, a, x1, k1, c, x2, k2):
    """
    Computes the symmetric double Fermi-Dirac function.

    Parameters:
    x  : float or numpy array, the input value(s)
    A  : float, amplitude scaling factor
    x1 : float, center position of first transition
    k1 : float, steepness of first transition
    C  : float, offset constant
    x2 : float, center position of second transition
    k2 : float, steepness of second transition

    Returns:
    float or numpy array: Computed Fermi-Dirac function value(s)
    """
    term1 = 1.0 / (1.0 + np.exp(-k1 * (x - x1)))
    term2 = 1.0 / (1.0 + np.exp(-k2 * (x - x2)))
    return (a * term1 * term2) + c


def ion_tail_model(x, a, k, b, x_sigmoid, k_sigmoid):
    """
    Exponential model of ion tail
    :param x:
    :param a:
    :param k:
    :param b:
    :param x_start:
    :return:
    """
    x_shift = x - x_sigmoid
    sigmoid = 1 / (1 + np.exp(-k_sigmoid * x_shift))
    return a * np.exp(-k * x_shift) * sigmoid + b

def ion_tail_model_simple(x, p0,p1,p2,p3 ):
    """ Sigmoid model of ion tail"""
    ":param x: time points"
    ":param p0: amplitude"
    ":param p1: steepness"
    ":param p2: offset"
    ":param p3: baseline"

    return p0 / (1 + np.exp(-p2 * (x - p1))) + p3


# def full_fit(x, a, x1, k1, c, x2, k2, a_ion, k_ion, x_sigmoid, k_sigmoid):
#     """
#     Full fit of signal and ion tail
#     :param x:
#     :param a:
#     :param x1:
#     :param k1:
#     :param c:
#     :param x2:
#     :param k2:
#     :param a_ion:
#     :param k_ion:
#     :param x_sigmoid:
#     :param k_sigmoid:
#     :return:
#     """
#     # return fermi_dirac_sym(x, a, x1, k1, c, x2, k2) + ion_tail_model(x, a_ion, k_ion, c, x_sigmoid, k_sigmoid)
#     # return a * (fermi_dirac_sym(x, 1, x1, k1, 0, x2, k2) + ion_tail_model(x, a_ion, k_ion, 0, x_sigmoid, k_sigmoid)) + c

def full_fit(x, a, x1, k1, c, x2, k2, a_ion, k_ion, mid_point, baseline):
    """
    Full fit of signal and ion tail
    :param x:
    :param a:
    :param x1:
    :param k1:
    :param c:
    :param x2:
    :param k2:
    :param a_ion:
    :param k_ion:
    :param x_sigmoid:
    :param k_sigmoid:
    :return:
    """
    # return fermi_dirac_sym(x, a, x1, k1, c, x2, k2) + ion_tail_model(x, a_ion, k_ion, c, x_sigmoid, k_sigmoid)
    # return a * (fermi_dirac_sym(x, 1, x1, k1, 0, x2, k2) + ion_tail_model(x, a_ion, k_ion, 0, x_sigmoid, k_sigmoid)) + c
    return a * (fermi_dirac_sym(x, 1, x1, k1, 0, x2, k2) + ion_tail_model_simple(x, a_ion, k_ion, mid_point, baseline)) + c


def full_fit2(x, a, x1, k1, c, x2, k2, a_ion, k_ion, mid_point, baseline, x_sigmoid, k_sigmoid):
    """
    Full fit of signal and ion tail
    :param x:
    :param a:
    :param x1:
    :param k1:
    :param c:
    :param x2:
    :param k2:
    :param a_ion:
    :param k_ion:
    :param x_sigmoid:
    :param k_sigmoid:
    :return:
    """
    # return fermi_dirac_sym(x, a, x1, k1, c, x2, k2) + ion_tail_model(x, a_ion, k_ion, c, x_sigmoid, k_sigmoid)
    # return a * (fermi_dirac_sym(x, 1, x1, k1, 0, x2, k2) + ion_tail_model(x, a_ion, k_ion, 0, x_sigmoid, k_sigmoid)) + c
    return sigmoid(x, x_sigmoid, k_sigmoid) * fermi_dirac_sym(x, a, x1, k1, c, x2, k2) + (1 - sigmoid(x, x_sigmoid, k_sigmoid)) * ion_tail_model_simple(x, a_ion, k_ion, mid_point, baseline)


def sigmoid(x, x0, k):
    """
    Sigmoid function
    :param x:
    :param a:
    :param x0:
    :param k:
    :return:
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

def plot_integral(x, y, x_int, y_int, n, threshold):
    """
    Plot the integral waveform
    :param x:
    :param y:
    :param x_int:
    :param y_int:
    :param n:
    :param threshold:
    :return:
    """
    # # Create figure and axis
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # Scatter plots with ROOT-style markers
    # ax.scatter(x, y, color='black', s=10, label="Original Data")  # Small black points
    # ax.scatter(x_int, y_int, color='blue', s=10, label="Integrated Points")  # Small blue points
    #
    # # Threshold Line
    # ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label="Threshold")
    #
    # # Set labels and title with ROOT-like font
    # ax.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    # ax.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    # ax.set_title(f'Integral waveform with {n} points', fontsize=18, fontweight='bold', family='serif')
    #
    # # ROOT-like axis styling
    # ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    #
    # # Remove grid for clean ROOT look
    # ax.grid(False)
    #
    # # Add legend (optional, for clarity)
    # ax.legend(frameon=False, fontsize=12)
    #
    # # Adjust layout
    # fig.tight_layout()
    # Create figure with GridSpec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0)

    # Scatter plot
    ax = fig.add_subplot(gs[0])
    ax.scatter(x, y, color='black', s=10, label="Original Data")
    ax.scatter(x_int, y_int, color='blue', s=10, label="Integrated Points")

    # Threshold Line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label="Threshold")

    # Set labels and title with ROOT-like font
    ax.set_xlabel('Time (ns)', fontsize=16, fontweight='bold', family='serif')
    ax.set_ylabel('Voltage (V)', fontsize=16, fontweight='bold', family='serif')
    ax.set_title(f'Integral waveform with {n} points', fontsize=18, fontweight='bold', family='serif')

    # ROOT-like axis styling
    ax.tick_params(axis='both', which='both', direction='in', length=8, width=2, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Remove grid for clean ROOT look
    ax.grid(False)

    # Add legend
    ax.legend(frameon=False, fontsize=12)

    # Histogram on the right
    ax_hist = fig.add_subplot(gs[1], sharey=ax)
    ax_hist.hist(y, bins=20, orientation='horizontal', color='gray', edgecolor='black', alpha=0.3)
    ax_hist.axhline(threshold, color='red', linestyle='--', linewidth=2, label="Threshold")

    # Histogram of integral points
    ax_hist.hist(y_int, bins=20, orientation='horizontal', color='blue', edgecolor='black', alpha=0.3)

    ax_hist.yaxis.set_visible(False)  # This ensures the main plot retains its y-axis labels

    # Remove labels and ticks for clarity
    # ax_hist.set_yticklabels([])
    ax_hist.set_xticks([])
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['bottom'].set_visible(False)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)


def trigger_fired(y, threshold):
    """
    Check if the trigger fired
    :param y:
    :param threshold:
    :return:
    """
    return np.any(y < threshold)


def print_threshold_fraction(y, threshold):
    """
    Print the fraction of points below the threshold
    :param y:
    :param threshold:
    :return:
    """
    y_below_threshold = y[y < threshold]
    print(f'Threshold: {threshold}')
    print(f'Number of points below threshold points: {len(y_below_threshold)}')
    print(f'Fraction of points below threshold: {len(y_below_threshold)/len(y)}')
    print(f'Percentage of points below threshold: {len(y_below_threshold)/len(y)*100}%')
    return len(y_below_threshold)/len(y)


if __name__ == '__main__':
    main()
