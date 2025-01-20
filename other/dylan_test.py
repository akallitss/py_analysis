
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import uproot


def main():
    file_path = './data/2024_June_h4/Run303-Pool3_TESTBEAMraw_tree.root'
    tree_vars = [['npoints', 'eventNo', 't0', 'gain', 'offset', 'rmax', 'rmin', 'channelCoupling', 'dt', 'epoch', 'nn',
                  'date', 'npoints1', 'amplC1', 'npoints2', 'amplC2', 'npoints3', 'amplC3', 'npoints4', 'amplC4']]
    with uproot.open(file_path) as file:
        tree = file[file.keys()[0]]
        ampC1 = tree['amplC2']
        # Convert to numpy array
        ampC1_np = ampC1.array(entry_stop=500).to_numpy()
        print(type(ampC1_np))
        print(ampC1_np.shape)
        # for i in range(5):
        #     plot_waveform(ampC1_np[i])
        #     plt.show()
        #hist_min_times(ampC1_np)
        min_ampl_hist(ampC1_np)
        plt.show()
        amp_frac = 0.2
        # Calculate time at which each waveform reaches a fraction of the minimum amplitude
        min_times = np.apply_along_axis(calc_sigmoid_frac_time, 1, ampC1_np, amp_frac)
        # Count number of nan values in min_times and print number and fraction
        nan_count = np.sum(np.isnan(min_times))
        #plot the waveforms of events for the nan values
        # for i in range(len(min_times)):
        #     if np.isnan(min_times[i]):
        #         plot_waveform(ampC1_np[i], fit=False)
        #         plt.show()
        print(f'Number of nan values: {nan_count}')
        print(f'Fraction of nan values: {nan_count / len(min_times)}')
        plt.hist(min_times, bins=100)
        # min_ampl1_vals = calc_min_ampl_vals(ampC1_np)
        # ampC1_low_mins = ampC1_np[min_ampl1_vals < -0.05]
        # for i in range(len(ampC1_low_mins)):
        #     plot_waveform(ampC1_low_mins[i], fit=True)
        #     plt.show()
        # plot_var_hist(tree, 'gain')
    plt.show()
    print('bonzo')


def calc_min_ampl_vals(waveforms):
    """
    Calculate minimum amplitudes of waveforms
    :param waveforms:
    :return:
    """
    min_ampl_vals = np.min(waveforms, axis=1)
    return min_ampl_vals


def plot_var_hist(tree, var_name):
    """
    Plot histogram of a variable in a tree
    :param tree: Uproot tree
    :param var_name: Name of the variable to plot
    :return:
    """
    var = tree.array(var_name)
    plt.hist(var, bins=100)


def plot_waveform(waveform, fit=False):
    """
    Plot input waveform
    :param waveform: Input waveform
    :param fit: Fit waveform with sigmoid function
    :return:
    """
    min_point = np.min(waveform)
    min_time = np.argmin(waveform)
    window_size_before_min, window_size_after_min = 150, 1500
    time_window = np.arange(min_time - window_size_before_min, min_time + window_size_after_min)
    plt.plot(time_window, waveform[time_window])

    if fit:
        points_back = int(5 * 1000 / 50)  # 5 ns for 50 ps per point
        amp_frac = 0.2
        fit_window = np.arange(min_time - points_back, min_time)
        popt, pcov = cf(sigmoid, fit_window, waveform[fit_window], p0=[1, 0, 0.1, min_time])
        perr = np.sqrt(np.diag(pcov))
        print(popt)
        print(perr)
        plt.plot(fit_window, sigmoid(fit_window, *popt), 'r-')
        time_point = sigmoid_fraction_x(*popt, amp_frac)
        plt.axvline(time_point, color='k', linestyle='--')
        plt.axhline(min_point * amp_frac, color='k', linestyle='--')
        return time_point


def calc_sigmoid_frac_time(waveform, amp_frac=0.2):
    """
    Calculate time at which waveform reaches a fraction of the minimum amplitude
    :param waveform: Input waveform
    :param amp_frac: Fraction of minimum amplitude
    :return:
    """
    try:
        min_time = np.argmin(waveform)
        points_back = int(5 * 1000 / 50)  # 5 ns for 50 ps per point
        fit_window = np.arange(min_time - points_back, min_time)
        popt, pcov = cf(sigmoid, fit_window, waveform[fit_window], p0=[1, 0, 0.1, min_time])
        perr = np.sqrt(np.diag(pcov))
        time_point = sigmoid_fraction_x(*popt, amp_frac)
        return time_point
    except RuntimeError:
        return np.nan



def hist_min_times(waveforms):
    """
    Plot histogram of min times of waveforms
    :param waveforms:
    :return:
    """
    min_times = np.argmin(waveforms, axis=1)
    plt.hist(min_times, bins=100)


def min_ampl_hist(waveforms):
    """
    Plot histogram of min amplitudes of waveforms
    :param waveforms:
    :return:
    """
    min_amps = np.min(waveforms, axis=1)
    plt.hist(min_amps, bins=100)


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b


def sigmoid_max(a, b):
    # As x -> infinity
    sigmoid_inf = a + b

    # As x -> -infinity
    sigmoid_neg_inf = b

    # Find maximum or minimum value
    if a > 0:
        return sigmoid_inf  # Maximum at positive infinity
    else:
        return sigmoid_neg_inf  # Minimum at negative infinity


def sigmoid_fraction_x(a, b, c, d, f):
    # Max and min values of the sigmoid
    if a > 0:
        S_max = a + b  # Max value as x -> infinity
        S_min = b  # Min value as x -> -infinity
    else:
        S_max = b  # Max value as x -> -infinity
        S_min = a + b  # Min value as x -> infinity

    # The target sigmoid value at fraction f of the way
    S_target = f * (S_max - S_min) + S_min

    # Solve for x using the derived formula
    x = d - (1 / c) * np.log((S_target - b) / (a - (S_target - b)))

    return x


if __name__ == "__main__":
    main()
