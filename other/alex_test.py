
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import uproot


def main():
    file_path = './data/ENUBET/run362620_raw.root'
    tree_vars  = ['strip_data_0_0', 'strip_data_0_1', 'strip_data_1_0', 'strip_data_1_1', 'waveform_data_0_0', 'digi_timestamp_0',
                  'waveform_data_0_1', 'digi_timestamp_0', 'waveform_data_0_2', 'digi_timestamp_0', 'waveform_data_0_3', 'digi_timestamp_0', 'waveform_data_0_4',
                  'digi_timestamp_0', 'waveform_data_0_5', 'digi_timestamp_0', 'waveform_data_0_6', 'digi_timestamp_0', 'waveform_data_0_7', 'digi_timestamp_0',
                  'waveform_data_1_0', 'digi_timestamp_1', 'waveform_data_1_1', 'digi_timestamp_1', 'waveform_data_1_2', 'digi_timestamp_1', 'waveform_data_1_3',
                  'digi_timestamp_1', 'waveform_data_1_5', 'digi_timestamp_1', 'waveform_data_2_0', 'digi_timestamp_2', 'fasttrig_data_2_0', 'fasttrig_data_2_1',
                  'fasttrig_data_2_2', 'fasttrig_data_2_3', 'waveform_data_2_2', 'digi_timestamp_2', 'fasttrig_data_2_0', 'fasttrig_data_2_1', 'fasttrig_data_2_2', 'fasttrig_data_2_3',
                  'waveform_data_2_4', 'digi_timestamp_2', 'fasttrig_data_2_0', 'fasttrig_data_2_1', 'fasttrig_data_2_2', 'fasttrig_data_2_3', 'sys_timestamp']

    #read a root file and print the branch names of the tree
    with uproot.open(file_path) as file:
        tree = file[file.keys()[0]]
        print(tree.keys())

    #plot the entries of the waveform_data_0_2 branch

    waveform_data_0_1 = tree['waveform_data_0_1']
    waveform_data_0_1_np = waveform_data_0_1.array(entry_stop=500).to_numpy()
    print(waveform_data_0_1_np)
    print(type(waveform_data_0_1_np))
    print(waveform_data_0_1_np.shape)
    for i in range(50):
        plot_waveform(waveform_data_0_1_np[i])
        plt.show()

    #plot the histogram of the minimum amplitudes of the waveforms
    #min_ampl_hist(waveform_data_0_1_np)
    #plt.show()

    print('bonzo')

#def the plot_waveform function using data that have y values exeading a specific value
def min_ampl_hist(waveforms):
    """
    Plot histogram of minimum amplitudes of waveforms
    :param waveforms:   waveforms to plot
    :return:
    """
    min_ampl_vals = np.min(waveforms, axis=1)
    plt.hist(min_ampl_vals, bins=100)
    plt.xlabel('Minimum amplitude')
    plt.ylabel('Entries')
    plt.title('Minimum amplitude histogram')
    plt.show()

def plot_waveform(waveform):
    """
    Plot a waveform
    :param waveform:    waveform to plot
    :return:
    """
    min_point = np.min(waveform)
    min_time = np.argmin(waveform)
    window_size_before_min, window_size_after_min = 450, 600
    time_window = np.arange(min_time - window_size_before_min, min_time + window_size_after_min)
    time_window = np.clip(time_window, 0, len(waveform) - 1)  # Ensure indices are within bounds

    # Define error values (for example, standard deviation or a constant value)
    error_values = np.full_like(waveform[time_window], 0.1)  # Example: constant error of 0.1

    # Plot with error bars
    plt.errorbar(time_window, waveform[time_window], yerr=error_values, fmt='o', markersize=5, capsize=3)
    plt.xlabel('Time[ns]')
    plt.ylabel('ADC')
    plt.title('Waveform')
    #plt.show()

if __name__ == "__main__":
    main()
