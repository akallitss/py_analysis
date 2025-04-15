
import os
import pandas as pd
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit as cf
from TriggerEvaluation.Measure import Measure


def get_tree(run_path, tree_name='Pico'):
    # rootfile = uproot.open(f"/eos/home-a/akallits/SWAN_projects/pico/Run{run}_parameters.root");
    rootfile = uproot.open(run_path)

    tree = rootfile[tree_name]

    return tree


def get_df(tree, *indexes):
    branches =[tree.keys()[i] for i in indexes]

    dfs = []
    for df, report in tree.iterate(branches, step_size='1 MB', library='pd', report = True):
        print(report)
        dfs.append(df)
    dataframe = pd.concat(dfs, ignore_index=True)

    return dataframe


def get_df_branches(tree, branches, step_size='1000 MB'):
    dfs = []
    for df, report in tree.iterate(branches, step_size=step_size, library='pd', report = True):
        print(report)
        dfs.append(df)
    dataframe = pd.concat(dfs, ignore_index=True)

    return dataframe


def rename_tree_branches(df,*col_names):
    l = len(col_names[0])  
    for i in range(l):
        df.rename(columns = {col_names[0][i]:col_names[1][i]}, inplace = True)
    
    return df


def store_new_tree(df):
    newfile = uproot.recreate("./example.root")
    newfile['tree'] = df
    newfile.close()


def set_matplotlib_style():
    plt.style.use(['default'])
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['text.usetex'] = False
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.framealpha'] = 1

    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titleweight'] = 'bold'

def get_single_peak(df, channel):
    "Extract a single peak for each channel"
    peak_no = 0
    columns = df.columns
    for col in columns:
        if f'peakparam_{channel}/peakparam_{channel}' in col:
            df[col] = df[col].apply(lambda x: x[peak_no] if len(x) > peak_no else np.nan)

def get_single_track(df, channel, track_no=None):
    "Extract a single track for each channel"
    track_no = 0 if track_no is None else track_no
    columns = df.columns
    for col in columns:
        if f'peakparam_{channel}/peakparam_{channel}.hit' in col:
            #select the first track
            if isinstance(track_no, int):
                df[col] = df[col].apply(lambda x: x[track_no] if isinstance(x, list) and len(x) > track_no else np.nan)
            else:
                df[col] = np.array(df[col])[track_no]


def get_best_tracks(df, channels, chi2_quality, plot=False):
    median_x, median_y = get_center_all_tracks(df, 'C1')
    ak_arrays = make_chi2_cut_tracks(df, channels, chi2_quality, plot=plot)

    min_indices = get_closest_track_indices_ak(ak_arrays, 'C1', (median_x, median_y))

    for key in ak_arrays.fields:
        min_chi2_tracks = ak_arrays[key][ak.local_index(ak_arrays[key]) == min_indices[:, None]]
        df[key] = ak.to_numpy(ak.firsts(min_chi2_tracks))


def make_chi2_cut_tracks(df, channels, chi2_quality=3, plot=False):
    ak_arrays = {'chi2track': ak.Array(df['chi2track'].to_list())}
    for channel in channels:
        for xy in ['X', 'Y']:
            hit_col = f'hit{xy}_{channel}'
            ak_arrays.update({hit_col: ak.Array(df[hit_col].to_list())})

    ak_arrays = ak.Array(ak_arrays)

    og_events_with_tracks = ak.sum(ak.count(ak_arrays['chi2track'], axis=1) > 0)

    if plot:
        chi2track = ak.flatten(ak_arrays['chi2track'])
        chi2track = ak.to_numpy(chi2track)

        fig, ax = plt.subplots()
        # ax.hist(chi2track, bins=300, log=True, alpha=0.7, color='blue', edgecolor='black')
        ax.hist(chi2track, bins=300, alpha=0.7, color='blue', edgecolor='black', histtype='stepfilled')
        # ax.set_ylim(top=100)
        ax.axvline(chi2_quality, color='red')

        ax.set_xlabel('chi2track')
        ax.set_yscale('log')
        ax.set_ylabel('# of events')
        ax.set_title('Chi2 Track Distribution')
        ax.grid(True, linestyle='--', alpha=0.6)

    ak_arrays = ak_arrays[ak_arrays['chi2track'] < chi2_quality]

    after_events_with_tracks = ak.sum(ak.count(ak_arrays['chi2track'], axis=1) > 0)

    print(f'Original number of events with tracks: {og_events_with_tracks}')
    print(f'After chi2 cut number of events with tracks: {after_events_with_tracks}')
    print(f'percentage of events sfter cuts chi2: {after_events_with_tracks / og_events_with_tracks * 100:.2f}%')
    print(f'percentage of events with tracks: {after_events_with_tracks / len(df) * 100:.2f}%')

    if plot:
        chi2track_flat = ak.flatten(ak_arrays['chi2track'])
        chi2track = ak.to_numpy(chi2track_flat)
        print(chi2track.min(), chi2track.max())
        fig, ax = plt.subplots()
        # ax.hist(chi2track, bins=300, log=True, alpha=0.7, color='blue', edgecolor='black')
        ax.hist(chi2track, bins=300, alpha=0.7, color='blue', edgecolor='black', histtype='stepfilled')
        ax.set_ylim(top=100)
        ax.axvline(chi2_quality, color='red')

        ax.set_xlabel('chi2track')
        ax.set_ylabel('# of events')
        ax.set_title('Chi2 Track Distribution after chi2 cut')
        ax.grid(True, linestyle='--', alpha=0.6)

    return ak_arrays


def get_closest_track_indices(df, channel, det_center=None):
    "Extract a single track for each channel"
    # Convert list-like column entries to NumPy arrays
    # df[f'hitX_{channel}'] = df[f'peakparam_{channel}/peakparam_{channel}.hitX[200]'].apply(np.array)
    # df[f'hitY_{channel}'] = df[f'peakparam_{channel}/peakparam_{channel}.hitY[200]'].apply(np.array)

    # Function to find the closest hit per row
    def get_closest_hit(hit_x, hit_y):
        if len(hit_x) == 0:  # Handle empty arrays
            return np.nan
        distances = (np.array(hit_x) - det_center[0]) ** 2 + (np.array(hit_y) - det_center[1]) ** 2  # Squared Euclidean distance
        min_idx = np.argmin(distances)  # Index of the closest hit
        # if min_idx != 0:
        #     print(f'hit_x: {hit_x}, hit_y: {hit_y}, distances: {distances}, min_idx: {min_idx}')
        return int(min_idx)

    # Apply function row-wise
    closest_hits = df.apply(lambda row: get_closest_hit(row[f'hitX_{channel}'], row[f'hitY_{channel}']), axis=1)

    # Split results into separate columns
    # df[['closest_hit_index']] = pd.DataFrame(closest_hits.tolist(), index=df.index)
    df['closest_hit_index'] = closest_hits


def get_closest_track_indices_ak(ak_arrays, channel, det_center=None):
    "Extract a single track for each channel"

    distances = (ak_arrays[f'hitX_{channel}'] - det_center[0]) ** 2 + (ak_arrays[f'hitY_{channel}'] - det_center[1]) ** 2  # Squared Euclidean distance
    min_idxs = ak.argmin(distances, axis=1)  # Index of the closest hit
    return min_idxs


def get_center_all_tracks(df_in, channel):
    "Extract the center of distribution for all tracks"
    # Convert list-like column entries to NumPy arrays
    df = df_in.copy()
    hit_xs = df[f'hitX_{channel}'].apply(np.array)
    hit_ys = df[f'hitY_{channel}'].apply(np.array)

    # Apply filtering using NumPy vectorized operations
    # filtered_xs = hit_xs.apply(lambda x: x[x > -1111])
    # filtered_ys = hit_ys.apply(lambda y: y[y > -1111])

    # Concatenate all valid hit coordinates into a single NumPy array for efficiency
    good_hit_xs = np.concatenate(hit_xs.values)
    good_hit_ys = np.concatenate(hit_ys.values)

    # Get median x and y
    return np.median(good_hit_xs), np.median(good_hit_ys)


def get_run_event_start(df):
    hitx = df['hitX_C1'].iloc[0]

    if df['hitX_C1'].iloc[1] != df['hitX_C1'].iloc[0]:
        return None
    for row_i in range(len(df['hitX_C1'])):
        if df['hitX_C1'].iloc[row_i] != hitx:
            print(f'{row_i} -> {hitx}')
            event_start = row_i
            break
    return event_start


def poly_even_fit(x, x0, a, b, c):
    return a + b * (x - x0)**2 + c * (x - x0)**4


def get_pad_center(charges, xs, ys, bin_width=0.5, min_tracks_per_2d_bin=20, min_avg_charge_per_2d_bin=4,
                        plot=False, plot_only=False):
    """
    Estimate pad center from average signal charge distribution
    Parameters:
        charges (np.ndarray): Array of charge values
        xs (np.ndarray): Array of x coordinates
        ys (np.ndarray): Array of y coordinates
        plot (bool): Whether to plot the results
        bin_width (float): [mm] Width of the 2D histogram bins. Same in both x and y.
        min_tracks_per_2d_bin (int): Minimum number of tracks per bin
        min_avg_charge_per_2d_bin (float): [pC] Minimum average charge per bin
        plot_only (bool): If True, only plot the results without returning pad center estimates
    Returns:
        tuple(Measure, Measure): Tuple of Measure objects representing the x and y pad center estimates
    """
    parameter_string = (f'Bin Width: {bin_width} mm\nMinimum Tracks per Bin: {min_tracks_per_2d_bin}'
                        f'\nMinimum Average Charge per Bin: {min_avg_charge_per_2d_bin} pC')


    # Define histogram bins
    bin_x_min, bin_x_max = np.min(xs), np.max(xs)
    bin_y_min, bin_y_max = np.min(ys), np.max(ys)
    print(f"bin_x_min: {bin_x_min}, bin_x_max: {bin_x_max}")
    print(f"bin_y_min: {bin_y_min}, bin_y_max: {bin_y_max}")
    if np.isnan(bin_x_min) or np.isnan(bin_x_max) or np.isinf(bin_x_min) or np.isinf(bin_x_max):
        raise ValueError("Invalid bin range for X!")
    if np.isnan(bin_y_min) or np.isnan(bin_y_max) or np.isinf(bin_y_min) or np.isinf(bin_y_max):
        raise ValueError("Invalid bin range for Y!")
    # # avoid nan values
    # bin_x_min = np.nanmin(xs)
    # bin_x_max = np.nanmax(xs)
    bins_x = np.arange(bin_x_min, bin_x_max + bin_width, bin_width)
    bins_y = np.arange(bin_y_min, bin_y_max + bin_width, bin_width)
    bin_centers_x = (bins_x[1:] + bins_x[:-1]) / 2
    bin_centers_y = (bins_y[1:] + bins_y[:-1]) / 2


    # Create 2D histogram (track multiplicity)
    xytrks, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y])

    # Create weighted histogram
    xytrksW_sum, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y], weights=charges)

    # Create boolean mask for bins where xytrks < x
    min_track_mask = xytrks < min_tracks_per_2d_bin

    # Set corresponding bins to zero in both histograms
    xytrks[min_track_mask] = 0
    xytrksW_sum[min_track_mask] = 0

    # Normalize by dividing the weighted histogram by the original histogram
    xytrksW = xytrksW_sum / np.where(xytrks > 0, xytrks, 1)  # Avoid division by zero

    min_avg_charge_mask = xytrksW < min_avg_charge_per_2d_bin

    xytrks[min_avg_charge_mask] = 0
    xytrksW[min_avg_charge_mask] = 0
    xytrksW_sum[min_avg_charge_mask] = 0

    # Project 2D track multiplicity histograms to 1D
    hist_1d_x = np.sum(xytrks, axis=1)
    hist_1d_y = np.sum(xytrks, axis=0)

    # Find nonzero bins for dynamic range
    nonzero_x = np.nonzero(hist_1d_x)[0]
    nonzero_y = np.nonzero(hist_1d_y)[0]

    if nonzero_x.size > 0:
        x_min, x_max = bins_x[nonzero_x[0]], bins_x[nonzero_x[-1] + 1]
    else:
        x_min, x_max = bins_x[0], bins_x[-1]

    if nonzero_y.size > 0:
        y_min, y_max = bins_y[nonzero_y[0]], bins_y[nonzero_y[-1] + 1]
    else:
        y_min, y_max = bins_y[0], bins_y[-1]

    x_range, y_range = x_max - x_min, y_max - y_min

    if plot:  # Plot 1D of xs (track multiplicity)
        fig, axs = plt.subplots(ncols=2, sharey='all', figsize=(12, 6))

        axs[0].step(bin_centers_x, hist_1d_x, where='mid', label="Projection on X")
        axs[1].step(bin_centers_y, hist_1d_y, where='mid', label="Projection on Y")

        extend_range = 0.1
        axs[0].set_xlim(x_min - extend_range * x_range, x_max + extend_range * x_range)
        axs[1].set_xlim(y_min - extend_range * y_range, y_max + extend_range * y_range)

        axs[0].set_ylim(bottom=0)
        axs[1].set_ylim(bottom=0)
        axs[0].legend()
        axs[1].legend()
        fig.subplots_adjust(wspace=0.0)


    # Get pad center from 1D projections after filtering on minimum tracks and minimum average charge
    sum_x_charge = np.sum(xytrksW_sum, axis=1)
    sum_y_charge = np.sum(xytrksW_sum, axis=0)
    sum_x_tracks = np.sum(xytrks, axis=1)
    sum_y_tracks = np.sum(xytrks, axis=0)
    avg_x_charge = sum_x_charge / np.where(sum_x_tracks > 0, sum_x_tracks, 1)
    avg_y_charge = sum_y_charge / np.where(sum_y_tracks > 0, sum_y_tracks, 1)

    if not plot_only:
        # avg_x_charge_err = np.abs(avg_x_charge) / np.sqrt(np.where(sum_x_tracks > 0, sum_x_tracks, 1))
        avg_x_charge_err = np.where(sum_x_tracks > 0, np.abs(avg_x_charge) / np.sqrt(sum_x_tracks), 1)
        # avg_y_charge_err = np.abs(avg_y_charge) / np.sqrt(np.where(sum_y_tracks > 0, sum_y_tracks, 1))
        avg_y_charge_err = np.where(sum_y_tracks > 0, np.abs(avg_y_charge) / np.sqrt(sum_y_tracks), 1)

        x_fit_mask = (bin_centers_x > x_min) & (bin_centers_x < x_max)
        avg_x_charge_x0_guess = np.nanmedian(xs)
        avg_x_charge_max = np.max(avg_x_charge[x_fit_mask])
        avg_x_charge_curvature_guess = avg_x_charge_max / (x_range / 2) ** 2
        p0_x = [avg_x_charge_x0_guess, avg_x_charge_max, -avg_x_charge_curvature_guess, 0]
        popt_x_charge, pcov_x_charge = cf(poly_even_fit, bin_centers_x[x_fit_mask], avg_x_charge[x_fit_mask], p0=p0_x,
                                          sigma=avg_x_charge_err[x_fit_mask], absolute_sigma=True)
        perr_x_charge = np.sqrt(np.diag(pcov_x_charge))
        meas_x_charge = [Measure(val, err) for val, err in zip(popt_x_charge, perr_x_charge)]

        y_fit_mask = (bin_centers_y > y_min) & (bin_centers_y < y_max)
        avg_y_charge_y0_guess = np.nanmedian(ys)
        avg_y_charge_max = np.max(avg_y_charge[y_fit_mask])
        avg_y_charge_curvature_guess = avg_y_charge_max / (y_range / 2) ** 2
        p0_y = [avg_y_charge_y0_guess, avg_y_charge_max, -avg_y_charge_curvature_guess, 0]
        popt_y_charge, pcov_y_charge = cf(poly_even_fit, bin_centers_y[y_fit_mask], avg_y_charge[y_fit_mask], p0=p0_y,
                                          sigma=avg_y_charge_err[y_fit_mask], absolute_sigma=True)
        perr_y_charge = np.sqrt(np.diag(pcov_y_charge))
        meas_y_charge = [Measure(val, err) for val, err in zip(popt_y_charge, perr_y_charge)]

    if plot:  # Plot 1D of xs and ys weighted by charge (formatted like above)
        fig, axs = plt.subplots(ncols=2, sharey='all', figsize=(12, 6))

        axs[0].step(bin_centers_x, avg_x_charge, where='mid', label="Avg Charge X", c='k', alpha=0.5)
        axs[1].step(bin_centers_y, avg_y_charge, where='mid', label="Avg Charge Y", c='k', alpha=0.5)

        if not plot_only:
            axs[0].errorbar(bin_centers_x[x_fit_mask], avg_x_charge[x_fit_mask], yerr=avg_x_charge_err[x_fit_mask], ls='none',
                            c='k', marker='o', ms=5, capsize=5, elinewidth=0.5)
            axs[1].errorbar(bin_centers_y[y_fit_mask], avg_y_charge[y_fit_mask], yerr=avg_y_charge_err[y_fit_mask], ls='none',
                            c='k', marker='o', ms=5, capsize=5, elinewidth=0.5)

            axs[0].plot(bin_centers_x, poly_even_fit(bin_centers_x, *p0_x), color='gray', ls='-', alpha=0.1)
            axs[0].plot(bin_centers_x, poly_even_fit(bin_centers_x, *popt_x_charge), color='red', ls='--')
            axs[1].plot(bin_centers_y, poly_even_fit(bin_centers_y, *p0_y), color='gray', ls='-', alpha=0.1)
            axs[1].plot(bin_centers_y, poly_even_fit(bin_centers_y, *popt_y_charge), color='red', ls='--')

            axs[0].annotate(
                f'Pad X-Center: {meas_x_charge[0]}',
                xy=(0.5, 0.05), xycoords='axes fraction',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
            )
            axs[1].annotate(
                f'Pad Y-Center: {meas_y_charge[0]}',
                xy=(0.5, 0.05), xycoords='axes fraction',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
            )

            axs[0].axvline(popt_x_charge[0], color='salmon', ls='-', alpha=0.5)
            axs[1].axvline(popt_y_charge[0], color='salmon', ls='-', alpha=0.5)

        axs[0].annotate(
            parameter_string,
            xy=(0.5, 0.2), xycoords='axes fraction',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
        )

        axs[1].annotate(
            parameter_string,
            xy=(0.5, 0.2), xycoords='axes fraction',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
        )

        axs[0].set_xlim(x_min - extend_range * x_range, x_max + extend_range * x_range)
        axs[1].set_xlim(y_min - extend_range * y_range, y_max + extend_range * y_range)
        axs[0].set_ylim(bottom=0, top=np.max(avg_x_charge) * 1.1)
        axs[1].set_ylim(bottom=0, top=np.max(avg_y_charge) * 1.1)

        fig.suptitle('Average Charge per Bin X, Y Distributions')
        fig.subplots_adjust(wspace=0.0)

        axs[0].legend()
        axs[1].legend()

    if plot:  # Plot 2D track multiplicity and average charge per track distributions after filtering
        # Mask xytrks <1
        masked_xy_trks = np.ma.masked_less(xytrks, 1)
        masked_xy_trksW = np.ma.masked_equal(xytrksW, 0)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot xytrks
        im1 = axs[0].imshow(masked_xy_trks.T, origin="lower", extent=[bin_x_min, bin_x_max, bin_y_min, bin_y_max],
                            cmap='jet', aspect="auto")
        axs[0].set_title("Track Multiplicity")
        axs[0].set_xlabel("x [mm]")
        axs[0].set_ylabel("y [mm]")
        # axs[0].set_xlim(x_min - extend_range * x_range, x_max + extend_range * x_range)
        # axs[0].set_ylim(y_min - extend_range * y_range, y_max + extend_range * y_range)
        fig.colorbar(im1, ax=axs[0])

        # Plot xytrksW
        # im2 = axs[1].imshow(masked_xy_trksW.T, origin="lower", extent=[0, 50, 0, 50], cmap='jet', aspect="auto", vmax=np.max(masked_xy_trksW) / 1.8)
        im2 = axs[1].imshow(masked_xy_trksW.T, origin="lower", extent=[bin_x_min, bin_x_max, bin_y_min, bin_y_max],
                            cmap='jet', aspect="auto")
        axs[1].set_title("Average Charge per Track")
        axs[1].set_xlabel("x [mm]")
        axs[1].set_ylabel("y [mm]")
        if not plot_only:
            axs[0].scatter(popt_x_charge[0], popt_y_charge[0], marker='x', c='k')
            axs[1].scatter(popt_x_charge[0], popt_y_charge[0], marker='x', c='k')
            axs[1].annotate(
                f'Pad Center: ({meas_x_charge[0]}, {meas_y_charge[0]})',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
            )
        # axs[1].set_xlim(x_min - extend_range * x_range, x_max + extend_range * x_range)
        # axs[1].set_ylim(y_min - extend_range * y_range, y_max + extend_range * y_range)
        fig.colorbar(im2, ax=axs[1])

    if plot_only:
        return None, None
    else:
        return meas_x_charge[0], meas_y_charge[0]


def time_walk_func_power_law(x, a, b, c):
    return c + a * x ** b

def time_walk_double_exponential(x, a1, l1, a2, l2, c):
    return a1 * np.exp(l1 * x) + a2 * np.exp(l2 * x) + c

def time_walk_linear(x, a, b):
    return a * x + b

def double_expo(x, *p):
    return np.exp(p[0]*x+p[1])+np.exp(p[2]*x+p[3])+p[4]

def gaus(x, a, mu, sigma):
    return a * np.exp(-(x-mu)**2/(2*sigma**2))

def line(x, a, b):
    return a*x + b

def get_time_walk_parameterization(time_diff, charges, time_walk_func, time_walk_p0, percentile_cut=(None, None), binning_type='equal_stats', plot=False, plot_indiv_fits=False):
    """
    Get time walk correction for a given channel
    Parameters:
        time_diff (list): The time differences between micromegas and mcp
        charges (list): List of micromegas charges
        time_walk_func (function): The time walk function to fit
        time_walk_p0 (list): Initial parameters for the time walk function
        binning_type (str): Type of binning for the time walk correction
        percentile_cut (tuple): Percentile cut for the time difference
        plot (bool): Whether to plot the results
    Returns:
        tuple(Measure, Measure): Tuple of Measure objects representing the x and y pad center estimates
    """
    time_diff_na_filter = ~pd.isna(time_diff) & ~pd.isna(charges)

    time_diff = np.array(time_diff[time_diff_na_filter])
    charges = np.array(charges[time_diff_na_filter])

    sorted_indices = np.argsort(charges)
    charges, time_diff = charges[sorted_indices], time_diff[sorted_indices]

    # avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs = get_time_walk_binned(time_diff, charges)
    avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs = get_time_walk_binned_better(time_diff, charges, 100, binning_type, percentile_cut, plot_indiv_fits)

    try:
        popt_indiv, pcov_indiv = cf(time_walk_func, charges, time_diff, p0=time_walk_p0, maxfev=10000)
        pmeas_indiv = [Measure(val, err) for val, err in zip(popt_indiv, np.sqrt(np.diag(pcov_indiv)))]
    except RuntimeError as e:
        print(f"Error fitting individual points: {e}")
        popt_indiv, pcov_indiv, pmeas_indiv = None, None, None

    try:
        popt_dyn_bin, pcov_dyn_bin = cf(time_walk_func, avg_charges, med_time_diffs, sigma=std_err_time_diffs,
                                    absolute_sigma=True, p0=time_walk_p0, maxfev=10000)
        pmeas_dyn_bin = [Measure(val, err) for val, err in zip(popt_dyn_bin, np.sqrt(np.diag(pcov_dyn_bin)))]
    except RuntimeError as e:
        print(f"Error fitting dynamic bin: {e}")
        popt_dyn_bin, pcov_dyn_bin, pmeas_dyn_bin = None, None, None

    try:
        popt_gaus_fits, pcov_gaus_fits = cf(time_walk_func, avg_charges, gaus_means, sigma=gaus_mean_errs,
                                        absolute_sigma=True, p0=time_walk_p0, maxfev=10000)
        pmeas_gaus_fit = [Measure(val, err) for val, err in zip(popt_gaus_fits, np.sqrt(np.diag(pcov_gaus_fits)))]
    except RuntimeError as e:
        print(f"Error fitting Gaussian fits: {e}")
        popt_gaus_fits, pcov_gaus_fits, pmeas_gaus_fit = None, None, None

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        binning_t20_diff = np.arange(0, 15, 0.1)
        ax.hist(time_diff, bins=binning_t20_diff)
        ax.set_xlabel('SAT Raw [ns]')


        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(charges, time_walk_func(charges, *time_walk_p0), color='gray', alpha=0.2)
        if popt_indiv is not None:
            ax.plot(charges, time_walk_func(charges, *popt_indiv), color='red', ls='--')
        ax.scatter(charges, time_diff, alpha=0.5)
        ax.set_ylim(np.percentile(time_diff, 1), np.percentile(time_diff, 99))
        ax.set_xlabel('Charge [pC]')
        ax.set_ylabel('SAT Individual [ns]')

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(avg_charges, med_time_diffs, yerr=std_err_time_diffs, fmt='.', color='black',
                    label='Average charges')
        if popt_dyn_bin is not None:
            ax.plot(charges, time_walk_func(charges, *popt_dyn_bin), ls='--', color='red', label='Dynamic bin')
        ax.set_xlabel('Total Charge [pC]')
        ax.set_ylabel('SAT Medians [ns]')


        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(avg_charges, gaus_means, yerr=gaus_mean_errs, fmt='.', color='black', label='Average charges')
        if popt_gaus_fits is not None:
            ax.plot(charges, time_walk_func(charges, *popt_gaus_fits), ls='--', color='red', label='Dynamic bin')
        ax.set_xlabel('Total Charge [pC]')
        ax.set_ylabel('SAT Gaussian [ns]')


        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(charges, time_diff, alpha=0.2)
        if popt_dyn_bin is not None:
            ax.plot(charges, time_walk_func(charges, *popt_dyn_bin), color='red', ls='--', label='Median of Bin Fit')
        if popt_gaus_fits is not None:
            ax.plot(charges, time_walk_func(charges, *popt_gaus_fits), color='green', ls='--', label='Gaus Fit of Bin Fit')
        if popt_indiv is not None:
            ax.plot(charges, time_walk_func(charges, *popt_indiv), color='blue', ls='--', label='Individual Point Fit')
        ax.set_xlabel('Charge [pC]')
        ax.set_ylabel('SAT Raw [ns]')
        ax.set_ylim(np.percentile(time_diff, 1), np.percentile(time_diff, 99))
        ax.legend()

    return pmeas_indiv, pmeas_dyn_bin, pmeas_gaus_fit


# def get_time_walk_binned(time_diff, charges, n_bins=100, plot=False):
#     n_event_bins = int(len(charges) / n_bins)
#     print('n_event_bins:', n_event_bins)
#     n_gaus_bins = 10
#     bin_start = 0
#     avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs = [], [], [], [], []
#     while bin_start < len(charges) - 1:
#         bin_end = bin_start + n_event_bins
#         if bin_end > len(charges):
#             bin_end = len(charges) - 1
#             print(f'{bin_end - bin_start} points in the last bin')
#         bin_charges = charges[bin_start:bin_end]
#         bin_time_diffs = time_diff[bin_start:bin_end]
#         avg_charges.append(np.mean(bin_charges))
#         med_time_diffs.append(np.median(bin_time_diffs))
#         std_err = np.std(bin_time_diffs) / np.sqrt(len(bin_time_diffs)) if len(bin_time_diffs) > 0 else np.nan
#         std_err = std_err if std_err > 0 else 1
#         std_err_time_diffs.append(std_err)
#
#         bin_time_diff_hist, bin_time_diff_charge_bin_edges = np.histogram(bin_time_diffs, bins=n_gaus_bins)
#         bin_time_diff_charge_bin_centers = (bin_time_diff_charge_bin_edges[1:] + bin_time_diff_charge_bin_edges[
#                                                                                  :-1]) / 2
#         p0_gaus_bin = [np.max(bin_time_diff_hist), np.mean(bin_time_diffs), np.std(bin_time_diffs)]
#         try:
#             popt_gaus_bin, pcov_gaus_bin = cf(gaus, bin_time_diff_charge_bin_centers, bin_time_diff_hist,
#                                               p0=p0_gaus_bin, maxfev=10000)
#             perr_gaus_bin = np.sqrt(np.diag(pcov_gaus_bin))
#
#             if plot:
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 bin_time_diff_charg_bin_widths = np.diff(bin_time_diff_charge_bin_edges)
#                 ax.bar(bin_time_diff_charge_bin_centers, bin_time_diff_hist, width=bin_time_diff_charg_bin_widths,
#                        color='black')
#                 x_plot = np.linspace(bin_time_diff_charge_bin_edges[0], bin_time_diff_charge_bin_edges[-1], 200)
#                 ax.plot(x_plot, gaus(x_plot, *p0_gaus_bin), color='gray', alpha=0.2)
#                 ax.plot(x_plot, gaus(x_plot, *popt_gaus_bin), color='red')
#                 ax.set_title(f'Fit from {charges[bin_start]:.2f} pC to {charges[bin_end]:2f} pC')
#
#             gaus_means.append(popt_gaus_bin[1])
#             gaus_mean_errs.append(perr_gaus_bin[1])
#
#         except RuntimeError:
#             print(f'gaus_bin_hist failed for bin {charges[bin_start]:.2f} pC to {charges[bin_end]:2f} pC')
#             gaus_means.append(p0_gaus_bin[1])
#             gaus_mean_errs.append(p0_gaus_bin[1])
#
#         bin_start = bin_end
#
#     return avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs


def get_time_walk_binned_better(time_diff, charges, n_bins=100, binning_type='equal_steps', percentile_cut=(None, None), plot=False):
    if binning_type == 'equal_stats':
        print(f'n_event_bins: {int(len(charges) / n_bins)}')
        charge_bin_edges = np.percentile(charges, np.linspace(0, 100, n_bins + 1))
    elif binning_type == 'equal_steps':
        charge_bin_edges = np.linspace(np.min(charges), np.max(charges), n_bins + 1)
    else:
        raise ValueError(f"Invalid binning type: {binning_type}")

    if percentile_cut[0] is not None or percentile_cut[1] is not None:
        percentile_filter = make_percentile_cuts(time_diff, percentile_cuts=percentile_cut, return_what='filter')
        time_diff = time_diff[percentile_filter]
        charges = charges[percentile_filter]

    avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs = [], [], [], [], []
    for i in range(n_bins):
        bin_charge_min, bin_charge_max = charge_bin_edges[i], charge_bin_edges[i + 1]
        charges_filter = (charges > bin_charge_min) & (charges < bin_charge_max)
        bin_charges = charges[charges_filter]
        bin_time_diffs = time_diff[charges_filter]

        bin_n_events = np.sum(~np.isnan(bin_time_diffs))
        if bin_n_events == 0:
            print(f'No events in bin {i} ({bin_charge_min:.2f} pC to {bin_charge_max:.2f} pC)')
            continue
        elif bin_n_events == 1:
            print(f'Only one event in bin {i} ({bin_charge_min:.2f} pC to {bin_charge_max:.2f} pC)')
            med = np.nanmedian(bin_time_diffs)
            # Use last bin std if it exists
            if len(std_err_time_diffs) > 0:
                std = std_err_time_diffs[-1]
            else:
                std = med
            med_time_diffs.append(med)
            std_err_time_diffs.append(std)
            gaus_means.append(med)
            gaus_mean_errs.append(std)
            if np.isnan(med) or np.isnan(std):
                print(f'NaN in bin {i} ({bin_charge_min:.2f} pC to {bin_charge_max:.2f} pC)')
            avg_charges.append(np.nanmean(bin_charges))
            continue

        avg_charges.append(np.nanmean(bin_charges))
        if np.isnan(avg_charges[-1]):
            print(f'NaN in bin {i} ({bin_charge_min:.2f} pC to {bin_charge_max:.2f} pC)')

        med_time_diffs.append(np.nanmedian(bin_time_diffs))
        std_err = np.nanstd(bin_time_diffs) / np.sqrt(bin_n_events) if bin_n_events > 0 else 1
        std_err = std_err if std_err > 0 else 1
        std_err_time_diffs.append(std_err)

        bin_n_fit_bins = 2 * (np.percentile(bin_time_diffs, 75) - np.percentile(bin_time_diffs, 25))
        bin_n_fit_bins /=  bin_n_events**(1/3)  # Freedman-Diaconis Rule
        bin_n_fit_bins = max(int(bin_n_fit_bins), 10)
        fit_meases = fit_time_diffs(bin_time_diffs, n_bins=bin_n_fit_bins, min_events=10)

        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(bin_time_diffs, bins=bin_n_fit_bins, histtype='stepfilled', color='black')
            x_plot = np.linspace(np.min(bin_time_diffs), np.max(bin_time_diffs), 200)
            # ax.plot(x_plot, gaus(x_plot, *p0_gaus_bin), color='gray', alpha=0.2)
            ax.plot(x_plot, gaus(x_plot, *[x.val for x in fit_meases]), color='red')
            ax.set_title(f'Fit from {bin_charge_min:.2f} pC to {bin_charge_max:2f} pC')

        gaus_mean = fit_meases[1].val if not np.isnan(fit_meases[1].val) else np.nanmean(bin_time_diffs)
        gaus_mean_err = fit_meases[1].err if not np.isnan(fit_meases[1].err) else np.nanstd(bin_time_diffs)
        gaus_means.append(gaus_mean)
        gaus_mean_errs.append(gaus_mean_err)

    return avg_charges, med_time_diffs, std_err_time_diffs, gaus_means, gaus_mean_errs


def make_percentile_cuts(data, percentile_cuts=(None,None), return_what='data'):
    if len(data) == 0:
        return data

    if percentile_cuts[0] is not None and percentile_cuts[1] is not None:
        low_percentile = np.percentile(data, percentile_cuts[0])
        high_percentile = np.percentile(data, percentile_cuts[1])
        percentile_filter = (data > low_percentile) & (data < high_percentile)
    elif percentile_cuts[0] is not None:
        low_percentile = np.percentile(data, percentile_cuts[0])
        percentile_filter = data > low_percentile
    elif percentile_cuts[1] is not None:
        high_percentile = np.percentile(data, percentile_cuts[1])
        percentile_filter = data < high_percentile
    else:
        return data

    if return_what == 'filter':
        return percentile_filter
    else:
        return data[percentile_filter]

def generate_line_scan(x_center, y_center, scan_radius, n_steps, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    ts = np.linspace(-scan_radius, scan_radius, n_steps + 1)
    xs_scan = x_center + ts * np.cos(angle_rad)
    ys_scan = y_center + ts * np.sin(angle_rad)
    return list(zip(xs_scan, ys_scan))


def relative_distances(xy_pairs, x_center, y_center):
    return [np.hypot(x - x_center, y - y_center) * np.sign((x - x_center) * np.cos(theta_rad) + (y - y_center) * np.sin(theta_rad))
            for x, y in xy_pairs]


def get_circle_scan(time_diffs, xs, ys, xy_pairs, ns_to_ps=False, radius=1, time_diff_lims=None, min_events=100, percentile_cuts=(None, None), plot=False):
    if ns_to_ps:
        time_diffs = time_diffs * 1000
    if time_diff_lims is not None:
        if ns_to_ps:
            time_diff_lims = np.array(time_diff_lims) * 1000
        time_diffs[(time_diffs < time_diff_lims[0]) | (time_diffs > time_diff_lims[1])] = np.nan

    resolutions, means, events = [], [], []
    for x, y in xy_pairs:
        # print(f'Circle Scan: ({x}, {y})')
        rs = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
        mask = rs < radius
        time_diffs_bin = time_diffs[mask]
        time_diffs_bin = np.array(time_diffs_bin[~np.isnan(time_diffs_bin)])

        time_diffs_bin = make_percentile_cuts(time_diffs_bin, percentile_cuts)

        n_events = time_diffs_bin.size

        hist_bin, bin_edges = np.histogram(time_diffs_bin, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fit_meases = fit_time_diffs(time_diffs_bin, n_bins=100, min_events=min_events)
        resolutions.append(fit_meases[2])
        means.append(fit_meases[1])
        events.append(n_events)

        if plot:
            fig, ax = plt.subplots()
            ax.bar(bin_centers, hist_bin, width=bin_edges[1] - bin_edges[0], align='center', alpha=0.5)
            x_plt = np.linspace(bin_centers[0], bin_centers[-1], 200)
            ax.plot(x_plt, gaus(x_plt, *[par.val for par in fit_meases]), color='red')
            ax.set_title(f'Circle Scan: ({x}, {y})')
            ax.set_xlabel('SAT [ps]')
            ax.set_ylabel('Counts')
            time_unit = 'ps' if ns_to_ps else 'ns'
            fit_str = f'Fit:\nEvents={n_events}\nA={fit_meases[0]}\nμ={fit_meases[1]} {time_unit}\nσ={fit_meases[2]} {time_unit}'
            ax.annotate(fit_str, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow'))
            fig.tight_layout()

    return resolutions, means, events


def get_ring_scan(time_diff_cor, rings, ring_bin_width, rs, percentile_cuts=(None, None), xs=None, ys=None, plot=False):

    if plot:
        if xs is not None and ys is not None:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
            ax, ax_xy = axs
            ax_xy.set_xlabel('X [mm]')
            ax_xy.set_ylabel('Y [mm]')
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(rs, time_diff_cor, alpha=0.2)
        ax.set_xlabel('Radial Distance from Pad Center [mm]')
        ax.set_ylabel('SAT [ns]')
        for r_bin_edge in rings:
            ax.axvline(r_bin_edge, color='red')
            ax.set_xlim(0, 10)
            ax.set_ylim(-2, 2)

            if xs is not None and ys is not None:
                pass
                # Make circle patch at r_bin_edge
                # ax_xy.add_patch(plt.Circle((center_x, center_y), r_bin_edge, color='red', fill=False, alpha=0.5))

    time_diff_binning = 100
    r_bin_centers, time_resolutions, mean_sats = [], [], []
    for r_bin_edge in rings:
        r_bin_upper_edge = r_bin_edge + ring_bin_width
        r_bin_filter = (rs > r_bin_edge) & (rs <= r_bin_upper_edge)
        time_diffs_r_bin = time_diff_cor[r_bin_filter]

        time_diffs_r_bin = make_percentile_cuts(time_diffs_r_bin, percentile_cuts)

        time_hist, bin_edges = np.histogram(time_diffs_r_bin, bins=time_diff_binning)
        time_diff_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fit_meases = fit_time_diffs(time_diffs_r_bin, n_bins=time_diff_binning, min_events=100)
        mean_sats.append(fit_meases[1]*1e3)
        time_resolutions.append(fit_meases[2] * 1e3)
        r_bin_centers.append((r_bin_edge + r_bin_upper_edge) / 2)

        if plot:
            fig_ring, ax_ring = plt.subplots(figsize=(8, 5))
            # ax_ring.bar(time_diff_bin_centers, time_hist, width=bin_edges[1] - bin_edges[0], align='center',
            #        label=f'{r_bin_edge:.2f} - {r_bin_upper_edge:.2f} mm')

            # Poisson error (sqrt of counts)
            yerr = np.where(time_hist > 0, np.sqrt(time_hist), 1)

            # Only plot bins with non-zero entries
            mask_nonzero = time_hist > 0
            ax_ring.errorbar(time_diff_bin_centers[mask_nonzero], time_hist[mask_nonzero],
                             yerr=yerr[mask_nonzero], fmt='o', color='black',
                             label=f'{r_bin_edge:.2f} - {r_bin_upper_edge:.2f} mm')

            fit_str = rf'$A = {fit_meases[0]}$' + '\n' + rf'$\mu = {fit_meases[1]}$' + '\n' + rf'$\sigma = {fit_meases[2]}$'
            ax_ring.plot(time_diff_bin_centers, gaus(time_diff_bin_centers, *[par.val for par in fit_meases]), color='red', label='Fit')
            ax_ring.set_xlabel('SAT [ns]')
            ax_ring.annotate(
            fit_str,
            xy=(0.1, 0.9), xycoords='axes fraction',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
            )
            ax_ring.legend()

            if xs is not None and ys is not None:
                xs_ring, ys_ring = xs[r_bin_filter], ys[r_bin_filter]
                ax_xy.scatter(xs_ring, ys_ring, alpha=0.2)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        r_bin_width = rings[1] - rings[0]
        ax.errorbar(r_bin_centers, [x.val for x in time_resolutions], yerr=[x.err for x in time_resolutions],
                    xerr=r_bin_width / 2, marker='o', color='black', ls='none')
        ax.set_xlabel('Radial Distance from Pad Center [mm]')
        ax.set_ylabel('Time Resolution [ps]')
        ax.set_ylim(bottom=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        r_bin_width = rings[1] - rings[0]
        ax.errorbar(r_bin_centers, [x.val for x in mean_sats], yerr=[x.err for x in mean_sats],
                    xerr=r_bin_width / 2, marker='o', color='black', ls='none')
        ax.set_xlabel('Radial Distance from Pad Center [mm]')
        ax.set_ylabel('SAT [ps]')

    return r_bin_centers, time_resolutions, mean_sats



def get_charge_scan(time_diffs, charges, charge_bins, ns_to_ps=False, time_diff_lims=None, min_events=100, percentile_cuts=(None, None), plot=False):
    if ns_to_ps:
        time_diffs = time_diffs * 1000
    if time_diff_lims is not None:
        if ns_to_ps:
            time_diff_lims = np.array(time_diff_lims) * 1000
        time_diffs[(time_diffs < time_diff_lims[0]) | (time_diffs > time_diff_lims[1])] = np.nan

    resolutions, means, events = [], [], []
    for charge_bin in charge_bins:
        charge_bin_low, charge_bin_high = charge_bin
        # print(f'Charge Scan: ({charge_bin_low} - {charge_bin_low})')
        mask = (charge_bin_low <= charges) & (charges < charge_bin_high)
        time_diffs_bin = time_diffs[mask]
        time_diffs_bin = np.array(time_diffs_bin[~np.isnan(time_diffs_bin)])

        time_diffs_bin = make_percentile_cuts(time_diffs_bin, percentile_cuts)

        n_events = time_diffs_bin.size

        hist_bin, bin_edges = np.histogram(time_diffs_bin, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fit_meases = fit_time_diffs(time_diffs_bin, n_bins=100, min_events=min_events)
        resolutions.append(fit_meases[2])
        means.append(fit_meases[1])
        events.append(n_events)

        if plot:
            fig, ax = plt.subplots()
            ax.bar(bin_centers, hist_bin, width=bin_edges[1] - bin_edges[0], align='center', alpha=0.5)
            x_plt = np.linspace(bin_centers[0], bin_centers[-1], 200)
            ax.plot(x_plt, gaus(x_plt, *[par.val for par in fit_meases]), color='red')
            ax.set_title(f'Charge Scan: ({charge_bin_low} - {charge_bin_low})')
            ax.set_xlabel('SAT [ps]')
            ax.set_ylabel('Counts')
            time_unit = 'ps' if ns_to_ps else 'ns'
            fit_str = f'Fit:\nEvents={n_events}\nA={fit_meases[0]}\nμ={fit_meases[1]} {time_unit}\nσ={fit_meases[2]} {time_unit}'
            ax.annotate(fit_str, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow'))
            fig.tight_layout()

    return resolutions, means, events


def plot_2D_circle_scan(scan_resolutions, scan_means, xs, ys, scan_events=None, radius=None):
    radius_str = f' radius={radius:.1f} mm' if radius is not None else ''

    scan_resolution_vals = [res.val for res in scan_resolutions]
    scan_mean_val = [mean.val for mean in scan_means]

    x_mesh, y_mesh = np.meshgrid(xs, ys)

    # Convert to 2D arrays
    scan_resolutions_2d = np.array(scan_resolution_vals).reshape(len(ys), len(xs))
    scan_means_2d = np.array(scan_mean_val).reshape(len(ys), len(xs))

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(scan_resolutions_2d, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto",
                  cmap="jet")
    plt.colorbar(c, label="Timing Resolution [ps]")
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Timing Resolution Heatmap{radius_str}")

    # Create the contour plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x_mesh, y_mesh, scan_resolutions_2d, levels=50, cmap="jet")

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("Timing Resolution [ps]")

    # Labels and title
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Timing Resolution Contour Plot{radius_str}")

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(scan_means_2d, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto",
                  cmap="jet")
    plt.colorbar(c, label="SAT [ps]")
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"SAT Heatmap{radius_str}")

    # Create the contour plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x_mesh, y_mesh, scan_means_2d, levels=50, cmap="jet")

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("SAT [ps]")

    # Labels and title
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"SAT Contour Plot{radius_str}")

    if scan_events is not None:
        scan_events_2d = np.array(scan_events).reshape(len(ys), len(xs))
        masked_scan_events_2d = np.ma.masked_equal(scan_events_2d, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.imshow(masked_scan_events_2d, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower",
                      aspect="auto", cmap="jet")
        plt.colorbar(c, label="Number of Events")
        ax.set_xlabel("X Position [mm]")
        ax.set_ylabel("Y Position [mm]")
        ax.set_title(f"Event Statistics Heatmap{radius_str}")


def plot_rise_fit_params(df):
    print(df[f'peakparam_C4/peakparam_C4.sigmoidR[4]'])
    sigmoidR = df[f'peakparam_C4/peakparam_C4.sigmoidR[4]'][pd.notnull(df['peakparam_C4/peakparam_C4.sigmoidR[4]'])]
    # print(sigmoidR)
    np_sigmoidR = np.array(sigmoidR.tolist())
    # print(np_sigmoidR)

    np_sigmoidR = np_sigmoidR[(np_sigmoidR[:, 0] > -999.) & (np_sigmoidR[:, 3] > -999.)]

    # Extract the amplitudes and baselines
    amplitudes = np_sigmoidR[:, 0]
    baselines = np_sigmoidR[:, 3]

    # Perform the division
    division = baselines / amplitudes

    # Create subplots in a 1x3 grid for the three histograms
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 1 row, 3 columns

    # Plot for amplitudes
    ax = axes[0]
    n_overflows_ampl = np.sum(amplitudes <= 0)  # Count of overflows (invalid or 0 values)
    ax.hist(amplitudes, bins=100, alpha=0.7, color='blue', edgecolor='black', zorder=2)
    ax.set_title('Amplitude Distribution')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Counts')
    ax.grid(True, zorder=0)

    # Add overflow text for amplitudes
    ax.text(
        0.95, 0.9,  # Position (relative to axes, 95% right, 90% up)
        f'Overflows: {n_overflows_ampl}',
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=12, color='red', ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    # Plot for baselines
    ax = axes[1]
    n_overflows_baseline = np.sum(baselines <= 0)  # Count of overflows (invalid or 0 values)
    ax.hist(baselines, bins=100, alpha=0.7, color='blue', edgecolor='black', zorder=2)
    ax.set_title('Baseline Distribution')
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Counts')
    ax.grid(True, zorder=0)

    # Add overflow text for baselines
    ax.text(
        0.95, 0.9,  # Position (relative to axes, 95% right, 90% up)
        f'Overflows: {n_overflows_baseline}',
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=12, color='red', ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    # Plot for division
    ax = axes[2]
    n_overflows_div = np.sum(division <= 0)  # Count of overflows (invalid or 0 values)
    ax.hist(division, bins=100, alpha=0.7, color='blue', edgecolor='black', zorder=2)
    ax.set_title('Division of Baseline/Amplitude')
    ax.set_xlabel('Division')
    ax.set_ylabel('Counts')
    ax.grid(True, zorder=0)

    # Add overflow text for division
    ax.text(
        0.95, 0.9,  # Position (relative to axes, 95% right, 90% up)
        f'Overflows: {n_overflows_div}',
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=12, color='red', ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    # Adjust layout to avoid overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to avoid overlap
    plt.show()


def fit_time_diffs(time_diffs, n_bins=100, min_events=100):
    time_diffs = np.array(time_diffs)
    time_diffs = time_diffs[~np.isnan(time_diffs)]

    meases = [Measure(np.nan, np.nan) for _ in range(3)]

    n_events = time_diffs.size
    if n_events < min_events:
        return meases

    hist, bin_edges = np.histogram(time_diffs, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist_err = np.where(hist > 0, np.sqrt(hist), 1)  # Assume not negative!

    try:
        p0 = [np.max(hist), np.mean(time_diffs), np.std(time_diffs)]
        # popt, pcov = cf(gaus, bin_centers, hist, p0=p0)
        popt, pcov = cf(gaus, bin_centers, hist, p0=p0, sigma=hist_err, absolute_sigma=True)
        popt[2] = abs(popt[2])  # Ensure sigma is positive
        perr = np.sqrt(np.diag(pcov))
        meases = [Measure(val, err) for val, err in zip(popt, perr)]
        return meases
    except RuntimeError:
        return meases


def update_pad_centers_csv(csv_path, run_number, pool_number, channel,
                           x_center, x_center_err, y_center, y_center_err):
    # Create a DataFrame from the new data
    new_row = {
        "run_number": run_number,
        "pool_number": pool_number,
        "channel_number": channel,
        "x_center": x_center,
        "x_center_err": x_center_err,
        "y_center": y_center,
        "y_center_err": y_center_err
    }

    # If the CSV exists, load and update it
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Check if row with the same identifiers exists
        mask = (
            (df["run_number"] == run_number) &
            (df["pool_number"] == pool_number) &
            (df["channel_number"] == channel)
        )

        if mask.any():
            df.loc[mask, ["x_center", "x_center_err", "y_center", "y_center_err"]] = [
                x_center, x_center_err, y_center, y_center_err
            ]
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create new DataFrame if CSV does not exist
        df = pd.DataFrame([new_row])

    # Save the updated DataFrame
    df.to_csv(csv_path, index=False)


def gaus(x, a, mu, sigma):
    return a * np.exp(-(x-mu)**2/(2*sigma**2))

# print('END OF SCRIPT')