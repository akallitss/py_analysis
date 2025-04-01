import pandas as pd
import uproot
import numpy as np
import matplotlib.pyplot as plt

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


def get_df_branches(tree, branches, step_size='10 MB'):
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
        if min_idx != 0:
            print(f'hit_x: {hit_x}, hit_y: {hit_y}, distances: {distances}, min_idx: {min_idx}')
        return int(min_idx)

    # Apply function row-wise
    closest_hits = df.apply(lambda row: get_closest_hit(row[f'hitX_{channel}'], row[f'hitY_{channel}']), axis=1)

    # Split results into separate columns
    # df[['closest_hit_index']] = pd.DataFrame(closest_hits.tolist(), index=df.index)
    df['closest_hit_index'] = closest_hits

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


def poly_even_fit(x, x0, a, b, c):
    return a + b * (x - x0)**2 + c * (x - x0)**4


def get_pad_center(df, channel, plot=False, bin_width=0.5, min_tracks_per_2d_bin=20, min_avg_charge_per_2d_bin=4,
                   charge_col_name='totcharge_filtered', charge_cut_low=0, charge_cut_high=100):
    """
    Estimate pad center from average signal charge distribution
    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        channel (string): Channel number (e.g. 'C1', 'C2', 'C3', 'C4')
        plot (bool): Whether to plot the results
        bin_width (float): [mm] Width of the 2D histogram bins. Same in both x and y.
        min_tracks_per_2d_bin (int): Minimum number of tracks per bin
        min_avg_charge_per_2d_bin (float): [pC] Minimum average charge per bin
        charge_col_name (string): Name of the charge column
        charge_cut_low (float): [pC] Lower charge cut. If None, no cut is applied.
        charge_cut_high (float): [pC] Upper charge cut. If None, no cut is applied.
    Returns:
        tuple(Measure, Measure): Tuple of Measure objects representing the x and y pad center estimates
        with their uncertainties

    """
    charge_col = f'peakparam_{channel}/peakparam_{channel}.{charge_col_name}'

    basic_charge_cut = pd.Series(True, index=df.index)
    if charge_cut_low is not None:
        basic_charge_cut &= df[charge_col] > charge_cut_low  # Use `&` for element-wise AND
    if charge_cut_high is not None:
        basic_charge_cut &= df[charge_col] < charge_cut_high

    basic_charge_cut = basic_charge_cut.to_numpy()

    xs = df[f'hitX_{channel}'][basic_charge_cut]
    ys = df[f'hitY_{channel}'][basic_charge_cut]
    charges = df[charge_col][basic_charge_cut]

    parameter_string = (f'Bin Width: {bin_width} mm\nMinimum Tracks per Bin: {min_tracks_per_2d_bin}'
                        f'\nMinimum Average Charge per Bin: {min_avg_charge_per_2d_bin} pC')

    # Define histogram bins
    bin_x_min, bin_x_max = np.min(xs), np.max(xs)
    bin_y_min, bin_y_max = np.min(ys), np.max(ys)
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

    avg_x_charge_err = np.abs(avg_x_charge) / np.sqrt(np.where(sum_x_tracks > 0, sum_x_tracks, 1))
    avg_y_charge_err = np.abs(avg_y_charge) / np.sqrt(np.where(sum_y_tracks > 0, sum_y_tracks, 1))

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

        axs[0].annotate(
            parameter_string,
            xy=(0.5, 0.2), xycoords='axes fraction',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
        )

        axs[0].axvline(popt_x_charge[0], color='salmon', ls='-', alpha=0.5)
        axs[1].axvline(popt_y_charge[0], color='salmon', ls='-', alpha=0.5)

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
        axs[0].scatter(popt_x_charge[0], popt_y_charge[0], marker='x', c='k')
        fig.colorbar(im1, ax=axs[0])

        # Plot xytrksW
        # im2 = axs[1].imshow(masked_xy_trksW.T, origin="lower", extent=[0, 50, 0, 50], cmap='jet', aspect="auto", vmax=np.max(masked_xy_trksW) / 1.8)
        im2 = axs[1].imshow(masked_xy_trksW.T, origin="lower", extent=[bin_x_min, bin_x_max, bin_y_min, bin_y_max],
                            cmap='jet', aspect="auto")
        axs[1].set_title("Average Charge per Track")
        axs[1].set_xlabel("x [mm]")
        axs[1].set_ylabel("y [mm]")
        axs[1].scatter(popt_x_charge[0], popt_y_charge[0], marker='x', c='k')
        axs[1].annotate(
            f'Pad Center: ({meas_x_charge[0]}, {meas_y_charge[0]})',
            xy=(0.05, 0.95), xycoords='axes fraction',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightyellow')
        )
        axs[1].set_xlim(x_min - extend_range * x_range, x_max + extend_range * x_range)
        axs[1].set_ylim(y_min - extend_range * y_range, y_max + extend_range * y_range)
        fig.colorbar(im2, ax=axs[1])

    return meas_x_charge[0], meas_y_charge[0]


def get_circle_scan(df, xy_pairs, channel, time_col='time_diff', ns_to_ps=False, radius=1, time_diff_lims=None, min_events=100, plot=False):
    time_diffs = df[f'{time_col}_{channel}']
    if ns_to_ps:
        time_diffs = time_diffs * 1000
    if time_diff_lims is not None:
        if ns_to_ps:
            time_diff_lims = np.array(time_diff_lims) * 1000
        time_diffs[(time_diffs < time_diff_lims[0]) | (time_diffs > time_diff_lims[1])] = np.nan
    xs = df[f'hitX_{channel}']
    ys = df[f'hitY_{channel}']

    resolutions, means, events = [], [], []
    for x, y in xy_pairs:
        print(f'Circle Scan: ({x}, {y})')
        rs = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
        mask = rs < radius
        time_diffs_bin = time_diffs[mask]
        time_diffs_bin = np.array(time_diffs_bin[~np.isnan(time_diffs_bin)])
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
            ax.set_xlabel('Time Difference')
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
                  cmap="jet_r")
    plt.colorbar(c, label="Timing Resolution [ps]")
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Timing Resolution Heatmap{radius_str}")

    # Create the contour plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x_mesh, y_mesh, scan_resolutions_2d, levels=50, cmap="jet_r")

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("Timing Resolution [ps]")

    # Labels and title
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Timing Resolution Contour Plot{radius_str}")

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(scan_means_2d, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto",
                  cmap="jet_r")
    plt.colorbar(c, label="Mean Time Difference [ps]")
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Time Difference Heatmap{radius_str}")

    # Create the contour plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x_mesh, y_mesh, scan_means_2d, levels=50, cmap="jet_r")

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("Time Difference [ps]")

    # Labels and title
    ax.set_xlabel("X Position [mm]")
    ax.set_ylabel("Y Position [mm]")
    ax.set_title(f"Time Difference Contour Plot{radius_str}")

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


def fit_time_diffs(time_diffs, n_bins=100, min_events=100):
    time_diffs = np.array(time_diffs)
    time_diffs = time_diffs[~np.isnan(time_diffs)]

    meases = [Measure(np.nan, np.nan) for _ in range(3)]

    n_events = time_diffs.size
    if n_events < min_events:
        return meases

    hist, bin_edges = np.histogram(time_diffs, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        p0 = [np.max(hist), np.mean(time_diffs), np.std(time_diffs)]
        popt, pcov = cf(gaus, bin_centers, hist, p0=p0)
        popt[2] = abs(popt[2])  # Ensure sigma is positive
        perr = np.sqrt(np.diag(pcov))
        meases = [Measure(val, err) for val, err in zip(popt, perr)]
        return meases
    except RuntimeError:
        return meases

def gaus(x, a, mu, sigma):
    return a * np.exp(-(x-mu)**2/(2*sigma**2))

# print('END OF SCRIPT')