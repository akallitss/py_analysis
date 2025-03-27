import pandas as pd
import uproot
import numpy as np
import matplotlib.pyplot as plt


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
    df[f'peakparam_{channel}/peakparam_{channel}.hitX[200]_np'] = df[f'peakparam_{channel}/peakparam_{channel}.hitX[200]'].apply(np.array)
    df[f'peakparam_{channel}/peakparam_{channel}.hitY[200]_np'] = df[f'peakparam_{channel}/peakparam_{channel}.hitY[200]'].apply(np.array)

    # Function to find the closest hit per row
    def get_closest_hit(hit_x, hit_y):
        if len(hit_x) == 0:  # Handle empty arrays
            return np.nan
        distances = (hit_x - det_center[0]) ** 2 + (hit_y - det_center[1]) ** 2  # Squared Euclidean distance
        min_idx = np.argmin(distances)  # Index of the closest hit
        if min_idx != 0:
            print(f'hit_x: {hit_x}, hit_y: {hit_y}, distances: {distances}, min_idx: {min_idx}')
        return min_idx

    # Apply function row-wise
    closest_hits = df.apply(lambda row: get_closest_hit(row[f'peakparam_{channel}/peakparam_{channel}.hitX[200]_np'],
                                                        row[f'peakparam_{channel}/peakparam_{channel}.hitY[200]_np']), axis=1)

    # Split results into separate columns
    # df[['closest_hit_index']] = pd.DataFrame(closest_hits.tolist(), index=df.index)
    df['closest_hit_index'] = closest_hits

def get_center_all_tracks(df_in, channel):
    "Extract the center of distribution for all tracks"
    # Convert list-like column entries to NumPy arrays
    df = df_in.copy()
    hit_xs = df[f'peakparam_{channel}/peakparam_{channel}.hitX[200]'].apply(np.array)
    hit_ys = df[f'peakparam_{channel}/peakparam_{channel}.hitY[200]'].apply(np.array)

    # Apply filtering using NumPy vectorized operations
    filtered_xs = hit_xs.apply(lambda x: x[x > -1111])
    filtered_ys = hit_ys.apply(lambda y: y[y > -1111])

    # Concatenate all valid hit coordinates into a single NumPy array for efficiency
    good_hit_xs = np.concatenate(filtered_xs.values)
    good_hit_ys = np.concatenate(filtered_ys.values)

    # Get median x and y
    return np.median(good_hit_xs), np.median(good_hit_ys)


# print('END OF SCRIPT')