# %%  LOAD DATA
# Load imports
from pathlib import Path
import numpy as np
import pandas as pd
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
import py2p.load as load
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#  Load ExperimentData class
root = Path(DATA_DIR)
data = ExperimentData(root)

load_modality = {
    "psychopy" : load.beh_csv,
    "beh": load.beh_csv,
    "roi_fluorescence": load.suite2p_npy, 
    "neuropil_fluorescence": load.suite2p_npy,
    "cell_identifier": load.suite2p_npy,
    "pupil": load.deeplabcut_pickle
}

data.load(load_modality)

database = data.df

# data.save_hdf5(df=database, path=r'C:\dev\2408_SU24_F31.h5')

# Promote all single-level columns to two-level columns
database.columns = pd.MultiIndex.from_tuples(
    [(col if isinstance(col, tuple) else ('raw', col)) for col in database.columns]
)

# %% ROI FLUORESCENCE PROCESSING
# Filter the data by boolean 'is_cell'
database['filter','roi_fluorescence'] = database.apply(lambda row: row['raw','roi_fluorescence'][row['raw','cell_identifier'][:, 0].astype(bool)], axis=1)
database['filter','neuropil_fluorescence'] = database.apply(lambda row: row['raw','neuropil_fluorescence'][row['raw','cell_identifier'][:, 0].astype(bool)], axis=1)

# Subtract neuropil fluorescence from raw fluorescence
database['calculate', 'neuropil_subtract'] = database.apply(lambda row: row['filter','roi_fluorescence'] - (0.3*row['filter','neuropil_fluorescence']), axis =1)

# calculate the specified percentile along each roi's raw fluorescence 
percentile = 3 
database['calculate','baseline_percentile'] = database['filter','roi_fluorescence'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) 

#calculate the dF/F values for each ROI based on the raw fluorescence and baseline fluorescence
database['calculate','deltaf_f'] = database['filter','roi_fluorescence'].combine(database['calculate','baseline_percentile'], 
                                 lambda raw, baseline: (raw - baseline) / baseline)

# interpolation from 9.865 Hz to 10 Hz
import scipy.interpolate
database['calculate','interpolated'] = database.apply(lambda row: scipy.interpolate.CubicSpline(np.linspace(0,(row['filter','roi_fluorescence'].shape[1] - 81) / 9.865,
    row['filter','roi_fluorescence'].shape[1]), row['filter','roi_fluorescence'],axis=1)(np.linspace(0,(row['filter','roi_fluorescence'].shape[1] - 81) / 10,
    row['filter','roi_fluorescence'].shape[1]) ),axis=1)

# calculate percentile along each roi's interpolated fluorescence
percentile = 3 
database['calculate','interp_percentile'] = database['calculate','interpolated'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) 

#calculate the dF/F values after interpolation
database['calculate','interp_deltaf_f'] = database['calculate','interpolated'].combine(database['calculate','interp_percentile'], 
    lambda raw, baseline: (raw - baseline) / baseline)

# calculate the smoothed dF/F values using a moving average with a window size of 3
smoothing_kernel = 3
database[('calculate','smoothed_dff')]= database[('calculate','interp_deltaf_f')].apply(lambda arr: np.array([np.convolve(row, np.ones(smoothing_kernel)/ smoothing_kernel, mode = 'same') for row in arr]))

#create a 10Hz interpolated time vector
database[('toolkit','timestamps')] = (database[('calculate','interpolated')].map(lambda arr: pd.Series(
         np.arange(arr.shape[1]) * 0.1, name='time')))

database[('analysis','peaks_prominence')] = database[('analysis','mean_deltaf_f')].apply(
    lambda arr: find_peaks(arr, prominence=0.3)[0])

database[('analysis','num_peaks_prominence')] = database[('analysis','peaks_prominence')].apply(len)

#creates a new column with the relevant psychopy timestamps for trials
# %%
database['toolkit','psychopy_trials'] = database.apply(lambda row: pd.DataFrame({'trial_num': row['raw','psychopy']['trials.thisN'][1:],'trial_start': row['raw','psychopy']['display_gratings.started'][1:], 'grey_start': 
    row['raw','psychopy']['stim_grayScreen.started'][1:], 'gratings_start': row['raw','psychopy']['stim_grating.started'][1:], 'trial_end' : row['raw','psychopy']['display_gratings.stopped'][1:]}),axis=1)

#subtracts the first timestamp from all timestamps for a zero-start index 
database[('toolkit','trial_index')] = database[('toolkit','psychopy_trials')].apply(
    lambda df: (df[['trial_start','grey_start','gratings_start','trial_end']].subtract(df['trial_start'].iloc[0].astype(int))).astype(int))

# create a dataframe with trial #, block #, trial orientation, time(s), and smoothed dF/F values
from py2p.transform import trials
database[('toolkit','trials')] = database.apply(trials, axis=1)

# calculate the mean dF/F across all ROIs for each session
database['analysis','mean_deltaf_f'] = database['calculate','interp_deltaf_f'].apply(lambda x: np.mean(x, axis=0))

# %% PUPIL DATA PROCESSING
from py2p.process import analyze_pupil_data
database['analysis','pupil_diameter_mm'] = database['raw','pupil'].apply(lambda x: analyze_pupil_data(x))

# create timestamps for the pupil data
database['toolkit','pupil_timestamps'] = database['analysis','pupil_diameter_mm'].apply(lambda arr: np.arange(arr.shape[0])/40.0)

def process_pupil_data(database: pd.DataFrame,
                       downsample_threshold: int = None,
                       downsample_factor: int = 5):
    """
    Downsample pupil diameter timecourses for each (subject, session).

    Adds new columns:
      ('analysis','pupil_timestamps_ds') : downsampled timestamps
      ('analysis','pupil_diameter_ds')  : downsampled pupil values

    Args:
        database            : main database DataFrame
        downsample_threshold: if length exceeds this, downsample
        downsample_factor   : factor by which to downsample
    """
    import numpy as np
    # initialize storage
    ts_ds = {}
    pup_ds = {}
    for (subj, sess), pup in database[('analysis','pupil_diameter_mm')].items():
        # get raw timestamp and pupil arrays
        ts = database[('toolkit','pupil_timestamps')].loc[(subj, sess)]
        ts_arr = np.array(ts)
        pup_arr = np.array(pup)
        # downsample if requested
        if downsample_threshold and len(pup_arr) > downsample_threshold:
            idx = np.arange(0, len(pup_arr), downsample_factor)
            ts2 = ts_arr[idx]
            pup2 = pup_arr[idx]
        else:
            ts2 = ts_arr
            pup2 = pup_arr
        ts_ds[(subj, sess)] = ts2
        pup_ds[(subj, sess)] = pup2
    database[('analysis','pupil_timestamps_ds')] = pd.Series(ts_ds)
    database[('analysis','pupil_diameter_ds')] = pd.Series(pup_ds)
    return database

# %% LOCOMOTION DATA PROCESSING
# Process and filter locomotion (speed) data from ('raw','beh')
from scipy.signal import savgol_filter

def process_locomotion_data(database: pd.DataFrame,
                             method: str = 'savgol',
                             window_length: int = 5,
                             polyorder: int = 2,
                             downsample_threshold: int = None,
                             downsample_factor: int = 5):
    """
    Smooth and optionally downsample locomotion speed for each (subject, session).

    Adds two new columns to the database MultiIndex:
      ('analysis','time_smoothed') and ('analysis','speed_smoothed')

    Args:
        database            : main database DataFrame
        method              : smoothing method ('savgol' supported)
        window_length       : window length for Savitzky-Golay filter (must be >=3 odd)
        polyorder           : polynomial order for filter
        downsample_threshold: if signal length exceeds this, downsample
        downsample_factor   : factor by which to downsample
    """
    import numpy as np
    # initialize storage
    time_smoothed = {}
    speed_smoothed = {}

    # iterate over each session row
    for (subj, sess), beh in database[('raw','beh')].items():
        # assume beh is a DataFrame with 'timestamp' and 'speed'
        ts = beh['timestamp'].values
        sp = beh['speed'].values
        # smoothing
        if method == 'savgol' and len(sp) > 5:
            wl = min(window_length, len(sp))
            if wl % 2 == 0:
                wl -= 1
            if wl >= 3:
                sp_sm = savgol_filter(sp, wl, polyorder)
            else:
                sp_sm = sp
        else:
            sp_sm = sp
        # downsample if requested
        if downsample_threshold and len(sp_sm) > downsample_threshold:
            idx = np.arange(0, len(sp_sm), downsample_factor)
            ts_sm = ts[idx]
            sp_sm = sp_sm[idx]
        else:
            ts_sm = ts
        # store results
        time_smoothed[(subj, sess)] = ts_sm
        speed_smoothed[(subj, sess)] = sp_sm
    # assign back to database
    database[('analysis','time_smoothed')] = pd.Series(time_smoothed)
    database[('analysis','speed_smoothed')] = pd.Series(speed_smoothed)
    return database

# Example usage:
process_locomotion_data(database, method='savgol', window_length=7, polyorder=3,
                                 downsample_threshold=1000, downsample_factor=100)

# pupil event detection


# %% PLOT THIS DATA
from py2p.plot import plot_trial, plot_block, plot_all_rois_tuning_polar, plot_session_overview

# Plot a single trial 
plot_trial(database = database, subject= 'sub-SB03', session= 'ses-01', trial_idx=0) 

# Plot a trial block
plot_block(database = database, subject= 'sub-SB03', session= 'ses-04', block_idx=2)

fig, axes = plot_all_rois_tuning_polar(database, 'sub-SB03', 'ses-04')
fig.savefig('all_rois_tuning_polar.svg', dpi=300)

fig, axes = plot_session_overview(database, subject='sub-SB03', session='ses-04')

# %% NEW FUNCTIONS
from scipy.signal import find_peaks_cwt

database[('analysis','peaks_cwt')] = database[('analysis','mean_deltaf_f')].apply(
    lambda arr: find_peaks_cwt(arr, widths=[50]))


# %% What is LAMBDA?????


def my_cool_function(num1, num2):
    return num1 + num2


pd.DataFrame.apply
database['key'].apply(lambda path: np.load(path, allow_pickle = True))

lambda variable_as_argument: print(variable_as_argument) if variable_as_argument > 0 else None

my_cool_function(maths["x"], maths["y"])