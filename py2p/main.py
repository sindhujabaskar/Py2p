# %%  LOAD DATA
# Load imports
from pathlib import Path
import numpy as np
import pandas as pd
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
import py2p.load as load
import matplotlib.pyplot as plt

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

#creates a new column with the relevant psychopy timestamps for trials
database['toolkit','psychopy_trials'] = database.apply(lambda row: pd.DataFrame({'trial_num': row['raw','psychopy']['trials.thisN'][1:],'display_gratings_started': 
    row['raw','psychopy']['display_gratings.started'][1:], 'display_gratings_stopped' : row['raw','psychopy']['display_gratings.stopped'][1:]}),axis=1)

#subtracts the first timestamp from all timestamps to create a data frame of start stop times per grating 
database['toolkit','trial_index'] = database['toolkit','psychopy_trials'].apply(
    lambda df: pd.DataFrame(list(
        zip(
            (df['display_gratings_started'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int),
            (df['display_gratings_stopped'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int))
    ), columns=['start', 'stop']))

# create a dataframe with trial #, block #, trial orientation, time(s), and smoothed dF/F value
from py2p.transform import trials
database[('toolkit','trials')] = database.apply(trials, axis=1)

#calculate the mean dF/F across all ROIs for each session
database['analysis','mean_deltaf_f'] = database['calculate','interp_deltaf_f'].apply(lambda x: np.mean(x, axis=0))



# %% PUPIL DATA PROCESSING
from py2p.process import analyze_pupil_data
database['analysis','pupil_diameter_mm'] = database['raw','pupil'].apply(lambda x: analyze_pupil_data(x))


# %% LOCOMOTION DATA PROCESSING


# %% PLOT THIS DATA
from py2p.plot import plot_trial, plot_block

# Plot a single trial 
# plot_trial(df = database, subject= 'sub-SB03', session= 'ses-01', trial_idx=0) 

# Plot a trial block
plot_block(df = database, subject= 'sub-SB03', session= 'ses-01', block_idx=1)

# %% NEW FUNCTIONS





# %% What is LAMBDA?????


def my_cool_function(num1, num2):
    return num1 + num2


pd.DataFrame.apply
database['key'].apply(lambda path: np.load(path, allow_pickle = True))

lambda variable_as_argument: print(variable_as_argument) if variable_as_argument > 0 else None

my_cool_function(maths["x"], maths["y"])