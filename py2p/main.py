
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

#  Load paths into the data structure for quick review
# path_loaders = {
#     "psychopy" : load.file_path,
#     "beh": load.file_path,
#     "roi_fluorescence": load.file_path, 
#     "neuropil_fluorescence": load.file_path,
#     "cell_identifier": load.file_path,
#     "pupil": load.file_path
# }
# data.load(path_loaders)

#  Modality-specific data loaders into multiindex dataframe

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
database['transform','interpolated'] = database.apply(lambda row: scipy.interpolate.CubicSpline(np.linspace(0,(row['filter','roi_fluorescence'].shape[1] - 81) / 9.865,
    row['filter','roi_fluorescence'].shape[1]), row['filter','roi_fluorescence'],axis=1)(np.linspace(0,(row['filter','roi_fluorescence'].shape[1] - 81) / 10,
    row['filter','roi_fluorescence'].shape[1]) ),axis=1)

# calculate percentile along each roi's interpolated fluorescence
percentile = 3 
database['calculate','interp_percentile'] = database['transform','interpolated'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) 

#calculate the dF/F values after interpolation
database['calculate','interp_deltaf_f'] = database['transform','interpolated'].combine(database['calculate','interp_percentile'], 
    lambda raw, baseline: (raw - baseline) / baseline)

#create a time index for the dF/F values
from py2p.transform import append_time_index, trials
database['transform','time_vector'] = database['transform','interpolated'].apply(append_time_index)

#creates a new column with the relevant psychopy timestamps for trials
database['transform','trial_index'] = database.apply(lambda row: pd.DataFrame({'trial_num': row['raw','psychopy']['trials.thisN'][1:],'display_gratings_started': 
    row['raw','psychopy']['display_gratings.started'][1:], 'display_gratings_stopped' : row['raw','psychopy']['display_gratings.stopped'][1:]}),axis=1)

#subtracts the first timestamp from all timestamps to create a data frame of start stop times per grating 
database['transform','trial_times'] = database['transform','trial_index'].apply(
    lambda df: pd.DataFrame(list(
        zip(
            (df['display_gratings_started'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int),
            (df['display_gratings_stopped'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int)
        )
    )
, columns=['start', 'stop']))


# use trial tuples to restructure the dF/F data into trials
# database['transform','deltaf_f_trials'] = database.apply(trials, axis=1)

# %% PUPIL DATA PROCESSING
from py2p.calculate import analyze_pupil_data
database['analysis','pupil_diameter_mm'] = database['raw','pupil'].apply(lambda x: analyze_pupil_data(x))


# %% LOCOMOTION DATA PROCESSING


# %% NEW FUNCTIONS
def trials(row):
    deltaf_f = np.array(row['calculate', 'interp_deltaf_f'])         # shape: (n_rois, n_timepoints)
    timestamps = np.array(row['transform', 'time_vector'])[-1]     # shape: (n_timepoints,)
    trial_windows = row['transform', 'trial_tuple']                # list of (start, stop)

    all_data = []  # Will store dictionaries for each trial × ROI × time point

    for trial_idx, (start, stop) in enumerate(trial_windows):
        # Create a mask for the time window
        mask = (timestamps >= start) & (timestamps < stop)
        if not np.any(mask):
            continue  # Skip if no data in this trial window

        time_slice = timestamps[mask]                   # (t,)
        trial_data = deltaf_f[:, mask]                  # (n_rois, t)

        # Loop over each ROI
        for roi_idx in range(trial_data.shape[0]):
            for time_idx, time_val in enumerate(time_slice):
                all_data.append({
                    'trial': trial_idx,
                    'roi': roi_idx,
                    'time': time_val,
                    'dff': trial_data[roi_idx, time_idx]
                })

    return pd.DataFrame(all_data)

database[('transform', 'deltaf_f_trials')] = database.apply(trials, axis=1)

grouped = database['transform','deltaf_f_trials']['sub-SB03','ses-01'].groupby('trial')




# %% What is LAMBDA?????


def my_cool_function(num1, num2):
    return num1 + num2


pd.DataFrame.apply
database['key'].apply(lambda path: np.load(path, allow_pickle = True))

lambda variable_as_argument: print(variable_as_argument) if variable_as_argument > 0 else None

my_cool_function(maths["x"], maths["y"])