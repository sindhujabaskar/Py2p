
# %%  
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

# %% ROI FLUORESCENCE PROCESSING
# Filter the data by boolean 'is_cell'
database['filter','roi_fluorescence'] = database.apply(lambda row: row['roi_fluorescence'][row['cell_identifier'][:, 0].astype(bool)], axis=1)
database['filter','neuropil_fluorescence'] = database.apply(lambda row: row['neuropil_fluorescence'][row['cell_identifier'][:, 0].astype(bool)], axis=1)

# calculate the specified percentile along each roi's raw fluorescence 
percentile = 3 
database['process','baseline_percentile'] = database['filter','roi_fluorescence'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) 

#calculate the dF/F values for each ROI based on the raw fluorescence and baseline fluorescence
database['process','deltaf_f'] = database['filter','roi_fluorescence'].combine(database['process','baseline_percentile'], 
                                 lambda raw, baseline: (raw - baseline) / baseline)

# interpolation from 9.865 Hz to 10 Hz
import scipy.interpolate
database['transform','interpolated'] = database.apply(lambda row: scipy.interpolate.CubicSpline(np.linspace(0,(row['roi_fluorescence'].shape[1] - 81) / 9.865,
    row['roi_fluorescence'].shape[1]), row['roi_fluorescence'],axis=1)(np.linspace(0,(row['roi_fluorescence'].shape[1] - 81) / 10,
    row['roi_fluorescence'].shape[1]) ),axis=1)

#create a time index for the dF/F values
from py2p.transform import append_time_index, trials
database['transform','time_vector'] = database['filter','roi_fluorescence'].apply(append_time_index)

#creates a new column with the relevant psychopy timestamps for trials
database['transform','trial_index'] = database.apply(lambda row: pd.DataFrame({'trial_num': row['psychopy']['trials.thisN'][1:],'display_gratings_started': 
    row['psychopy']['display_gratings.started'][1:], 'display_gratings_stopped' : row['psychopy']['display_gratings.stopped'][1:]}),axis=1)

#subtracts the first timestamp from all timestamps to create a time index starting at 0 and casts into a (start,stop) tuple 
database['transform', 'trial_tuple'] = database['transform','trial_index'].apply(
    lambda df: list(
        zip(
            (df['display_gratings_started'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int),
            (df['display_gratings_stopped'] - df['display_gratings_started'].iloc[0].astype(int)).astype(int)
        )
    )
)

# use trial tuples to restructure the dF/F data into trials
database['transform','deltaf_f_trials'] = database.apply(trials, axis=1)

# %% PUPIL DATA PROCESSING
from py2p.process import analyze_pupil_data
database['analysis','pupil_diameter_mm'] = database['pupil'].apply(lambda x: analyze_pupil_data(x))


# %% LOCOMOTION DATA PROCESSING


# %% NEW FUNCTIONS





# %% What is LAMBDA?????


def my_cool_function(num1, num2):
    return num1 + num2


pd.DataFrame.apply
database['key'].apply(lambda path: np.load(path, allow_pickle = True))

lambda variable_as_argument: print(variable_as_argument) if variable_as_argument > 0 else None

my_cool_function(maths["x"], maths["y"])