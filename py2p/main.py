# %%
from py2p.io import load_suite2p_outputs, create_roi_dataframe
from py2p.transform import filter_data_by_boolean, interpolate_roi, smooth_dff, active_rois
from py2p.process import calculate_baseline, calculate_dff
from py2p.plot import plot_onething
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
from pathlib import Path

# %% Load imports
from pathlib import Path
import numpy as np
import pandas as pd
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
import py2p.load as load

#  Load in ExperimentData class
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
# Takes a few seconds to run

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
# 
# Process the data
from py2p.transform import filter_cells
import py2p.process as process
# database.get(['roi_fluorescence', 'neuropil_fluorescence', 'cell_identifier'])

#%% STAYING SANE (?)
# Filter the data by boolean 'is_cell'
database['roi_fluorescence'] = database.apply(lambda row: row['roi_fluorescence'][row['cell_identifier'][:, 0].astype(bool)], axis=1)
database['neuropil_fluorescence'] = database.apply(lambda row: row['neuropil_fluorescence'][row['cell_identifier'][:, 0].astype(bool)], axis=1)
database['cell_identifier'] = database.apply(lambda row: row['cell_identifier'][row['cell_identifier'][:, 0].astype(bool)], axis=1)

# calculate the specified percentile along each roi's raw fluorescence 
percentile = 3 
database['baseline_percentile'] = database['roi_fluorescence'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) 

#calculate the dF/F values for each ROI based on the raw fluorescence and baseline fluorescence
database['deltaf_f'] = database['roi_fluorescence'].combine(database['baseline_percentile'], 
                                 lambda raw, baseline: (raw - baseline) / baseline)

# %%

root = Path(DATA_DIR)
data = ExperimentData(root)

# data = load_suite2p_outputs(DATA_DIR) #loading data into a dictionary

# df = create_roi_dataframe(data) #create a multiindex dataframe from the dictionary

filtered_data = filter_data_by_boolean(df) #filter fluorescence data by boolean 'is_cell

baseline_fluorescence = calculate_baseline(filtered_data['roi_fluorescence'], percentile = 3)
print("baseline_fluorescence shape:", baseline_fluorescence.shape)

roi_dff = calculate_dff(filtered_data['roi_fluorescence'], baseline_fluorescence)
print("roi_dff shape:", roi_dff.shape)


plot_onething(roi_dff.loc[3][:1000])
# plot_twothings(interpolated_dff, smoothed_dff)






#%%
interpolated_data = interpolate_roi(filtered_data, offset_frames=81, original_rate=9.865, target_rate=10)
print("interpolated_dff shape:", interpolated_data.shape)

interpolated_dff = calculate_dff(interpolated_data, baseline_fluorescence)

smoothed_dff = smooth_dff(interpolated_dff, smoothing_kernel=3)
print("smoothed_dff shape:", smoothed_dff.shape)

active_rois_only = active_rois(roi_dff)
print("active_rois_only:", active_rois_only)




# %% What is LAMBDA?????


def my_cool_function(num1, num2):
    return num1 + num2


pd.DataFrame.apply
database['key'].apply(lambda path: np.load(path, allow_pickle = True))

lambda variable_as_argument: print(variable_as_argument) if variable_as_argument > 0 else None

my_cool_function(maths["x"], maths["y"])