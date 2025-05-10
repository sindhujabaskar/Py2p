#%%
from py2p.io import load_suite2p_outputs, create_roi_dataframe
from py2p.transform import filter_data_by_boolean, interpolate_roi, smooth_dff, active_rois
from py2p.process import calculate_baseline, calculate_dff
from py2p.plot import plot_onething
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
from pathlib import Path

#%%
# Load the data into a multiindex dataframe

from pathlib import Path
import numpy as np
import pandas as pd
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
from py2p.load import load_path, array_loader, csv_loader

root = Path(DATA_DIR)
data = ExperimentData(root)
#%%
path_loaders = {
    "psychopy" : load_path,
    "beh": load_path,
    "roi_fluorescence": load_path, 
    "neuropil_fluorescence": load_path,
    "cell_identifier": load_path,
    "pupil": load_path
}

data.load(path_loaders)
database = data.df
#%%
load_modality = {
    "psychopy" : csv_loader,
    "beh": csv_loader,
    "roi_fluorescence": array_loader, 
    "neuropil_fluorescence": array_loader,
    "cell_identifier": array_loader,
    "pupil": csv_loader
}

data.load(load_modality)
database = data.df
    
#%%
# Process the data

from py2p.transform import filter_cells
from py2p.load import create_dataframe, array_loader
from py2p.process import calculate_baseline

# index just suite2p outputs from database
roi_f = array_loader(database, 'roi_fluorescence')
neuropil_f = array_loader(database, 'neuropil_fluorescence')
is_cell = array_loader(database, 'cell_identifier')

# concatenate suite2p outputs into a single multiindex dataframe
df = create_dataframe(roi_f, neuropil_f, is_cell)

#
filtered_df = filter_cells(df)
baseline = calculate_baseline(filtered_df, 3)


#%%

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