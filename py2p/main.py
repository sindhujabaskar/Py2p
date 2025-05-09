#%%
from py2p.io import load_suite2p_outputs, create_roi_dataframe
from py2p.transform import filter_data_by_boolean, interpolate_roi, smooth_dff, active_rois
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.plot import plot_onething
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData
from pathlib import Path

#%%
from py2p.config import DATA_DIR
from py2p.dataset import ExperimentData

from pathlib import Path
import numpy as np
import pandas as pd

root = Path(DATA_DIR)
data = ExperimentData(root)

def load_modality(path):
    return path

loaders = {
    "psychopy" : load_modality,
    "beh": load_modality,
    "roi_fluorescence": load_modality, 
    "neuropil_fluorescence": load_modality,
    "cell_identifier": load_modality,
    "pupil": load_modality
}

data.load(loaders)
database = data.df
#%%
from py2p.transform import filter_cells_iter, create_dataframe, array_loader

roi_f = array_loader(database, 'roi_fluorescence')
neuropil_f = array_loader(database, 'neuropil_fluorescence')
is_cell = array_loader(database, 'cell_identifier')

df = create_dataframe(roi_f, neuropil_f, is_cell)
filtered_df = filter_cells_iter(df)

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