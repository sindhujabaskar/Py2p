import numpy as np

"""
Filters the ROIs and neuropil fluorescence based on the cell identifier.
    
Parameters:
suite2p_data_output (dict): Dictionary containing Suite2p output data.
    
Returns:
tuple: Filtered ROIs and neuropil fluorescence.
"""

def filter_rois(suite2p_data_output):
    filtered_roi = np.array(suite2p_data_output['roi_fluorescence'][suite2p_data_output['cell_identifier'][:, 0].astype(bool)])
    print("filter_roi shape:", filtered_roi.shape)
    return filtered_roi

def filter_neuropil(suite2p_data_output):
    filtered_neuropil = np.array(suite2p_data_output['neuropil_fluorescence'][suite2p_data_output['cell_identifier'][:, 0].astype(bool)])
    print("filter_pil shape:", filtered_neuropil.shape)   
    return filtered_neuropil



