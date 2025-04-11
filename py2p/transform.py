import numpy as np

def filter_rois(suite2p_data_output):
    """
    Filters the ROIs and neuropil fluorescence based on the cell identifier.
    
    Parameters:
    suite2p_data_output (dict): Dictionary containing Suite2p output data.
    
    Returns:
    tuple: Filtered ROIs and neuropil fluorescence.
    """
    # Assign bool T/F based on confidence index
    true_cells_only = suite2p_data_output['cell_identifier'][:, 0].astype(bool)
    
    # Filter ROIs and neuropil based on bool value
    filtered_roi = np.array(suite2p_data_output['roi_fluorescence'][true_cells_only])
    filtered_neuropil = np.array(suite2p_data_output['neuropil_fluorescence'][true_cells_only])
    
    return true_cells_only, filtered_roi, filtered_neuropil




