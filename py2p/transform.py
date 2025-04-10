from py2p.io import load_suite2p_outputs




# assign bool T/F based on confidence index
true_cells_only = suite2p_data_output['cell_identifier'][:,0].astype(bool)
print(true_cells_only)

# filter ROIs and neuropil based on bool value
filtered_roi = np.array(suite2p_data_output['roi_fluorescence'][true_cells_only])
filtered_neuropil = np.array(suite2p_data_output['neuropil_fluorescence'][true_cells_only])