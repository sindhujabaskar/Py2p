
DATA_DIR = r"E:\sbaskar\2408_SU24_F31"

SUITE2P_FILE_PATTERNS = {
    'roi_fluorescence': '*F.npy', # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'neuropil_fluorescence': '*Fneu.npy', # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'cell_identifier' : '*iscell.npy', # np.ndarray(137, 1) [rois, boolean_label] 
    # 'intermediate_outputs' : '*ops.npy', 
    # 'roi_traces' : '*spks.npy',
    # 'roi_statistics' : '*stat.npy'  
}
