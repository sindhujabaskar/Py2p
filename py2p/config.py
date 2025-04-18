
DATA_DIR = r"C:\dev\su24_f31_data\func\SB03\tiff\sb03_high\suite2p\plane0"

GLOBAL_FILE_PATTERNS = {
    'roi_fluorescence': '*F.npy', # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'neuropil_fluorescence': '*Fneu.npy', # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'cell_identifier' : '*iscell.npy', # np.ndarray(137, 1) [rois, boolean_label] 
    # 'intermediate_outputs' : '*ops.npy', 
    # 'roi_traces' : '*spks.npy',
    # 'roi_statistics' : '*stat.npy'  
}
