from typing import Dict, List

"""
Module for enumerating BIDS-compliant datasets.

ExperimentRoot/
├── data/              # Raw acquisition files
│   ├── sub-01/
│   │   ├── ses-01/    # Session-specific subdirectories
│   │   │   ├── beh/   # behavior data
│   │   │   └── func/   # Functional data
│   │   └── ses-02/
│   └── sub-02/
└── processed/         # Processed outputs
    ├── sub-01/
    │   └── ses-01/
│   │   │   ├── dlc_output/   # DeepLabCut pickle file
│   │   │   └── suite2p/   # suite2p outputs
    └── sub-02/
"""

DATA_DIR = r"D:\2408_SU24_F31"

# Mapping of modality keys to glob patterns
MODALITY_PATTERNS: Dict[str, List[str]] = {
    "psychopy" : ['*_.csv'],
    "beh": ["*_wheeldf.csv"],
    'roi_fluorescence': ['*F.npy'], # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'neuropil_fluorescence': ['*Fneu.npy'], # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'cell_identifier' : ['*iscell.npy'], # np.ndarray(137, 1) [rois, boolean_label] 
    "pupil": ["*.pickle"]
}

two_photon_frame_rate = 9.865 # Hz

