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
from pathlib import Path
from typing import List, Dict

# Mapping of modality keys to glob patterns
MODALITY_PATTERNS: Dict[str, List[str]] = {
    "beh": ["*_wheeldf.csv"],
    'roi_fluorescence': ['*F.npy'], # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'neuropil_fluorescence': ['*Fneu.npy'], # np.ndarray(137, 6000) [rois, fluorescence_by_frame]
    'cell_identifier' : ['*iscell.npy'], # np.ndarray(137, 1) [rois, boolean_label] 
    "pupil": ["*.pickle"]
}

def find_files(root: Path, modality: str) -> List[Path]:
    """
    Recursively find all files matching the given modality
    under `root`, using MODALITY_PATTERNS.
    Parameters
    ----------
    root : Path
        Absolute path to the experiment root directory.
    modality : str
        Key corresponding to a modality in MODALITY_PATTERNS.

    Returns
    -------
    List[Path]
        Sorted list of matching file paths.
    """
    patterns = MODALITY_PATTERNS.get(modality, [])
    file_list: List[Path] = []
    for pattern in patterns:
        file_list.extend(root.rglob(pattern))
    return sorted(file_list)