"""file_path_finder.py

Module for enumerating BIDS-compliant datasets. AKA compiling file paths for all data sets in a given directory.

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
    "func": ["*.npy"],
    "pupil": ["*_.full.pickle"]
}

def find_files(root: Path, modality: str) -> List[Path]:
    """
    Identify file paths for a specified modality.
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
        file_list.extend(root.glob(f"data/**/*{pattern}", recursive=True))
    return sorted(file_list)