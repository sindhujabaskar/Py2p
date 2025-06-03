from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from py2p.config import MODALITY_PATTERNS

# Loader functions for finding filepaths and loading the functions for each modality
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

def file_path(path):
    return path

def suite2p_npy(path) -> pd.Series:
    return np.load(path, allow_pickle=True)

def beh_csv(path) -> pd.Series:
    return pd.read_csv(path)

def deeplabcut_pickle(filepath: Path) -> pd.DataFrame:
    """
    Custom loader for DeepLabCut pickle output.

    Reads a pickled dict where keys are frame identifiers and values are dicts
    containing:
      - 'coordinates': array-like of shape (n_landmarks, 2) for each frame
      - 'confidence': array-like of length n_landmarks or single float per frame

    This function:
      1. Loads the pickle file.
      2. Skips the first entry (assumed metadata).
      3. Constructs a DataFrame indexed by frame keys (str).
      4. Provides exactly two columns:
         - 'coordinates': list/array of (x, y) coordinate pairs for that frame.
         - 'confidence' : list/array of confidence values (or single float) for that frame.

    Returned DataFrame shape: (F, 2)
      where F = number of actual frames (total keys minus metadata entry).

    Parameters
    ----------
    filepath : str
        Path to the pickled DeepLabCut output dict.

    Returns
    -------
    pd.DataFrame
        Index: frame identifiers (str), name='frame'
        Columns:
          - coordinates : object (array-like per row)
          - confidence  : object (array-like or float per row)
    """
    data = pd.read_pickle(filepath)

    # Build dictionaries for coordinates and confidence
    coordinates_dict = {}
    confidence_dict = {}
    for frame_key, frame_data in data.items():
        coordinates_dict[frame_key] = frame_data.get('coordinates')
        confidence_dict[frame_key] = frame_data.get('confidence')

    # Skip the first (metadata) entry by slicing off index 0
    coords_series = pd.Series(coordinates_dict).iloc[1:]
    conf_series  = pd.Series(confidence_dict).iloc[1:]

    # Create the DataFrame
    df = pd.DataFrame({
        'coordinates': coords_series,
        'confidence': conf_series,
    })
    df.index.name = 'frame'
    # Drop any leftover metadata column
    df = df.drop(columns=['metadata'], errors='ignore')

    # Debug statement
    # print(f"[load_deeplabcut_pickle][DEBUG] Loaded DeepLabCut pickle from: {filepath}")
    return df

