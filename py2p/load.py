from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from py2p.config import MODALITY_PATTERNS

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

def load_path(path):
    return path

def array_loader(database: pd.DataFrame, column_name: str = 'modality') -> pd.Series:
    return database[column_name].apply(lambda path: np.load(path, allow_pickle=True))

def csv_loader(database: pd.DataFrame, column_name: str = 'modality') -> pd.Series:
    return database[column_name].apply(lambda path: pd.read_csv(path))

def create_dataframe(roi_f, neuropil_f, is_cell):
    all_f = {'roi_f': roi_f, 'neuropil_f': neuropil_f, 'is_cell': is_cell}
    df = pd.concat(all_f, axis=1)
    return df
