"""
Defines ExperimentData for integrated BIDS data management.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Callable, Dict, List
from py2p.load import find_files

def make_multiindex(paths: List[Path]) -> pd.MultiIndex:
    """
    Construct a MultiIndex of (Subject, Session) from BIDS file paths.
    """
    indices = []
    for p in paths:
        parts = list(p.parts)
        subj = next(part for part in parts if part.startswith("sub-"))  #if the first item in the path does not start with sub- then StopIterationError will be raised
        sess = next(part for part in parts if part.startswith("ses-"))
        indices.append((subj, sess))
    return pd.MultiIndex.from_tuples(indices, names=["Subject", "Session"])

class ExperimentData:
    """
    High-level interface for aggregating BIDS datasets into a DataFrame.
    """
    def __init__(self, root: Path): #initialize the root directory path
        self.root = root
        self._df: pd.DataFrame = pd.DataFrame()

    def load(self, loaders: Dict[str, Callable[[Path], object]]) -> pd.DataFrame: 
        """
        Batch-apply modality-specific loaders and assemble a unified DataFrame.
        Parameters
        ----------
        loaders : dict
            Mapping from modality keys to loader callables.
        Returns
        -------
        pd.DataFrame
            MultiIndexed DataFrame where each column corresponds to a modality.
        """
        series_list = []
        for modality, loader in loaders.items():
            file_paths = find_files(self.root, modality) # find all files in the root directory that match the modality
            idx = make_multiindex(file_paths) # create a MultiIndex from the file paths
            series = pd.Series(file_paths, index=idx, name="filepath")
            loaded = series.map(loader)
            loaded.name = modality
            series_list.append(loaded)
        self._df = pd.concat(series_list, axis=1) # combine all modality data into a single DataFrame
        return self._df
    
    def save_hdf5(self, df, path, key="nested_data", compression="blosc"):
        """
        Export nested DataFrame to HDF5 format.

        Args:
            df (pd.DataFrame):  The combined nested DataFrame to save
            path (str):         Output path to HDF5 file
            key (str):          Dataset name within HDF5 file
            compression (str):  Compression backend (ignored in fixed format but kept for future)
        """
        self._log(f"Saving to HDF5 (fixed format): {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with pd.HDFStore(path, mode='w') as store:
            store.put(key, df, format='fixed')
            self._log(f"Saved nested DataFrame to HDF5 under key '{key}'")

    def load_hdf5(self, path, key="nested_data"):
        """
        Load nested DataFrame from HDF5 format.

        Args:
            path (str): Path to the HDF5 file
            key (str): Dataset name within HDF5 file

        Returns:
            pd.DataFrame: The loaded nested DataFrame
        """
        self._log(f"Loading from HDF5: {path}")
        with pd.HDFStore(path, mode='r') as store:
            df = store.get(key)
            self._log(f"Loaded nested DataFrame from HDF5 under key '{key}'")
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Return the DataFrame containing all loaded experimental data across subjects and sessions."""
        return self._df
    
    @property
    def raw_fluorescence(self) -> pd.Series:
        """Return the raw fluorescence data."""
        return self.df['roi_fluorescence']
    
    @property
    def neuropil_fluorescence(self) -> pd.Series:
        """Return the neuropil fluorescence data."""
        return self.df['neuropil_fluorescence']
    
    @property
    def cell_identifier(self) -> pd.Series:
        """Return the cell identifier data."""
        return self.df['cell_identifier']
    
    @property
    def beh(self) -> pd.Series:
        """Return the pupil data."""
        return self.df['analyze','pupil_diameter_mm']
    
    @property
    def psychopy(self) -> pd.Series:
        """Return the psychopy data."""
        return self.df['psychopy']
