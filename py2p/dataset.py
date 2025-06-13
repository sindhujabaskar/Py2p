"""
Defines ExperimentData for integrated BIDS data management.
"""
import pandas as pd
import numpy as np
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
