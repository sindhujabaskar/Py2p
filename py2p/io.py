from pathlib import Path

import numpy as np

from config import GLOBAL_FILE_PATTERNS

def load_suite2p_outputs(directory_path: Path): 
    """ loads all Suite2p output files from the specified directory into a pickled dictionary """

    loaded_data_files = {}
    pathlist = Path(directory_path)
    for key, value in GLOBAL_FILE_PATTERNS.items():
        suite2p_files = list(pathlist.glob(value))
        for file in suite2p_files:
            loaded_data_files[key] = np.load(file, allow_pickle = True)
    return loaded_data_files
