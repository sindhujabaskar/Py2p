from pathlib import Path
import numpy as np
import pandas as pd
from py2p.config import SUITE2P_FILE_PATTERNS

def load_suite2p_outputs(directory_path): 
    """ loads all Suite2p output files from the specified directory into a pickled dictionary """

    loaded_data_files = {}
    pathlist = Path(directory_path)
    for key, value in SUITE2P_FILE_PATTERNS.items():
        suite2p_files = list(pathlist.glob(value))
        for file in suite2p_files:
            loaded_data_files[key] = np.load(file, allow_pickle = True)
    return loaded_data_files

# def np_loader(file_path: str) -> np.ndarray: #trying to add index keys to the loader function so that the files can be called via keys later on
#     for key, value in SUITE2P_FILE_PATTERNS.items():
#         file_path = pd.Series(file_path, name="filepath")
#     return np.load(file_path, allow_pickle=True)   

def create_roi_dataframe(loaded_data: dict) -> pd.DataFrame:
    """
    Builds a DataFrame where each row is one ROI and columns hold:
      - roi_fluorescence     : 1D np.ndarray of time series
      - neuropil_fluorescence: 1D np.ndarray of time series
      - is_cell              : bool
    """
    # Extract arrays
    roi_f      = loaded_data["roi_fluorescence"]
    neuropil_f = loaded_data["neuropil_fluorescence"]
    cell_id      = loaded_data["cell_identifier"]

    # Sanity checks
    n_rois = roi_f.shape[0]
    assert neuropil_f.shape[0] == n_rois, "ROI counts must match"
    # Flatten cell_id to shape (n_rois,)
    is_cell_bool = cell_id[:, 0].astype(bool)
    assert is_cell_bool.size == n_rois, "cell_id length must match number of ROIs"

    # Build the DataFrame
    labels = [f"ROI_{i}" for i in range(n_rois)]
    df = pd.DataFrame({
        "roi_fluorescence"     : list(roi_f),
        "neuropil_fluorescence": list(neuropil_f),
        "is_cell"              : list(is_cell_bool),
    }, index=labels)

    return df

def load_beh_data(directory_path):
    np.load(directory_path, allow_pickle=True)


def export_to_csv(dff_data, output_path):
    # Convert the list of ΔF/F arrays into a DataFrame
    dff_df = pd.DataFrame(dff_data)
    # Export to CSV
    dff_df.to_csv(output_path, index=False)
    print(f"ΔF/F data successfully exported to {output_path}")

