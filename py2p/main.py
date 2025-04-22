#%%
from py2p.io import load_suite2p_outputs, create_roi_dataframe, export_to_csv
from py2p.transform import filter_rois, filter_neuropil, interpolate_roi, smooth_dff, active_rois
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.plot import plot_onething, plot_twothings
from py2p.config import DATA_DIR
#%%

data = load_suite2p_outputs(DATA_DIR)

df = create_roi_dataframe(data)

filtered_data = filter_rois(data)
print('filtered_data:', filtered_data)

# true_cells_only contains numpy booleans (not Python booleans) so we can use sum - will treat as 1 and 0
filtered_neuropil = filter_neuropil(data)
print("filtered_neuropil shape:", filtered_neuropil.shape)

baseline_fluorescence = calculate_baseline(filtered_data, percentile = 3)
print("baseline_fluorescence shape:", baseline_fluorescence.shape)

roi_dff = calculate_dff(filtered_data, baseline_fluorescence)

interpolated_data = interpolate_roi(filtered_data, offset_frames=81, original_rate=9.865, target_rate=10)
print("interpolated_dff shape:", interpolated_data.shape)

interpolated_dff = calculate_dff(interpolated_data, baseline_fluorescence)

smoothed_dff = smooth_dff(interpolated_dff, smoothing_kernel=3)
print("smoothed_dff shape:", smoothed_dff.shape)

active_rois_only = active_rois(smoothed_dff)
print("active_rois_only:", active_rois_only)

#export_to_csv(roi_dff, output_path='roi_dff.csv')

plot_onething(roi_dff)
plot_twothings(interpolated_dff, smoothed_dff)


# %%
from pathlib import Path
from typing import Dict, List

def build_experiment_filedict(root_dir: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """
    Traverse  
        root_dir/
            sub-*/           # subject folders
                ses-*/       # session folders
                    <datatype>/   # e.g. 2p_tiff, dlc_videos, suite2p, etc.
                        *         # any files in there
    and return a nested dict:
        {
            'sub-00': {
                'ses-00': {
                    '2p_tiff':    [Path(...), Path(...), …],
                    'dlc_videos': [Path(...), …],
                    'suite2p':    [Path(...), …],
                    …
                },
                'ses-01': { … },
                …
            },
            'sub-01': { … },
            …
        }
    """
    filedict: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}

    root = Path(root_dir)
    # look for all subject folders
    for sub_folder in sorted(root.glob("sub-*")):
        sub_id = sub_folder.name
        filedict[sub_id] = {}

        # look for all session folders under each subject
        for ses_folder in sorted(sub_folder.glob("ses-*")):
            ses_id = ses_folder.name
            filedict[sub_id][ses_id] = {}

            # each immediate subdirectory is a “datatype”
            for dtype_folder in sorted(p for p in ses_folder.iterdir() if p.is_dir()):
                dtype = dtype_folder.name
                # gather all files under that folder
                files = sorted(f for f in dtype_folder.rglob("*") if f.is_file())
                filedict[sub_id][ses_id][dtype] = files

    return filedict


root = Path("E:/sbaskar/2408_SU24_F31/processed")
filedict = build_experiment_filedict(root)

# e.g.:
#print(filedict["sub-SB03"]["ses-01"]["suite2p"])
# → [Path('E:/sbaskar/2408_SU24_F31/sub-00/ses-00/suite2p/iscell.npy'),
#    Path('E:/sbaskar/2408_SU24_F31/sub-00/ses-00/suite2p/F.npy'),
#    …]

# %%
import pandas as pd

rows = []
for sub, ses_dict in filedict.items():
    for ses, dtype_dict in ses_dict.items():
        for dtype, paths in dtype_dict.items():
            for p in paths:
                rows.append({"subject": sub,
                             "session": ses,
                             "datatype": dtype,
                             "filepath": str(p)})

df = pd.DataFrame(rows)
df = df.set_index(["subject", "session", "datatype"])

# %%
def make_wide_df(long_df: pd.DataFrame) -> pd.DataFrame:
    # Group and collect each group of filepaths into a list
    grouped = (
        long_df
        .groupby(["subject", "session", "datatype"])["filepath"]
        .apply(list)
    )
    # Unstack the last level (“datatype”) so it becomes columns
    wide = grouped.unstack("datatype").sort_index(axis=1)
    return wide


