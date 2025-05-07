#%%
from pathlib import Path
import pandas as pd
import re

def bids_dir_to_dataframe(
    root: Path,
    file_glob: str = "**/*.*",
    fallback_by_position: bool = True
) -> pd.DataFrame:
    # Step 1: Compile regex patterns to identify BIDS subject and session folders.
    subj_re = re.compile(r"sub-[A-Za-z0-9]+")
    sess_re = re.compile(r"ses-[A-Za-z0-9]+")

    # Step 2: Initialize an empty list to collect file records.
    records = []

    # Step 3: Recursively walk through the dataset directory to find files matching file_glob.
    for f in root.rglob(file_glob):
        if not f.is_file():
            # Skip any entries that are not files.
            continue
        rel = f.relative_to(root)
        # Compute the file's path relative to the root to extract subject/session information.
        parts = rel.parts

        # Attempt to identify subject and session using BIDS conventions (e.g., sub-001, ses-01 in the filename).
        sub = next((p for p in parts if subj_re.match(p)), None)
        ses = next((p for p in parts if sess_re.match(p)), None)

        # 2) Fallback to first two path components (ie sub-01/ses-01/..)
        if fallback_by_position and (sub is None or ses is None):
            if len(parts) >= 2:
                # Fallback: infer subject and session from the first two path components.
                sub, ses = parts[0], parts[1]
            else:
                continue  # cannot infer
        if sub is None or ses is None:
            continue  # skip entirely non‐BIDS

        dtype = f.name.split('.')[0]
        records.append({
            "Subject": sub,
            "Session": ses,
            "DataType": dtype,
            "Path": str(f)
        })

        # Append the record dict with Subject, Session, DataType, and full file Path.

    # Step 4: Convert the flat list of record dicts into a pandas DataFrame.
    df = pd.DataFrame.from_records(records)
    # Step 5: Pivot the DataFrame so that each DataType becomes a separate column,
    # with a MultiIndex on rows [Subject, Session].
    df = df.pivot_table(
        index=["Subject", "Session"],
        columns="DataType",
        values="Path",
        aggfunc="first"
    )
    # Final: Remove the 'DataType' name from the columns and return the structured DataFrame.
    df.columns.name = None
    return df
#%%
if __name__ == "__main__":

    from pathlib import Path
    bids_df = bids_dir_to_dataframe(Path(r'E:\sbaskar\2408_SU24_F31'))

    # 3. Inspect the resulting DataFrame
    print(bids_df)
# %%
