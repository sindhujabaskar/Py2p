import numpy as np
import pandas as pd

def calculate_baseline(df: pd.DataFrame, percentile): # calculates the specified percentile along each roi's raw fluorescence to yield a baseline fluorescence value
    baseline_fluorescence = df['roi_f'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) # calculate the specified percentile along each roi's raw fluorescence
    return pd.DataFrame({'baseline_fluorescence': baseline_fluorescence}, index=df.index)

def calculate_dff(roi_f: pd.DataFrame, baseline_f: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the ΔF/F (dF/F) values for each ROI based on the raw fluorescence and baseline fluorescence.

    Parameters
    ----------
    roi_f : pd.DataFrame
        MultiIndexed DataFrame with a column 'roi_f' containing raw fluorescence values.
    baseline_f : pd.DataFrame
        MultiIndexed DataFrame with a column 'baseline_fluorescence' containing baseline fluorescence values.

    Returns
    -------
    pd.DataFrame
        MultiIndexed DataFrame with a column 'dff' containing the calculated ΔF/F values.
    """
    # Ensure the indices match between the two DataFrames
    if not roi_f.index.equals(baseline_f.index):
        raise ValueError("Indices of roi_f and baseline_f must match.")

    # Perform vectorized ΔF/F calculation
    dff = roi_f['roi_f'].combine(baseline_f['baseline_fluorescence'], 
                                 lambda raw, baseline: (raw - baseline) / baseline)

    # Return a new DataFrame with the calculated dF/F values
    return pd.DataFrame({'dff': dff}, index=roi_f.index)