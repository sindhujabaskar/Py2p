import numpy as np
import pandas as pd

def calculate_baseline(filtered_roi, percentile): # calculates the specified percentile along each roi's raw fluorescence to yield a baseline fluorescence value
    baseline_percentile = np.array([])
    for row in filtered_roi:
        find_percentile = np.percentile(row, percentile, keepdims = True)
        np.append(baseline_percentile, find_percentile)
    return baseline_percentile

def calculate_dff(raw_f, baseline_f): # calculates the dff values for each roi based on a calculated baseline fluorescence
    dff_list = []
    for row in range(len(raw_f)):
        dff = ((raw_f[row] - baseline_f[row])/(baseline_f[row]))
        dff_list.append(dff)
    dff = pd.DataFrame(dff_list)
    return dff