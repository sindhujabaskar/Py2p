import numpy as np
import pandas as pd

def calculate_baseline(raw_f, percentile): # calculates the specified percentile along each roi's raw fluorescence to yield a baseline fluorescence value
    percentile_list = []
    for row in raw_f:
        find_percentile = np.percentile(row, percentile, keepdims = True)
        percentile_list.append(find_percentile)
    baseline_fluorescence = np.array(percentile_list)
    return baseline_fluorescence

def calculate_dff(raw_f, baseline_f): # calculates the dff values for each roi based on a calculated baseline fluorescence
    dff_list = []
    print("raw_f shape:", raw_f.shape)
    print("baseline_f shape:", baseline_f.shape)
    for row in range(len(raw_f)):
        dff = ((raw_f[row] - baseline_f[row])/(baseline_f[row]))
        dff_list.append(dff)
    dff = pd.DataFrame(dff_list)
    return dff

