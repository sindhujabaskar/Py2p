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
