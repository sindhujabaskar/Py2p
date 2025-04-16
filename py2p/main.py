from py2p.io import load_suite2p_outputs, export_to_csv
from py2p.transform import filter_rois, filter_neuropil, interpolate_roi, smooth_dff, active_rois
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.plot import plot_onething, plot_twothings
from py2p.config import DATA_DIR

def main():
    data = load_suite2p_outputs(DATA_DIR)

    filtered_data = filter_rois(data)
    print('filtered_data:', filtered_data)
    #assert filtered_data.shape == data['roi_fluorescence'].shape, "Filtered data shape does not match original data shape."
    # true_cells_only contains numpy booleans (not Python booleans) so we can use sum - will treat as 1 and 0
    filtered_neuropil = filter_neuropil(data)
    print("filtered_neuropil shape:", filtered_neuropil.shape)
    #assert filtered_data.sum == data['roi_fluorescence'].shape[0], "Filtered data does not match the number of true cells."
    #assert filtered_neuropil.shape == filtered_data[2].shape, "Filtered neuropil shape does not match original data shape."

    baseline_fluorescence = calculate_baseline(filtered_data, percentile = 3)
    print("baseline_fluorescence shape:", baseline_fluorescence.shape)
    #assert baseline_fluorescence == filtered_data.shape, "Baseline fluorescence does not match filtered data shape."

    roi_dff = calculate_dff(filtered_data, baseline_fluorescence)
    #assert roi_dff.shape == filtered_data.shape, "DFF shape does not match filtered data shape."

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
if __name__ == "__main__":
    main()

