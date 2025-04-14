from py2p.io import load_suite2p_outputs
from py2p.transform import filter_rois, filter_neuropil
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.config import DATA_DIR

def main():
    data = load_suite2p_outputs(DATA_DIR)

    filtered_data = filter_rois(data)
    #assert filtered_data.shape == data['roi_fluorescence'].shape, "Filtered data shape does not match original data shape."
    # true_cells_only contains numpy booleans (not Python booleans) so we can use sum - will treat as 1 and 0
    filtered_neuropil = filter_neuropil(data)
    print("filtered_neuropil shape:", filtered_neuropil.shape)
    assert filtered_data.sum == data['roi_fluorescence'].shape[0], "Filtered data does not match the number of true cells."
    #assert filtered_neuropil.shape == filtered_data[2].shape, "Filtered neuropil shape does not match original data shape."

    baseline_fluorescence = calculate_baseline(filtered_data, percentile = 3)
    print("baseline_fluorescence shape:", baseline_fluorescence.shape)
    #assert baseline_fluorescence == filtered_data.shape, "Baseline fluorescence does not match filtered data shape."

    roi_dff = calculate_dff(filtered_data, baseline_fluorescence)
    assert roi_dff.shape == filtered_data.shape, "DFF shape does not match filtered data shape."

if __name__ == "__main__":
    main()

