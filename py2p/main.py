from py2p.io import load_suite2p_outputs
from py2p.transform import filter_rois
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.config import DATA_DIR

def main():
    data = load_suite2p_outputs(DATA_DIR)

    filtered_data = filter_rois(data)
    #assert filtered_data[0].shape == data['roi_fluorescence'].shape, "Filtered data shape does not match original data shape."
    # Filtered data returns a tuple of (true_cells_only, filtered_roi, filtered_neuropil) where true_cells_only contains
    # numpy booleans (not Python booleans) so we can use sum 
    assert filtered_data[0].sum() == filtered_data[1].shape[0], "Filtered data does not match the number of true cells."

    baseline_fluorescence = calculate_baseline(filtered_data[1], percentile = 3)
    assert baseline_fluorescence == filtered_data[1][1]  

if __name__ == "__main__":
    main()

