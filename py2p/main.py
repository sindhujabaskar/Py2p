from py2p.io import load_suite2p_outputs
from py2p.transform import filter_rois
from py2p.calculate import calculate_baseline, calculate_dff
from py2p.config import DATA_DIR

def main():
    data = load_suite2p_outputs(DATA_DIR)
    print('Hello', data)

    filtered_data = filter_rois(data)
    print('Filtered Data:', filtered_data)
    
    baseline_fluorescence = calculate_baseline(filtered_data, percentile = 10)
    print(baseline_fluorescence)

if __name__ == "__main__":
    main()

