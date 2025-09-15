import math
import numpy as np
import statistics as st
import pandas as pd
import scipy.signal as find_peaks

def baseline_percentile(df: pd.DataFrame, percentile): # calculates the specified percentile along each roi's raw fluorescence to yield a baseline fluorescence value
    baseline_fluorescence = df['roi_fluorescence'].apply(lambda x: np.percentile(x, percentile, axis =1, keepdims=True)) # calculate the specified percentile along each roi's raw fluorescence
    return baseline_fluorescence

def deltaf_f(roi_f: pd.DataFrame, baseline_f: pd.DataFrame) -> pd.DataFrame:
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
    dff = roi_f['roi_fluorescence'].combine(baseline_f['baseline_fluorescence'], 
                                 lambda raw, baseline: (raw - baseline) / baseline)

    # Return a new DataFrame with the calculated dF/F values
    return pd.DataFrame({'dff': dff}, index=roi_f.index)

def active_rois(roi_stack: np.ndarray,
                      prominence: int = 2,
                      distance: int = 3):
    """
    Parameters
    ----------
    roi_stack : np.ndarray, shape (n_rois, n_frames)
        dF/F traces for each ROI.
    prominence : float
    distance   : int

    Returns
    -------
    peak_counts : np.ndarray, shape (n_rois,)
    is_active   : np.ndarray (bool), shape (n_rois,)
    """
    peak_counts = []
    for trace in roi_stack:
        peaks, _ = find_peaks(trace, prominence=prominence, distance=distance)
        peak_counts.append(len(peaks))
    peak_counts = np.array(peak_counts)
    is_active   = peak_counts > 3 #num events required to be considered active
    return peak_counts, is_active

def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two points."""
    return math.dist(coord1, coord2)

def compute_roi_tuning(database, subject, session, roi_idx,
                       blank_duration=3.0, stim_duration=2.0):
    """
    Compute a tuning curve for one ROI by measuring its mean ΔF/F
    in the grating window (blank_duration→blank_duration+stim_duration)
    for each orientation across all blocks.

    Returns:
      orientations : list of unique orientations in cycle order
      mean_resps   : array of shape (n_orientations,)
      sem_resps    : array of shape (n_orientations,)
      block_pref   : list of preferred orientation per block
    """
    # pull trial table for this subject/session
    trials = database[('toolkit','trials')].loc[(subject, session)]
    if trials.empty:
        raise ValueError(f"No trials for {subject} {session}")

    # time & dff arrays per trial
    orientations = trials['orientation'].values
    times = trials['time'].values
    dffs = trials['dff'].values  # array of shape (n_trials, n_rois, n_time)

    # window indices for the grating (2s) period
    t0 = blank_duration
    t1 = blank_duration + stim_duration
    
    # use first trial's time vector for indexing
    tvec = np.array(times[0])
    mask = (tvec >= t0) & (tvec < t1)

    # collect responses per trial: mean ΔF/F over stim window
    resp = np.array([dff[roi_idx, mask].mean() for dff in dffs])

    # unique orientations in presented order
    uniq_oris = np.unique(orientations)
    mean_resps = []
    sem_resps = []
    for ori in uniq_oris:
        sel = resp[orientations == ori]
        mean_resps.append(sel.mean())
        sem_resps.append(sel.std(ddof=1) / np.sqrt(len(sel)))

    # preferred orientation per block
    n_blocks = len(resp) // len(uniq_oris)
    block_pref = []
    for b in range(n_blocks):
        block_resp = resp[b*len(uniq_oris):(b+1)*len(uniq_oris)]
        block_pref.append(uniq_oris[np.argmax(block_resp)])

    return list(uniq_oris), np.array(mean_resps), np.array(sem_resps), block_pref

def analyze_pupil_data(
    pickle_data: pd.DataFrame,
    confidence_threshold: float = 0.95,
    pixel_to_mm: float = 53.6,
    dpi: int = 300
) -> pd.DataFrame:
    """
    Analyze pupil data from DeepLabCut output.

    This function processes a pandas DataFrame containing per-frame DeepLabCut outputs
    with 'coordinates' and 'confidence' columns, skipping an initial metadata row,
    and computes interpolated pupil diameters in millimetres.

    Steps
    -----
    1. Skip the first (metadata) row.
    2. Extract and convert 'coordinates' and 'confidence' to NumPy arrays.
    3. For each frame:
       - Squeeze arrays and validate dimensions.
       - Mark landmarks with confidence ≥ threshold.
       - Compute Euclidean distances for predefined landmark pairs.
       - Average valid distances as pupil diameter or assign NaN.
    4. Build a pandas Series of diameters, interpolate missing values, convert from pixels to mm.
    5. Reindex to include the metadata index, then drop the initial NaN to align with valid frames.

    Parameters
    ----------
    pickle_data : pandas.DataFrame
        Input DataFrame with an initial metadata row. Must contain:
        - 'coordinates': array-like of shape (n_points, 2) per entry
        - 'confidence': array-like of shape (n_points,) per entry
    threshold : float, optional
        Minimum confidence to include a landmark in diameter computation.
        Default is 0.1.
    pixel_to_mm : float, optional
        Conversion factor from pixels to millimetres.
        Default is 53.6.
    dpi : int, optional
        Dots-per-inch resolution (not used directly).
        Default is 300.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame ('pupil_diameter_mm') indexed by the input labels
        (excluding the metadata row), containing linearly interpolated
        pupil diameter measurements in millimetres.

    Example
    -------
    Suppose the function returns a DataFrame `result_df`. Its structure would look like:

       frame | pupil_diameter_mm
       ------|------------------
         1   | 1.23
         2   | 1.25
         3   | 1.22
         4   | 1.27
        ...  | ...
    """

    # 1) pull lists, skip metadata row
    coords_list = pickle_data['coordinates'].tolist()[1:]
    conf_list   = pickle_data['confidence'].tolist()[1:]
    
    # Return a warning if no confidence values are above the threshold
    if not any(np.any(np.array(c) >= confidence_threshold) for c in conf_list):
        print(f"[WARNING] {pickle_data.index[0:3]} No confidence values above threshold {confidence_threshold}.")
        
    # 2) to numpy arrays
    coords_arrs = [np.array(c) for c in coords_list]
    conf_arrs   = [np.array(c) for c in conf_list]

    # DEBUG: print first 3 shapes
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords.shape={c.shape}, conf.shape={f.shape}")
        
    # Print the first few values of c and f
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords values:\n{c}")
    #     print(f"[DEBUG] frame {idx} conf values:\n{f}")
        
    # 3) compute mean diameters
    pairs     = [(0, 1), (2, 3), (4, 5), (6, 7)]
    diameters = []
    for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
        pts   = np.squeeze(coords)   # expect (n_points, 2)
        cvals = np.squeeze(conf)     # expect (n_points,)
        # DEBUG unexpected shapes
        if pts.ndim != 2 or cvals.ndim != 1:
            print(f"[WARNING] frame {i} unexpected pts.shape={pts.shape}, conf.shape={cvals.shape}")
            diameters.append(np.nan)
            continue
        #print(f"cval type ={type(cvals)}, with values of type {cvals.dtype}\n compared to {type(confidence_threshold)}")
        valid = cvals >= confidence_threshold
        # print("cvals:", cvals)
        # print("threshold:", confidence_threshold)
        # print("mask  :", valid)  
        ds = [
            euclidean_distance(pts[a], pts[b])
            for a, b in pairs
            if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
        ]
        diameters.append(st.mean(ds) if ds else np.nan)

    # 4) interpolate & convert to mm, align with original index
    pupil_series = (
        pd.Series(diameters, index=pickle_data.index[1:])
          .interpolate()
          .divide(pixel_to_mm)
    )
    pupil_full = pupil_series.reindex(pickle_data.index)

    # DEBUG
    # print(f"[DEBUG analyze_pupil_data] input index={pickle_data.index}")
    # print(f"[DEBUG analyze_pupil_data] output series head:\n{pupil_full.head()}")

    # 5) return DataFrame without the metadata NaN
    return pd.DataFrame({'pupil_diameter_mm': pupil_full.iloc[1:]})

