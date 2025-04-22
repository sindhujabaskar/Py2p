import numpy as np
import scipy.interpolate

"""
Filters the ROIs and neuropil fluorescence based on the cell identifier.
    
Parameters:
suite2p_data_output (dict): Dictionary containing Suite2p output data.
    
Returns:
tuple: Filtered ROIs and neuropil fluorescence.
"""

def filter_rois(suite2p_data_output):
    filtered_roi = np.array(suite2p_data_output['roi_fluorescence'][suite2p_data_output['cell_identifier'][:, 0].astype(bool)])
    print("filter_roi shape:", filtered_roi.shape)
    return filtered_roi

def filter_neuropil(suite2p_data_output):
    filtered_neuropil = np.array(suite2p_data_output['neuropil_fluorescence'][suite2p_data_output['cell_identifier'][:, 0].astype(bool)])
    print("filter_pil shape:", filtered_neuropil.shape)   
    return filtered_neuropil

def interpolate_roi(filtered_roi, offset_frames=81, original_rate=9.865, target_rate=10):
    """
    Interpolates ROI data from an original sampling frequency to a new target rate.

    Parameters
    ----------
    filtered_roi : np.ndarray
        A 2D numpy array of shape (num_rois, num_frames) representing the ROI data.
    offset_frames : int, optional
        The number of frames to offset (default is 81).
    original_rate : float, optional
        The original sampling rate in Hz (default is 9.865).
    target_rate : float, optional
        The desired sampling rate in Hz (default is 10).

    Returns
    -------
    interpolated_roi : np.ndarray
        A numpy array with the interpolated data having the same shape as filtered_roi.
    old_time_vector : np.ndarray
        The original time vector based on the provided original_rate.
    new_time_vector : np.ndarray
        The new time vector based on the target_rate.

    Notes
    -----
    The time vectors are computed using a duration based on the total number of frames adjusted by the 
    offset (i.e. new_number_frames = num_frames - offset_frames), but the number of points in the time 
    vectors remains the same as the original number of frames.
    """
    
    # Determine the number of frames in the ROI data.
    # Assuming filtered_roi is of shape (num_rois, num_frames)
    num_frames = filtered_roi.shape[1]
    
    # Adjust the number of frames to account for the offset.
    new_number_frames = num_frames - offset_frames
    
    # Create the old time vector based on the original sampling frequency.
    old_time_vector = np.linspace(0, new_number_frames / original_rate, num_frames)
    
    # Create the new time vector based on the target sampling frequency.
    new_time_vector = np.linspace(0, new_number_frames / target_rate, num_frames)
    
    # Perform cubic spline interpolation along the frame axis (axis=1)
    cs = scipy.interpolate.CubicSpline(old_time_vector, filtered_roi, axis=1)
    interpolated_roi = cs(new_time_vector)
    
    return interpolated_roi #, old_time_vector, new_time_vector

def smooth_dff(dff_data, smoothing_kernel=3):
    """
    Smooths the dFF data using a simple moving average filter.

    Parameters
    ----------
    dff_data : np.ndarray
        A 2D numpy array of shape (num_rois, num_frames) representing the dFF data.
    smoothing_kernel : int, optional
        The size of the smoothing kernel (default is 3).

    Returns
    -------
    smoothed_dff : np.ndarray
        A numpy array with the smoothed dFF data. 
    """

    smoothed_roi_dff = [] # create an empty list to store smoothed dff values per roi

    for roi in range(dff_data.shape[0]):
        smoothed_roi = np.convolve(dff_data[roi] , np.ones(smoothing_kernel)/smoothing_kernel, mode='same') # convolution kernel is [0.33, 0.33, 0.33] 
        smoothed_roi_dff.append(smoothed_roi)
    smoothed_roi_dff = np.array(smoothed_roi_dff)
    return smoothed_roi_dff

def active_rois(filtered_roi, min_prominence=-.5, min_distance=3):
    """
    Identifies active ROIs based on the dFF data.

    Parameters
    ----------
    filtered_roi : np.ndarray
        A 2D numpy array of shape (num_rois, num_frames) representing the ROI data.
    min_prominence : float, optional
        The minimum prominence for peak detection (default is -0.5).
    min_distance : int, optional
        The minimum distance between peaks (default is 3).

    Returns
    -------
    active_rois : np.ndarray
        A boolean array indicating which ROIs are active.
    """
    active_rois = np.zeros(filtered_roi.shape[0], dtype=bool)
    for roi in filtered_roi:
        roi_events, _ = scipy.signal.find_peaks(filtered_roi[roi], prominence = min_prominence, distance = min_distance)
        active_rois[roi,0] = len(roi_events)
    return active_rois