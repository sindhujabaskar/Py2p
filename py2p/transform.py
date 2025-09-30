import numpy as np
import scipy.interpolate
import pandas as pd

def trials(row):
    directions = [0, 45, 90, 135, -180, -45, -90, -135]  # the 8 grating angles you cycle through
    orientations = [0, 45, 90, 135]
    block_size   = len(directions)

    # define inputs
    deltaf_f   = np.asarray(row[('calculate', 'smoothed_dff')])  
    timestamps = np.asarray(row[('toolkit',   'timestamps')])      
    trial_df   = row[('toolkit', 'trial_index')]          

    all_trials = []
    for trial_id, (start, grating, stop) in enumerate(zip(trial_df['trial_start'], trial_df['gratings_start'], trial_df['trial_end'])):
        trial_mask    = (timestamps >= start) & (timestamps < stop)
        off_mask      = (timestamps >= start) & (timestamps < grating)
        on_mask       = (timestamps >= grating) & (timestamps < stop)
        time_slice    = timestamps[trial_mask]           # (t,)
        trial_data    = deltaf_f[:, trial_mask]          # (n_rois, t)
        time_off      = timestamps[off_mask]            # (t_off,)
        dff_off       = deltaf_f[:, off_mask]           # (n_rois, t_off)
        time_on       = timestamps[on_mask]             # (t_on,)
        dff_on        = deltaf_f[:, on_mask]            # (n_rois, t_on)
        direction     = directions[trial_id % len(directions)]
        orientation   = orientations[trial_id % len(orientations)]
        trial_block   = trial_id // block_size

        all_trials.append({
            'trial':       trial_id,
            'block':       trial_block,
            'direction':   direction,
            'orientation': orientation,
            'time':        time_slice,
            'dff':         trial_data,
            'time_off':    time_off,
            'dff_off':     dff_off,
            'time_on':     time_on,
            'dff_on':      dff_on
        })

    return pd.DataFrame(all_trials)

## OLD FUNCTIONS


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
    num_frames = filtered_roi['roi_fluorescence'][0].size
    
    # Adjust the number of frames to account for the offset.
    new_number_frames = num_frames - offset_frames
    
    # Create the old time vector based on the original sampling frequency.
    old_time_vector = np.linspace(0, new_number_frames / original_rate, num_frames)
    
    # Create the new time vector based on the target sampling frequency.
    new_time_vector = np.linspace(0, new_number_frames / target_rate, num_frames)
    
    # Perform cubic spline interpolation along the frame axis (axis=1)
    cs = scipy.interpolate.CubicSpline(old_time_vector, filtered_roi, axis=1)
    interpolated_roi = cs(new_time_vector)
    
    return interpolated_roi

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


