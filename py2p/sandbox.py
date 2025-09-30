'''
This the sandbox for just getting things down, giving it a slight structure/procedural order,

and most importantly-->playing around without getting too serious about it.
so that u can get back to it later and see what u was thinking.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_peak_session_stats(database, session_labels=None):
    """
    Compute per-session peak count statistics (mean, std, count, sem, label).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # Get peak counts with MultiIndex
    peak_counts = database[('analysis', 'num_peaks_prominence')]
    # Convert to DataFrame for easier grouping
    peak_df = peak_counts.reset_index()
    peak_df.columns = ['Subject', 'Session', 'num_peaks']
    # Group by session and compute mean and SEM
    session_stats = peak_df.groupby('Session')['num_peaks'].agg(['mean', 'std', 'count']).reset_index()
    session_stats['sem'] = session_stats['std'] / np.sqrt(session_stats['count'])
    # Apply default session labels if none provided
    if session_labels is None:
        session_labels = {
            'ses-01': 'Baseline',
            'ses-02': 'Low EtOH',
            'ses-03': 'Saline',
            'ses-04': 'High EtOH'
        }
    session_stats['label'] = session_stats['Session'].map(session_labels)
    # Set up high-quality plotting parameters
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'legend.frameon': False
    })
    return session_stats

def plot_poster_trace(database, subject, session, roi_idx, num_points=1000):
    """
    Create a poster-quality ΔF/F trace for one ROI.
    
    Args:
      database   : main DataFrame
      subject    : e.g. 'sub-SB03'
      session    : e.g. 'ses-01'
      roi_idx    : ROI index (e.g. 61)
      num_points : number of frames to plot (default: 1000)
    
    Returns:
      fig, ax : Matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    # extract data
    times = database.toolkit.timestamps.loc[subject, session].values[:num_points]
    trace = database.calculate.smoothed_dff.loc[subject, session][roi_idx][:num_points]

    # poster styling
    plt.rcParams.update({
        'figure.figsize': (8, 4),
        'figure.dpi': 600,
        'font.size': 16,
        'font.family': 'Arial',
        'axes.linewidth': 2,
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'xtick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.size': 8,
        'ytick.major.width': 2,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    fig, ax = plt.subplots()
    ax.plot(times, trace, color='#2E86AB', linewidth=3)

    # labels & title
    ax.set_xlabel('Time (s)', labelpad=10, weight='bold')
    ax.set_ylabel('ΔF/F', labelpad=10, weight='bold')
    ax.set_title(f'{subject} {session} – ROI {roi_idx}', pad=15, weight='bold')

    # thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    return fig, ax

def plot_session_avg_blocks(database, subject, session, save_path=None):
    """
    Plot the concatenated average dF/F trace across all blocks in a session.

    Args:
      database: pandas DataFrame with trials table
      subject : subject identifier (e.g. 'sub-SB03')
      session : session identifier (e.g. 'ses-01')

    Returns:
      plt  : matplotlib pyplot module with the figure displayed
    """
    # pull out the per‐trial DataFrame for the session
    df = database['toolkit','trials'][(subject, session)]
    if df.empty:
        raise ValueError(f"no trials found for session {subject} {session}")

    # compute block-wise average traces
    blocks = sorted(df['block'].unique())
    block_traces = []
    for blk in blocks:
        blk_df = df[df['block'] == blk]
        # concatenate trials within block
        times_blk = np.concatenate([np.array(r['time']) for _, r in blk_df.iterrows()])
        avg_blk = np.concatenate([np.array(r['dff']).mean(axis=0) for _, r in blk_df.iterrows()])
        block_traces.append(avg_blk)

    # assume consistent time axis across blocks
    time_axis = times_blk
    # average across blocks
    mean_blk_trace = np.mean(np.vstack(block_traces), axis=0)

    # create high-quality, vectorized figure
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2
    })
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.plot(time_axis, mean_blk_trace,
            label='avg across blocks',
            color='tab:green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Avg ΔF/F')
    ax.set_title(f'{subject} {session} – average across blocks')
    ax.grid(True)
    ax.set_ylim(0, 0.5)  # standard y-axis scale
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg')
    plt.show()
    return fig, ax

def plot_event_frequency(database, save_path=None):
    """
    Plot calcium event frequency per condition with individual subject points and connecting lines.
    """
    # recompute peak counts dataframe
    peak_counts = database[('analysis','num_peaks_prominence')]
    peak_df = peak_counts.reset_index()
    peak_df.columns = ['Subject','Session','num_peaks']

    # compute session stats
    session_stats = peak_df.groupby('Session')['num_peaks'].agg(['mean','std','count']).reset_index()
    session_stats['sem'] = session_stats['std'] / np.sqrt(session_stats['count'])
    session_labels = { 'ses-01':'Baseline','ses-02':'Low EtOH','ses-03':'Saline','ses-04':'High EtOH' }
    session_stats['label'] = session_stats['Session'].map(session_labels)

    # event frequency per trial
    ts = database[('toolkit','timestamps')]
    peak_df['duration'] = peak_df.apply(
        lambda r: ts.loc[(r.Subject,r.Session)].max()-ts.loc[(r.Subject,r.Session)].min(),axis=1
    )
    peak_df['freq'] = peak_df['num_peaks']/peak_df['duration']

    # aggregate frequency
    freq_lists = peak_df.groupby('Session')['freq'].apply(list)
    session_stats['freq_list'] = session_stats['Session'].map(freq_lists)
    session_stats['mean_freq'] = session_stats['freq_list'].apply(np.mean)

    # high-quality figure defaults
    plt.rcParams.update({ 'figure.dpi':300,'savefig.dpi':300,'font.size':14,
        'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'lines.linewidth':2 })
    fig, ax = plt.subplots(figsize=(10,7),dpi=300)

    # bar plot of mean frequency
    ax.bar(session_stats['label'], session_stats['mean_freq'],
           color='#2E86AB', alpha=0.8, edgecolor='white')
    # scatter individual subject points and connect by subject
    sessions = session_stats['Session'].tolist()
    x_positions = np.arange(len(sessions))
    freq_pivot = peak_df.pivot(index='Subject', columns='Session', values='freq')
    for subj, row in freq_pivot.iterrows():
        y = row[sessions].values
        ax.plot(x_positions, y, color='gray', alpha=0.5)
        jitter = np.random.normal(0,0.05,size=len(y))
        ax.scatter(x_positions+jitter, y, color='black', s=40, zorder=5)

    # labels and styling
    ax.set_xlabel('Condition', fontweight='bold')
    ax.set_ylabel('Event Frequency (Hz)', fontweight='bold')
    ax.set_title('Calcium Event Frequency per Condition', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max([max(l) for l in session_stats['freq_list']])*1.15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path,format='svg')
    plt.show()
    return fig, ax

def plot_single_roi_trace(database, subject, session, roi_idx,
                           window=None, save_path=None):
    """
    Plot the ΔF/F trace for a single ROI in a (subject, session).
    window : tuple (start, end) in seconds to limit plot range.
    save_path : optional path to save SVG output.
    Returns: fig, ax
    """

    # retrieve timestamps and ROI trace
    ts = database.toolkit.timestamps.loc[subject, session].values
    trace = database.calculate.smoothed_dff.loc[subject, session][roi_idx]

    # apply window if specified
    if window is not None:
        start, end = window
        mask = (ts >= start) & (ts <= end)
        ts_plot = ts[mask]
        trace_plot = trace[mask]
    else:
        ts_plot = ts
        trace_plot = trace

    # high-quality figure settings
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2
    })
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(ts_plot, trace_plot, color='#2E86AB')

    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('ΔF/F', fontweight='bold')
    ax.set_title(f'{subject} {session} – ROI {roi_idx}', fontweight='bold')
    ax.grid(True, alpha=0.3)

    if window is not None:
        ax.set_xlim(window)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg')
    plt.show()
    return fig, ax

def plot_multi_roi_traces(database, selections, window=None, save_path=None):
    """
    Plot ΔF/F traces for multiple ROIs in subplots with consistent y-axis scale.

    Args:
      database   : pandas DataFrame with toolkit timestamps and calculate.smoothed_dff
      selections : list of tuples (subject, session, roi_idx)
      window     : optional (start, end) time window in seconds
      save_path  : optional path to save SVG output
    Returns:
      fig, axes  : Matplotlib Figure and Axes array
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil, sqrt

    # High-quality defaults
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2
    })

    # Collect time vectors and traces
    ts_list = []
    trace_list = []
    for subj, sess, roi in selections:
        ts = database.toolkit.timestamps.loc[subj, sess].values
        trace = database.calculate.deltaf_f.loc[subj, sess][roi]
        if window is not None:
            start, end = window
            mask = (ts >= start) & (ts <= end)
            ts = ts[mask]
            trace = trace[mask]
        ts_list.append(ts)
        trace_list.append(trace)

    # Determine common y-axis limits
    y_min = min([tr.min() for tr in trace_list])
    y_max = max([tr.max() for tr in trace_list])

    # Setup subplots
    n = len(selections)
    ncols = int(ceil(sqrt(n)))
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 3*nrows), dpi=300)
    axes = np.array(axes).reshape(-1)

    # Plot each ROI
    for ax, (ts, trc), (subj, sess, roi) in zip(axes, zip(ts_list, trace_list), selections):
        ax.plot(ts, trc, color='#2E86AB')
        ax.set_title(f'{subj} {sess} – ROI {roi}', fontweight='bold')
        ax.set_xlim(window if window is not None else (ts.min(), ts.max()))
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg')
    plt.show()
    return fig, axes

def plot_event_rate_timecourse(database, sessions, bin_size=60, smoothing_sigma=1.0, save_path=None):
    """
    Plot dynamic event rate (events/min) over time for one or multiple sessions,
    averaged across all ROIs and subjects, with optional smoothing and overlay.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d  # type: ignore

    # ensure sessions is a list
    if isinstance(sessions, str):
        sessions = [sessions]
    # collect event times per session
    events_dict = {}
    max_t = 0
    for sess in sessions:
        idx = database[('analysis','peaks_prominence')].index
        subjects = sorted({subj for subj, s in idx if s == sess})
        times = []
        for subj in subjects:
            inds = database[('analysis','peaks_prominence')].loc[(subj, sess)]
            ts = database[('toolkit','timestamps')].loc[(subj, sess)].values
            times.extend(ts[inds])
        if times:
            events_dict[sess] = np.array(times)
            max_t = max(max_t, max(times))
    if not events_dict:
        raise ValueError(f"No events found for sessions {sessions}")

    # define common bins
    bins = np.arange(0, max_t + bin_size, bin_size)
    mid_bins = bins[:-1] + bin_size / 2

    # high-quality figure defaults
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2
    })
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    # plot each session with smoothing
    for sess in sessions:
        times = events_dict.get(sess, [])
        counts, _ = np.histogram(times, bins=bins)
        rates = counts * (60.0 / bin_size)
        if smoothing_sigma > 0:
            rates = gaussian_filter1d(rates, sigma=smoothing_sigma)
        ax.plot(mid_bins, rates, '-o', label=sess)
    ax.legend(title='Session')
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Event Rate (events/min)', fontweight='bold')
    # Title using provided session(s)
    session_labels = ', '.join(sessions)
    ax.set_title(f'{session_labels} Event Rate Timecourse', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg')
    plt.show()
    return fig, ax

def plot_session_trial_averages(database: pd.DataFrame,
                         block_idx: int = 0,
                         sessions: list = None,
                         save_path: str = None):
    """
    Plot block averages across all animals, organized by session/condition.
    
    Args:
        database  : The main database DataFrame
        block_idx : Which block to analyze (default: 0)
        sessions  : List of sessions to include (default: all available)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if sessions is None:
        sessions = ['ses-01', 'ses-02', 'ses-03', 'ses-04']
    
    # Session label mapping
    session_labels = {
        'ses-01': 'Baseline',
        'ses-02': 'Low EtOH',
        'ses-03': 'Saline',
        'ses-04': 'High EtOH'
    }
    
    # Color scheme for each session
    session_colors = {
        'ses-01': "#DFED1F",  # Blue
        'ses-02': "#35A9E8",  # Purple
        'ses-03': "#FD9F09",  # Orange
        'ses-04': "#1D3CC7"   # Red
    }
    
    # Set up high-quality plotting parameters
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'legend.frameon': False
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Process each session
    for session in sessions:
        session_data = []
        
        # Get all subjects for this session
        try:
            trials_data = database['toolkit', 'trials']
            subjects = [key[0] for key in trials_data.keys() if key[1] == session]
        except KeyError:
            print(f"Warning: No trial data found for session {session}")
            continue
        
        # Collect data from all subjects for this session
        for subject in subjects:
            try:
                df = database['toolkit','trials'][(subject, session)]
                blk = df[df['block'] == block_idx]
                
                if blk.empty:
                    continue
                
                # Collect time vectors and mean‐across‐ROIs dF/F per trial
                all_times = []
                all_avg_dffs = []
                for _, row in blk.iterrows():
                    time = np.array(row['time'])
                    dff = np.array(row['dff'])
                    avg_dff = dff.mean(axis=0)
                    all_times.append(time)
                    all_avg_dffs.append(avg_dff)
                
                # Concatenate across trials for this subject
                full_time = np.concatenate(all_times)
                full_avg_dff = np.concatenate(all_avg_dffs)
                
                session_data.append((full_time, full_avg_dff))
                
            except KeyError:
                print(f"Warning: No data found for {subject} {session}")
                continue
        
        if not session_data:
            continue
        
        # Find common time grid (use the first subject's time as reference)
        ref_time = session_data[0][0]
        
        # Interpolate all subjects to common time grid
        interpolated_data = []
        for time, dff in session_data:
            # Simple approach: use the data as-is if times match, otherwise interpolate
            if len(time) == len(ref_time):
                interpolated_data.append(dff)
            else:
                # For different lengths, we'll use a simpler approach
                # You might want to implement proper interpolation here
                interpolated_data.append(dff)
        
        # Calculate mean and SEM across subjects
        stacked_data = np.array(interpolated_data)
        mean_trace = np.mean(stacked_data, axis=0)
        sem_trace = np.std(stacked_data, axis=0) / np.sqrt(stacked_data.shape[0])
        
        # Plot mean trace
        ax.plot(ref_time, mean_trace, 
               label=f"{session_labels.get(session, session)} (n={stacked_data.shape[0]})",
               color=session_colors.get(session, '#000000'),
               linewidth=2.5,
               alpha=0.8)
        
        # Add SEM shading
        ax.fill_between(ref_time, 
                       mean_trace - sem_trace,
                       mean_trace + sem_trace,
                       color=session_colors.get(session, '#000000'),
                       alpha=0.2)
    
    # Customize the plot
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg ΔF/F (Mean ± SEM)', fontsize=14, fontweight='bold')
    ax.set_title(f'Calcium Activity Across Trials', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    # save figure if path provided
    if save_path:
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()

def plot_pupil_speed_matrix(database: pd.DataFrame,
                            
                            subject: str,
                            sessions: list = None,
                            save_path: str = None):
    """
    Create a subplot matrix with two columns: left for pupil diameter, right for inverted speed
    with inverted distance overlay. All y-axes are standardized across sessions for comparison.

    Args:
        database   : main database DataFrame
        subject    : subject identifier
        sessions   : list of session IDs (default: all available for subject)
        save_path  : optional SVG path to save the figure
    Returns:
        fig, axs   : Matplotlib Figure and Axes array
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Poster-ready defaults
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 14,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'legend.frameon': False
    })

    # Determine sessions
    if sessions is None:
        sessions = sorted({sess for subj0, sess in database[('analysis','mean_deltaf_f')].index if subj0 == subject})
    n = len(sessions)

    # Precompute global y-limits for original pupil data
    all_pupil = []
    all_speed = []
    all_dist = []
    for sess in sessions:
        # original pupil and locomotion data
        pup = database[('analysis','pupil_diameter_mm')].loc[(subject, sess)]
        ts_sm = database[('analysis','time_smoothed')].loc[(subject, sess)]
        sp_sm = database[('analysis','speed_smoothed')].loc[(subject, sess)]
        beh = database[('raw','beh')].loc[(subject, sess)]
        all_pupil.append(pup)
        all_speed.append(-sp_sm)               # inverted
        all_dist.append(-beh['distance'].values)  # inverted
    pupil_min, pupil_max = np.min([arr.min() for arr in all_pupil]), np.max([arr.max() for arr in all_pupil])
    speed_min, speed_max = np.min([arr.min() for arr in all_speed]), np.max([arr.max() for arr in all_speed])
    dist_min, dist_max   = np.min([arr.min() for arr in all_dist]),  np.max([arr.max() for arr in all_dist])

    # Setup figure and axes
    fig, axs = plt.subplots(n, 2, figsize=(12, 3*n), dpi=300, sharex=False)

    for i, sess in enumerate(sessions):
        ax0, ax1 = axs[i, 0], axs[i, 1]
        # Pupil (original)
        pup = database[('analysis','pupil_diameter_mm')].loc[(subject, sess)]
        ts_pup = database[('toolkit','pupil_timestamps')].loc[(subject, sess)]
        ax0.plot(ts_pup, pup, color='#2E86AB')
        ax0.set_ylim(pupil_min, pupil_max)
        ax0.set_ylabel('Pupil (mm)')
        if i == 0:
            ax0.set_title('Pupil Diameter')

        # Speed & Distance
        ts_sm = database[('analysis','time_smoothed')].loc[(subject, sess)]
        sp_sm = database[('analysis','speed_smoothed')].loc[(subject, sess)]
        beh = database[('raw','beh')].loc[(subject, sess)]
        # plot speed
        ax1.plot(ts_sm, -sp_sm, color='#A23B72', label='Speed')
        ax1.set_ylim(speed_min, speed_max)
        ax1.set_ylabel('Speed (inverted)')
        if i == 0:
            ax1.set_title('Speed & Distance')
        # overlay distance
        ax2 = ax1.twinx()
        ax2.plot(beh['timestamp'], -beh['distance'], color='#F18F01', alpha=0.6, label='Distance')
        ax2.set_ylim(dist_min, dist_max)
        ax2.set_ylabel('Distance (inverted)')

        # X-axis label
        ax1.set_xlabel('Time (s)')
        if i < n-1:
            ax0.tick_params(labelbottom=False)
            ax1.tick_params(labelbottom=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()
    return fig, axs

def plot_trial(database: pd.DataFrame,
               subject: str,
               session: str,
               trial_idx: int = 0,
               average: bool = True):
    """
    Plot the dF/F trace for one trial.

    Args:
      subject   : e.g. 'sub-SB03'
      session   : e.g. 'ses-01'
      trial_idx : zero-based index of the trial in that session
      average   : if True, plot the mean across all ROIs; 
                  if False, plot each ROI separately
    """
    # pull out the per‐trial DataFrame
    df = database['toolkit','trials'][(subject, session)]

    # grab the time vector and dff array for that trial
    t   = df['time'].iloc[trial_idx]
    dff = df['dff'].iloc[trial_idx]

    plt.figure()
    if average:
        y = dff.mean(axis=0)
        plt.plot(t, y, lw=2, label='mean dF/F')
    else:
        for i, roi in enumerate(dff):
            plt.plot(t, roi, label=f'ROI {i}')
        plt.legend(loc='best')

    plt.xlabel('Time (s)')
    plt.ylabel('dF/F')
    plt.title(f'{subject} {session} – trial #{trial_idx}')
    plt.tight_layout()
    plt.show()

def plot_block(database: pd.DataFrame,
               subject: str,
               session: str,
               block_idx: int):
    """
    Plot the concatenated average dF/F trace for all ROIs in a block.

    Args:
      subject   : e.g. 'sub-SB03'
      session   : e.g. 'ses-01'
      block_idx : integer block identifier
    """
    # pull out the per‐trial DataFrame and subset by block
    df = database['toolkit','trials'][(subject, session)]
    blk = df[df['block'] == block_idx]
    if blk.empty:
        raise ValueError(f"no trials found for block {block_idx}")

    # collect time vectors and mean‐across‐ROIs dF/F per trial
    all_times = []
    all_avg_dffs = []
    for _, row in blk.iterrows():
        time = np.array(row['time'])        # shape: (n_time,)
        dff  = np.array(row['dff'])         # shape: (n_rois, n_time)
        avg_dff = dff.mean(axis=0)          # shape: (n_time,)
        all_times.append(time)
        all_avg_dffs.append(avg_dff)

    # concatenate across trials
    full_time    = np.concatenate(all_times)
    full_avg_dff = np.concatenate(all_avg_dffs)

    # plot
    plt.figure(figsize=(14, 5))
    plt.plot(full_time, full_avg_dff,
             label=f'block {block_idx} avg ΔF/F',
             color='tab:blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Avg ΔF/F')
    plt.title(f'{subject} {session} – block #{block_idx}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from py2p.process import compute_roi_tuning

def plot_all_rois_tuning_polar(database, subject, session, roi_idxs=None,
                               blank_duration=3.0, stim_duration=2.0):
    """
    Create a grid of polar tuning plots for multiple ROIs in one session.
    roi_idxs: list of ROI indices to plot (default = all).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil, sqrt

    # pull trial table to verify session exists
    trials = database[('toolkit','trials')].loc[(subject, session)]
    if trials.empty:
        raise ValueError(f"No trials for {subject} {session}")

    # determine ROIs
    if roi_idxs is None:
        # assume filtered roi_fluorescence gives n_rois
        n_rois = database.loc[(subject, session)][('filter','roi_fluorescence')].shape[0]
        roi_idxs = list(range(n_rois))

    # precompute tuning for each ROI to determine consistent radial scale
    tuning_list = [compute_roi_tuning(
        database, subject, session, roi,
        blank_duration, stim_duration
    ) for roi in roi_idxs]
    # determine max radius across all ROIs
    rmax = max((np.max(means + sems) for _, means, sems, _ in tuning_list))
    n = len(roi_idxs)
    ncols = int(ceil(sqrt(n)))
    nrows = int(ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             subplot_kw={'projection':'polar'},
                             figsize=(4*ncols, 4*nrows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for ax, roi in zip(axes, roi_idxs):
        oris, means, sems, _ = compute_roi_tuning(
            database, subject, session, roi,
            blank_duration, stim_duration
        )
        thetas = np.deg2rad(np.concatenate([oris, oris[:1]]))
        vals   = np.concatenate([means, means[:1]])
        errs   = np.concatenate([sems, sems[:1]])

        ax.plot(thetas, vals, '-o', color='#2E86AB', linewidth=1.5)
        ax.fill_between(thetas, vals-errs, vals+errs, color='#2E86AB', alpha=0.3)
        ax.set_xticks(np.deg2rad(oris))
        ax.set_xticklabels([f"{int(o)}°" for o in oris], fontsize=8)
        ax.set_title(f"ROI {roi}", fontsize=10, pad=10)
        ax.set_theta_zero_location('E')  # 0° at right (East)
        ax.set_theta_direction(1)         # angles increase counterclockwise
        ax.set_ylim(0, rmax)              # consistent radial scale across ROIs

    # hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{subject} {session} — ROI tuning (polar)", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes

def plot_roi_tuning_polar(database, subject, session, roi_idx,
                           blank_duration=3.0, stim_duration=2.0,
                           save_path=None):
     """
     Plot a single ROI's tuning curve in polar coordinates.

     Args:
       database        : pandas DataFrame with required fields
       subject         : subject identifier (e.g. 'sub-SB03')
       session         : session identifier (e.g. 'ses-01')
       roi_idx         : index of the ROI to plot
       blank_duration  : pre-stimulus blank duration (s)
       stim_duration   : stimulus duration (s)
       save_path       : optional path to save SVG output
     Returns:
       fig, ax         : matplotlib Figure and Axes
     """
     
     # compute orientation tuning for the ROI
     oris, means, sems, _ = compute_roi_tuning(
         database, subject, session, roi_idx,
         blank_duration, stim_duration
     )
     # wrap for closed polar curve
     thetas = np.deg2rad(np.concatenate([oris, oris[:1]]))
     vals   = np.concatenate([means, means[:1]])
     errs   = np.concatenate([sems, sems[:1]])

     # wrap polar data and compute radial max
     rmax = np.max(vals + errs)
     # create high-resolution polar figure
     fig = plt.figure(figsize=(6, 6), dpi=300)
     ax = fig.add_subplot(111, projection='polar')

     # plot mean and SEM shading
     ax.plot(thetas, vals, '-o', color='#2E86AB', linewidth=1.5)
     ax.fill_between(thetas, vals-errs, vals+errs, color='#2E86AB', alpha=0.3)

     # angle ticks and labels
     ax.set_xticks(np.deg2rad(oris))
     ax.set_xticklabels([f"{int(o)}°" for o in oris], fontsize=10)

     # orientation and styling
     ax.set_theta_zero_location('E')  # 0° at right
     ax.set_theta_direction(1)         # CCW positive
     ax.set_title(f"{subject} {session} — ROI {roi_idx} tuning (polar)",
                  fontweight='bold', pad=10)
    
     # consistent radial scale
     ax.set_ylim(0, rmax)

     plt.tight_layout()
     if save_path:
         fig.savefig(save_path, format='svg')
     plt.show()
     return fig, ax

def plot_session_overview(database: pd.DataFrame,
                          subject: str,
                          session: str,
                          save_path: Optional[str] = None):
    """
    Plot a 4-panel overview for one (subject, session):
      1) mean ΔF/F across ROIs (10 Hz timestamps)
      2) pupil diameter (mm) vs 40 Hz pupil timestamps
      3) running speed (inverted)
      4) distance (inverted)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # poster-ready defaults
    plt.rcParams.update({
        'figure.dpi':      300,
        'savefig.dpi':     300,
        'font.size':       14,
        'axes.titlesize':  16,
        'axes.labelsize':  14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2
    })

    # set up 4 stacked subplots
    fig, axes = plt.subplots(4, 1, sharex=False, figsize=(10, 12))

    # 1) Mean ΔF/F (10 Hz)
    ts_img   = database[('toolkit','timestamps')].loc[(subject, session)].values
    mean_dff = database[('analysis','mean_deltaf_f')].loc[(subject, session)]
    axes[0].plot(ts_img, mean_dff, color='C0')
    axes[0].set_ylabel('Mean ΔF/F')
    axes[0].set_title(f'{subject} — {session}')

    # 2) Pupil diameter vs 40 Hz timestamps
    pupil    = database[('analysis','pupil_diameter_mm')].loc[(subject, session)].values
    pupil_ts = database[('toolkit','pupil_timestamps')].loc[(subject, session)]
    axes[1].plot(pupil_ts, pupil, color='C1')
    axes[1].set_ylabel('Pupil (mm)')

    # 3) Running speed (smoothed)
    # use preprocessed locomotion data
    ts_sm = database[('analysis','time_smoothed')].loc[(subject, session)]
    sp_sm = database[('analysis','speed_smoothed')].loc[(subject, session)]
    # invert speed so positive values plot downward
    axes[2].plot(ts_sm, -sp_sm, color='C2')
    axes[2].set_ylabel('Running speed (smoothed)')

    # 4) Distance (inverted)
    # use raw behavior data for distance
    beh_dist = database[('raw','beh')].loc[(subject, session)]
    axes[3].plot(beh_dist['timestamp'], -beh_dist['distance'], color='C3')
    axes[3].set_ylabel('Distance')

    # common x-axis label
    axes[3].set_xlabel('Time (s)')
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0, 2.0)
    axes[2].set_ylim(-0.5, 1.0)
    axes[3].set_ylim(0, 150)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg')
    plt.show()
    return fig, axes

def trial_mean(database: pd.DataFrame):
    """
    Compute mean grating-on dF/F response per trial for all ROIs across all sessions and add to database.
    
    Parameters
    ----------
    database : pd.DataFrame
        Main database with MultiIndex columns
        
    Returns
    -------
    None
        Modifies database in place by adding ('analysis', 'ROI_trial_mean_dff') column
    """
    
    # Initialize storage for all sessions
    roi_tuning_data = {}
    
    # Iterate through all subject-session pairs
    for (subject, session) in database.index:
        try:
            # define variables for this session
            trials = database[('toolkit','trials')].loc[(subject, session)]
            
            # Skip if no trials data
            if trials is None or trials.empty:
                print(f"No trials data for {subject} {session}, skipping...")
                roi_tuning_data[(subject, session)] = None
                continue
                
            orientations = trials['orientation'].values
            directions = trials['direction'].values
            dff_grat = trials['dff_on'].values
            dff_grey = trials['dff_off'].values

            # get number of ROIs from first trial
            num_rois = dff_grat[0].shape[0]
            num_trials = len(dff_grat)
            
            # initialize arrays to store responses for all ROIs
            grat_responses = np.zeros((num_rois, num_trials))
            grey_responses = np.zeros((num_rois, num_trials))
            
            # calculate mean dF/F over stim and grey windows for all ROIs
            for trial_idx, (dff_on, dff_off) in enumerate(zip(dff_grat, dff_grey)):
                grat_responses[:, trial_idx] = dff_on.mean(axis=1)  # mean across time for each ROI
                grey_responses[:, trial_idx] = dff_off.mean(axis=1)  # mean across time for each ROI
            
            # create results DataFrame
            trial_means = pd.DataFrame({
                'orientations': orientations, 
                'directions': directions, 
                'grat_resp_all_rois': [grat_responses[:, i] for i in range(num_trials)], 
                'grey_resp_all_rois': [grey_responses[:, i] for i in range(num_trials)]
            })
            
            # store in dictionary
            roi_tuning_data[(subject, session)] = trial_means
            print(f"Computed ROI tuning for {subject} {session}")
            
        except Exception as e:
            print(f"Error processing {subject} {session}: {e}")
            roi_tuning_data[(subject, session)] = None
    
    # Add the new column to database using pd.Series
    database[('analysis', 'ROI_trial_mean_dff')] = pd.Series(roi_tuning_data)
    
    print("ROI tuning analysis complete for all sessions!")
    return None

def identify_preferred_stimuli(database: pd.DataFrame):
    """
    Identify the preferred orientation and direction for each ROI based on highest mean dF/F 
    across all blocks for each session.
    
    Parameters
    ----------
    database : pd.DataFrame
        Main database with MultiIndex columns containing ('analysis', 'ROI_trial_mean_dff')
        
    Returns
    -------
    None
        Modifies database in place by adding ('analysis', 'ROI_preferred_stimuli') column
    """
    
    # Initialize storage for all sessions
    preferred_stimuli_data = {}
    
    # Iterate through all subject-session pairs
    for (subject, session) in database.index:
        try:
            # Get trial mean data for this session
            trial_means = database[('analysis', 'ROI_trial_mean_dff')].loc[(subject, session)]
            
            # Skip if no trial means data
            if trial_means is None or trial_means.empty:
                print(f"No trial means data for {subject} {session}, skipping...")
                preferred_stimuli_data[(subject, session)] = None
                continue
            
            # Extract data
            orientations = trial_means['orientations'].values
            directions = trial_means['directions'].values
            grat_responses = np.array([resp for resp in trial_means['grat_resp_all_rois']])
            
            # Get unique stimuli
            unique_orientations = np.unique(orientations)
            unique_directions = np.unique(directions)
            
            # Get number of ROIs
            num_rois = grat_responses.shape[1]
            
            # Initialize results arrays
            preferred_orientations = np.zeros(num_rois)
            preferred_directions = np.zeros(num_rois)
            max_orientation_responses = np.zeros(num_rois)
            max_direction_responses = np.zeros(num_rois)
            orientation_selectivity_indices = np.zeros(num_rois)
            
            # For each ROI, find preferred orientation and direction
            for roi_idx in range(num_rois):
                roi_responses = grat_responses[:, roi_idx]
                
                # Calculate mean response for each orientation
                orientation_means = []
                for ori in unique_orientations:
                    ori_mask = orientations == ori
                    orientation_means.append(roi_responses[ori_mask].mean())
                
                # Calculate OSI using vector average method (Banerjee et al., 2016)
                # OSI = sqrt((sum(R(θi)*sin(2θi))^2 + (sum(R(θi)*cos(2θi))^2)) / sum(R(θi))
                orientation_means = np.array(orientation_means)
                
                # Convert orientations to radians and double the angle for OSI calculation
                theta_rad = np.deg2rad(unique_orientations * 2)  # multiply by 2 for orientation selectivity
                
                # Calculate vector components
                sin_component = np.sum(orientation_means * np.sin(theta_rad))
                cos_component = np.sum(orientation_means * np.cos(theta_rad))
                total_response = np.sum(orientation_means)
                
                # Calculate OSI
                if total_response > 0:
                    osi = np.sqrt(sin_component**2 + cos_component**2) / total_response
                else:
                    osi = 0.0
                
                orientation_selectivity_indices[roi_idx] = osi
                
                # Calculate mean response for each direction
                direction_means = []
                for dir_val in unique_directions:
                    dir_mask = directions == dir_val
                    direction_means.append(roi_responses[dir_mask].mean())
                
                # Find preferred stimuli (highest mean response)
                preferred_ori_idx = np.argmax(orientation_means)
                preferred_dir_idx = np.argmax(direction_means)
                
                preferred_orientations[roi_idx] = unique_orientations[preferred_ori_idx]
                preferred_directions[roi_idx] = unique_directions[preferred_dir_idx]
                max_orientation_responses[roi_idx] = orientation_means[preferred_ori_idx]
                max_direction_responses[roi_idx] = direction_means[preferred_dir_idx]
            
            # Create results DataFrame
            preferred_stimuli = pd.DataFrame({
                'roi_id': np.arange(num_rois),
                'preferred_orientation': preferred_orientations,
                'preferred_direction': preferred_directions,
                'max_orientation_response': max_orientation_responses,
                'max_direction_response': max_direction_responses,
                'orientation_selectivity_index': orientation_selectivity_indices,
                'direction_selectivity': max_direction_responses / (max_direction_responses.mean() + 1e-10)
            })
            
            # Store in dictionary
            preferred_stimuli_data[(subject, session)] = preferred_stimuli
            print(f"Identified preferred stimuli for {num_rois} ROIs in {subject} {session}")
            
        except Exception as e:
            print(f"Error processing preferred stimuli for {subject} {session}: {e}")
            preferred_stimuli_data[(subject, session)] = None
    
    # Add the new column to database using pd.Series
    database[('analysis', 'ROI_preferred_stimuli')] = pd.Series(preferred_stimuli_data)
    
    print("Preferred stimuli identification complete for all sessions!")
    return None

def plot_orientation_tuning(database: pd.DataFrame, subject: str, session: str, roi_id: int, 
                           title: str = None, figsize: tuple = (8, 6)):
    """
    Plot orientation tuning curve for a specific ROI with OSI value displayed.
    
    Parameters
    ----------
    database : pd.DataFrame
        Main database with MultiIndex columns
    subject : str
        Subject identifier (e.g., 'sub-SB03')
    session : str
        Session identifier (e.g., 'ses-01')
    roi_id : int
        ROI index to plot
    title : str, optional
        Custom plot title
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get trial mean data for this session
    trial_means = database[('analysis', 'ROI_trial_mean_dff')].loc[(subject, session)]
    
    if trial_means is None or trial_means.empty:
        raise ValueError(f"No trial means data found for {subject} {session}")
    
    # Get OSI data
    try:
        preferred_stimuli = database[('analysis', 'ROI_preferred_stimuli')].loc[(subject, session)]
        roi_data = preferred_stimuli[preferred_stimuli['roi_id'] == roi_id]
        
        if not roi_data.empty:
            osi = roi_data['orientation_selectivity_index'].iloc[0]
            preferred_ori = roi_data['preferred_orientation'].iloc[0]
        else:
            osi = None
            preferred_ori = None
    except (KeyError, IndexError):
        osi = None
        preferred_ori = None
    
    # Extract data
    orientations = trial_means['orientations'].values
    grat_responses = np.array([resp[roi_id] for resp in trial_means['grat_resp_all_rois']])
    
    # Get unique orientations and calculate mean response for each
    unique_orientations = np.unique(orientations)
    orientation_means = []
    orientation_sems = []
    
    for orientation in unique_orientations:
        orientation_mask = orientations == orientation
        orientation_responses = grat_responses[orientation_mask]
        orientation_means.append(orientation_responses.mean())
        orientation_sems.append(orientation_responses.std() / np.sqrt(len(orientation_responses)))
    
    orientation_means = np.array(orientation_means)
    orientation_sems = np.array(orientation_sems)
    
    # Find peak response and normalize
    peak_response = np.max(orientation_means)
    if peak_response == 0:
        print(f"Warning: Peak response is 0 for ROI {roi_id}")
        normalized_means = orientation_means
        normalized_sems = orientation_sems
    else:
        normalized_means = orientation_means / peak_response
        normalized_sems = orientation_sems / peak_response
    
    # For circular plotting, duplicate the first point at the end
    plot_orientations = np.concatenate([unique_orientations, [unique_orientations[0] + 180]])
    plot_means = np.concatenate([normalized_means, [normalized_means[0]]])
    plot_sems = np.concatenate([normalized_sems, [normalized_sems[0]]])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the main line
    line = ax.plot(plot_orientations, plot_means, 
                   marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Mean response')
    
    # Add shaded region for SEM
    ax.fill_between(plot_orientations, 
                   plot_means - plot_sems,
                   plot_means + plot_sems,
                   alpha=0.3, color='#2E86AB', label='SEM')
    
    # Add reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Peak response')
    
    # Customize the plot
    ax.set_xlabel('Orientation (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Response (peak = 1.0)', fontsize=12)
    ax.set_title(title or f'{subject} {session} - ROI {roi_id} Orientation Tuning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis ticks
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(-10, 190)
    ax.set_ylim(0, 1.1)
    
    # Add OSI value at the top of the plot
    if osi is not None:
        osi_text = f"OSI: {osi:.3f}"
    else:
        osi_text = "OSI: not available"
    
    ax.text(0.5, 0.95, osi_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Annotate preferred orientation if available
    if preferred_ori is not None:
        peak_idx = np.where(unique_orientations == preferred_ori)[0][0]
        ax.annotate(f'Preferred: {preferred_ori:.0f}°', 
                   xy=(preferred_ori, normalized_means[peak_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

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