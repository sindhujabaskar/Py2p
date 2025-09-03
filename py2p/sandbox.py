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