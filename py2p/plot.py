import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

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

def plot_roi_tuning_polar(database = database, subject, session, roi_idx,
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
     import numpy as np
     import matplotlib.pyplot as plt

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
# %%
# plt.plot(database.toolkit.timestamps['sub-SB03','ses-01'],database.calculate.interp_deltaf_f['sub-SB03','ses-01'][77])
def plot_onething(data):
    plt.figure(figsize=(10,5))
    plt.plot(data)
    plt.title("Sample Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    return plt

def plot_twothings(data, data2):
    plt.subplots(1, 2, figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(data)
    plt.title("Sample Plot 1")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.subplot(1, 2, 2)
    plt.plot(data2)
    plt.title("Sample Plot 2")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    return plt




