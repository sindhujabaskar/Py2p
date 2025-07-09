import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    Plot the concatenated average dF/F trace for all trials in a block.

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
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    # hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{subject} {session} — ROI tuning (polar)", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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

