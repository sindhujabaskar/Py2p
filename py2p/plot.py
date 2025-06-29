import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_trial(df: pd.DataFrame,
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
    df = df['toolkit','trials'][(subject, session)]

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


def plot_block(df: pd.DataFrame,
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
    df = df['toolkit','trials'][(subject, session)]
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

