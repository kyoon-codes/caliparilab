import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy
from tqdm import tqdm
import numpy as np
from scipy.stats import sem
from statsmodels.formula.api import mixedlm
import pingouin as pg
from scipy.integrate import simpson


# -------------------------- ALL FILES --------------------------
# D2 MEDIUM SPINY NEURONS (ALCOHOL)
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHLearning/'
mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321','8729','8730','8731','8732']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_1WeekWD/'
# mice = ['7098','7099','7108','7296', '7311', '7319', '7321','8729','8730','8731','8732']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHExtinction/'
# mice = ['7098','7099','7108','7296', '7311', '7319', '7321','8729','8730','8731','8732']

# D2 MEDIUM SPINY NEURONS (SUCROSE)
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucLearning/'
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucExtinction/'
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# #,'8733','8742','8743','8747','8748','8750']

# D2 MEDIUM SPINY NEURONS (SUCROSE TO ETHANOL TO SUCROSE)
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_EtOHLearning/'
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_AlcExtinction/'
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_SucExtinction2/'
# mice = ['7678', '7680', '7899','8733','8742','8743','8748','8747','8750']


# D1 MEDIUM SPINY NEURONS
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_EtOHLearning/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_1WeekWD/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_EtOHExtinction/'
# mice = ['676', '679', '849', '873', '874', '917']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_SucLearning/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_SucExtinction/'
# mice = ['676', '679', '849', '873', '874', '917']

# -------------------------- PARAMETERS --------------------------
events = ['Cue', 'Press', 'Licks', 'Timeout Press']
epocs = ['Po0_', 'Po6_', 'Po4_', 'Po2_']

timerange_cue = [-2, 5]
timerange_lever = [-2, 5]
timerange_lick = [-2, 10]
active_time = [-2,40]
N = 100
max_interval = 0.5  # Example max interval for lick bouts, adjust as needed

# -------------------------- INIT STORAGE --------------------------
all_cue_trials = []
all_lever_trials = []
all_firstlick_trials = []
all_firstlick_cue_aligned = []
all_lickbouts = []
all_trials = []
lickspersession = {}
responsepersession = {}
timeoutpersession = {}
# -------------------------- HELPER FUNCTIONS --------------------------
def extract_events(data, events, epocs):
    split1 = str(data.epocs).split('\t')
    z = [item for sublist in [e.split('\n') for e in split1] for item in sublist if item != '[struct]']
    
    event_frames = []
    for a, b in zip(events, epocs):
        if b in z:
            timestamps = getattr(data.epocs[b], 'onset', [])
            if len(timestamps) > 0:
                event_frames.append(pd.DataFrame({'Event':[a]*len(timestamps), 'Timestamp':timestamps}))
    if event_frames:
        return pd.concat(event_frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Event','Timestamp'])

def downsample_trial(trial, N):
    trial = np.array(trial)
    trim = len(trial) % N
    trial = trial[:len(trial)-trim] if trim != 0 else trial
    return trial.reshape(-1, N).mean(axis=1)

def identify_firstlicks(track_lever, track_licks):
    firstlicks = []
    for press in track_lever:
        post_licks = np.array(track_licks) > press
        indices = np.where(post_licks)[0]
        if len(indices) > 0:
            firstlicks.append(indices[0])
    return list(set(firstlicks))

def identify_and_sort_lick_bouts(lick_timestamps, max_interval):
    bouts = []
    current_bout = []
    for i in range(len(lick_timestamps)):
        if not current_bout:
            current_bout.append(lick_timestamps[i])
        else:
            if lick_timestamps[i] - current_bout[-1] <= max_interval:
                current_bout.append(lick_timestamps[i])
            else:
                bouts.append(current_bout)
                current_bout = [lick_timestamps[i]]
    if current_bout:
        bouts.append(current_bout)
    # Calculate the length of each bout
    bout_lengths = [(len(bout), bout) for bout in bouts]
    # Sort bouts by their length
    sorted_bouts = sorted(bout_lengths, key=lambda x: x[0])
    return sorted_bouts

# -------------------------- MAIN LOOP --------------------------
for mouse in tqdm(mice, desc="Processing mice"):
    mouse_dir = os.path.join(folder, mouse)
    dates = sorted([x for x in os.listdir(mouse_dir) if x.isnumeric()])

    # Map date → session index (starting at 0)
    session_map = {date: i for i, date in enumerate(dates)}

    for date in dates:
        session_idx = session_map[date]
        date_dir = os.path.join(mouse_dir, date)
        data = tdt.read_block(date_dir)

        # ------------------- SIGNAL PROCESSING -------------------
        sig405 = data.streams._405B.data
        sig465 = data.streams._465B.data[:len(sig405)]
        dff = (sig465 - sig405) / sig465
        fs = round(data.streams._465B.fs)

        df = pd.DataFrame({'Sig405': sig405, 'Sig465': sig465, 'Dff': dff})

        # ------------------- EXTRACT EVENTS -------------------
        fp_df = extract_events(data, events, epocs)

        track_cue = fp_df.loc[fp_df['Event'] == 'Cue', 'Timestamp'].to_numpy()
        track_lever = fp_df.loc[fp_df['Event'] == 'Press', 'Timestamp'].to_numpy()
        track_licks = fp_df.loc[fp_df['Event'] == 'Licks', 'Timestamp'].to_numpy()
        track_to = fp_df.loc[fp_df['Event'] == 'Timeout Press', 'Timestamp'].to_numpy()

        # ------------------- SESSION COUNTS -------------------
        if len(track_cue) > 10:
            totallicks = track_licks[track_licks < track_cue[10]]
            totalresponse = track_lever[track_lever < track_cue[10]]
            totaltimeout = track_to[track_to < track_cue[10]]
        else:
            totallicks = track_licks
            totalresponse = track_lever
            totaltimeout = track_to

        lickspersession[mouse, session_idx] = len(totallicks)
        responsepersession[mouse, session_idx] = len(totalresponse)
        timeoutpersession[mouse, session_idx] = len(totaltimeout)

        # ------------------- CUE ALIGNMENT -------------------
        cue_baselines = []
        for trial_num, cue_time in enumerate(track_cue):
            cue_zero = round(cue_time * fs)
            cue_baseline = cue_zero + timerange_cue[0] * fs
            cue_end = cue_zero + timerange_cue[1] * fs

            baseline_mean = np.mean(df['Dff'].iloc[cue_baseline:cue_zero])
            baseline_std = np.std(df['Dff'].iloc[cue_baseline:cue_zero])
            cue_baselines.append((baseline_mean, baseline_std))

            trial_signal = (df['Dff'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            all_cue_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'Trace': sampletrial,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
            })

        # ------------------- LEVER ALIGNMENT -------------------
        for lever_time in track_lever:
            cue_trial_idx = np.where((lever_time - track_cue > 0) & (lever_time - track_cue < 20))[0]
            if len(cue_trial_idx) == 0:
                continue

            cue_trial = cue_trial_idx[0]
            baseline_mean, baseline_std = cue_baselines[cue_trial]  # use corresponding trial baseline

            lever_zero = round(lever_time * fs)
            lever_baseline = lever_zero + timerange_lever[0] * fs
            lever_end = lever_zero + timerange_lever[1] * fs
            trial_signal = (df['Dff'].iloc[lever_baseline:lever_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            all_lever_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'LeverTime': lever_time,
                'Trace': sampletrial
            })

        # ------------------- FIRST LICK ALIGNMENT -------------------
        firstlick_indices = identify_firstlicks(track_lever, track_licks)
        track_flicks = track_licks[firstlick_indices]

        for flick_time in track_flicks:
            cue_trial_idx = np.where((flick_time - track_cue > 0) & (flick_time - track_cue < 30))[0]
            if len(cue_trial_idx) == 0:
                continue

            cue_trial = cue_trial_idx[0]
            baseline_mean, baseline_std = cue_baselines[cue_trial]

            flick_zero = round(flick_time * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs

            trial_signal = (df['Dff'].iloc[flick_baseline:flick_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            all_firstlick_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'FlickTime': flick_time,
                'Trace': sampletrial
            })

        # ------------------- LICK BOUT ALIGNMENT -------------------
        max_interval = 0.4
        sorted_lick_bouts = identify_and_sort_lick_bouts(track_licks, max_interval)
        for bout_len, bout in sorted_lick_bouts:
            start_time = bout[0]

            # find the cue trial this lick bout belongs to
            cue_trial_idx = np.where((start_time - track_cue > 0) & (start_time - track_cue < 30))[0]
            if len(cue_trial_idx) == 0:
                continue

            cue_trial = cue_trial_idx[0]
            baseline_mean, baseline_std = cue_baselines[cue_trial]

            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero + timerange_lick[0] * fs
            lickb_end = lickb_zero + timerange_lick[1] * fs

            trial_signal = (df['Dff'].iloc[lickb_baseline:lickb_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            all_lickbouts.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'BoutLength': bout_len,
                'StartTime': start_time,
                'Trace': sampletrial
            })

        # ------------------- ALL TRIAL ALIGNMENT -------------------
        for trial_num, cue_time in enumerate(track_cue):
            baseline_mean, baseline_std = cue_baselines[trial_num]

            cue_zero = round(cue_time * fs)
            cue_baseline = cue_zero + active_time[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            trial_signal = (df['Dff'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            lever_times = track_lever[(track_lever - cue_time > 0) & (track_lever - cue_time < 20)]
            lever_time = lever_times[0] if len(lever_times) > 0 else np.nan

            flick_times = track_flicks[(track_flicks - cue_time > 0) & (track_flicks - cue_time < 30)]
            flick_time = flick_times[0] if len(flick_times) > 0 else np.nan

            all_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'LeverTime': lever_time,
                'FlickTime': flick_time,
                'Trace': sampletrial
            })

# -------------------------- CONVERT TO DATAFRAMES --------------------------
avgcuetrace_df = pd.DataFrame(all_cue_trials)
avglevertrace_df = pd.DataFrame(all_lever_trials)
avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
avglickbouttrace_df = pd.DataFrame(all_lickbouts)
alltrialtrace_df = pd.DataFrame(all_trials)

# ----------------------------------------------------------------------------
# ------------------------- NOW THE FUN STUFF!!!!!!! -------------------------
# ----------------------------------------------------------------------------

colors = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen',
          'mediumseagreen', 'mediumturquoise', 'deepskyblue',
          'dodgerblue', 'slateblue', 'darkorchid', 'purple']

# ------------------------- Ploting Traces -------------------------

def plot_traces_bysession(df, label,timerange, ylim, savepath):
    session_list = sorted(df['Session'].unique())
    plt.figure(figsize=(10, 8))
    for i, session in enumerate(session_list):
        traces = np.stack(df.loc[df['Session']==session, 'Trace'])
        mean_trace = traces.mean(axis=0)
        sem_trace = sem(traces)
        plt.plot(mean_trace, color=colors[i % len(colors)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)),
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[i % len(colors)], alpha=0.1)
    
    plt.xlabel('Time (samples)')
    plt.xticks(np.arange(0, len(mean_trace)+1,
                         len(mean_trace)/(timerange[1]-timerange[0])),
               np.arange(timerange[0], timerange[1]+1, 1, dtype=int))
    plt.axhline(y=0, linestyle=':', color='black')
    plt.axvline(x=len(mean_trace)/(timerange[1]-timerange[0])*(0-timerange[0]),
                linewidth=1, color='black')
    plt.ylabel('z-score')
    plt.title(f'Average {label}-Aligned Trace with SEM by Session')
    plt.legend()
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.show()

plot_traces_bysession(avgcuetrace_df, 'Cue', timerange_cue, (-1,3), '/Users/kristineyoon/Documents/cuebysessions.pdf')
plot_traces_bysession(avglevertrace_df, 'Lever', timerange_lever, (-3,6), '/Users/kristineyoon/Documents/leverbysessions.pdf')
plot_traces_bysession(avgflicktrace_df, 'First Lick', timerange_lick, (-3,9), '/Users/kristineyoon/Documents/firstlickbysessions.pdf')


# ------------------------- Finding Peak Height  -------------------------

analysis_window = [0, 2]  # time window in seconds
totalsessions = len(sorted(avgcuetrace_df['Session'].unique()))


def compute_peak_df(df, timerange, label):
    """Compute peak height in 0–2 s window for traces in df."""
    results = []

    for _, row in df.iterrows():
        mouse, session, trial, trace = row["Mouse"], row["Session"], row["Trial"], row["Trace"]
        time = np.linspace(timerange[0], timerange[1], len(trace))

        start_idx = np.searchsorted(time, analysis_window[0])
        end_idx = np.searchsorted(time, analysis_window[1])

        trace_segment = trace[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        peakheight = np.max(trace_segment)
        peak_idx = np.argmax(trace_segment)
        peak_time = time_segment[peak_idx]

        results.append({
            "Alignment": label,
            "Mouse": mouse,
            "Session": session,
            "Trial": trial,
            "PeakHeight": peakheight,
            "PeakTime": peak_time})
    return pd.DataFrame(results)


peak_cue_df = compute_peak_df(avgcuetrace_df, timerange_cue, "Cue")
peak_lever_df = compute_peak_df(avglevertrace_df, timerange_lever, "Lever")
peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, "FirstLick")

# Combine all
peak_all_df = pd.concat([peak_cue_df, peak_lever_df, peak_flick_df], ignore_index=True)

# ---------------------- Plotting Peak Height ----------------------

def plot_peak_by_session(df, label, ylabel, ylim, savepath):
    session_list = sorted(df['Session'].unique())
    plt.figure(figsize=(5,6))

    for i, session in enumerate(session_list):
        session_data = df.loc[df["Session"] == session, "PeakHeight"]

        if not session_data.empty:
            plt.scatter([session]*len(session_data), session_data,
                        color=colors[i % len(colors)], alpha=0.05)
            mean_val = session_data.mean()
            err_val = sem(session_data)
            plt.scatter(session, mean_val, color=colors[i % len(colors)], s=40, label=f'Session {session}')
            plt.errorbar(session, mean_val, yerr=err_val,
                         ecolor=colors[i % len(colors)], capsize=3)

    plt.xlabel('Session')
    plt.ylabel(ylabel)
    plt.title(f'{label}-Aligned Peak Height (0–2 s)')
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.show()

plot_peak_by_session(peak_cue_df, 'Cue', 'Peak Height (z)', (-2, 8),'/Users/kristineyoon/Documents/peakheight_cue.pdf')
plot_peak_by_session(peak_lever_df, 'Lever', 'Peak Height (z)', (-2, 10),'/Users/kristineyoon/Documents/peakheight_lever.pdf')
plot_peak_by_session(peak_flick_df, 'First Lick', 'Peak Height (z)', (-2, 12),'/Users/kristineyoon/Documents/peakheight_firstlick.pdf')


# ---------------------- Finding Area Under the Curve (AUC) ----------------------
def compute_auc(df, timerange, interval, trial_limit=10):
    """
    Compute AUC for each trial in df over the given interval.
    
    df: DataFrame with columns ['Mouse', 'Session', 'Trial', 'Trace']
    timerange: tuple (start, end) of full trace
    interval: tuple (start, end) of AUC interval
    trial_limit: only include trials less than this number
    """
    results = []

    for _, row in df.iterrows():
        if row['Trial'] < trial_limit:
            trace = row['Trace']
            time = np.linspace(timerange[0], timerange[1], len(trace))
            
            start_idx = np.searchsorted(time, interval[0])
            end_idx = np.searchsorted(time, interval[1])
            
            trace_segment = trace[start_idx:end_idx]
            time_segment = time[start_idx:end_idx]
            
            area = simpson(trace_segment, time_segment)
            
            results.append({
                'Mouse': row['Mouse'],
                'Session': row['Session'],
                'Trial': row['Trial'],
                'AUC': area
            })
    
    return pd.DataFrame(results)

# Compute AUC DataFrames
auc_cue_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(0, 2))
auc_lever_df = compute_auc(avglevertrace_df, timerange_lever, interval=(-1, 1))
auc_flick_df = compute_auc(avgflicktrace_df, timerange_lick, interval=(0, 2))

# ---------------------- Plotting Area Under the Curve (AUC) ----------------------
def plot_auc(df, ylabel, ylim):
    session_list = len(sorted(df['Session'].unique()))
    plt.figure(figsize=(5, 6))
    for session in range(session_list):
        session_data = df[df['Session'] == session]['AUC'].values
        if len(session_data) > 0:
            plt.scatter([session]*len(session_data), session_data,
                        color=colors[session % len(colors)], alpha=0.05)
            mean_val = session_data.mean()
            err_val = sem(session_data)
            plt.scatter(session, mean_val, color=colors[session % len(colors)])
            plt.errorbar(session, mean_val, yerr=err_val, color=colors[session % len(colors)],
                         capsize=3)
    plt.xlabel('Session')
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Plot AUC for each type
plot_auc(auc_cue_df, ylabel='Cue AUC', ylim=(-5,10))
plot_auc(auc_lever_df, ylabel='Lever AUC', ylim=(-5,10))
plot_auc(auc_flick_df, ylabel='First Lick AUC',ylim=(-5,20))




















