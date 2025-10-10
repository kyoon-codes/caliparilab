import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import sem, kruskal
from scipy.integrate import simpson

#NONCONTINGENT
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHNonconting2'
mice = ['8729', '8730', '8731','8732','8733','8742','8743','8747','8748','8750']

# -------------------------- PARAMETERS --------------------------
events = ['Cue', 'Licks']
epocs = ['Po0_', 'Po4_']

timerange_cue = [-2, 10]
timerange_lick = [-5, 15]
active_time = [-2,40]
N = 50
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
        track_licks = fp_df.loc[fp_df['Event'] == 'Licks', 'Timestamp'].to_numpy()

        # ------------------- SESSION COUNTS -------------------
        if len(track_cue) > 10:
            totallicks = track_licks[track_licks < track_cue[10]]
            track_cue = track_cue[:10]
        else:
            totallicks = track_licks
        lickspersession[mouse, session_idx] = len(totallicks)

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

        # ------------------- FIRST LICK ALIGNMENT -------------------
        firstlick_indices = identify_firstlicks(track_cue, track_licks)
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
                'Trace': sampletrial,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
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
                'Trace': sampletrial,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
            })

        # ------------------- ALL TRIAL ALIGNMENT -------------------
        for trial_num, cue_time in enumerate(track_cue):
            baseline_mean, baseline_std = cue_baselines[trial_num]

            cue_zero = round(cue_time * fs)
            cue_baseline = cue_zero + active_time[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            trial_signal = (df['Dff'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std
            sampletrial = downsample_trial(trial_signal, N)

            flick_times = track_flicks[(track_flicks - cue_time > 0) & (track_flicks - cue_time < 30)]
            flick_time = flick_times[0] if len(flick_times) > 0 else np.nan

            all_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'FlickTime': flick_time,
                'Trace': sampletrial,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
            })

# -------------------------- CONVERT TO DATAFRAMES --------------------------
avgcuetrace_df = pd.DataFrame(all_cue_trials)
avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
avglickbouttrace_df = pd.DataFrame(all_lickbouts)
alltrialtrace_df = pd.DataFrame(all_trials)



colors = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen',
          'mediumseagreen', 'mediumturquoise', 'deepskyblue',
          'dodgerblue', 'slateblue', 'darkorchid', 'purple']

# ------------------------- Plotting Traces -------------------------

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

plot_traces_bysession(avgcuetrace_df, 'Cue', timerange_cue, (-1,4), '/Users/kristineyoon/Documents/cuebysessions.pdf')
plot_traces_bysession(avgflicktrace_df, 'First Lick', timerange_lick, (-5,9), '/Users/kristineyoon/Documents/firstlickbysessions.pdf')


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
peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, "FirstLick")

# Combine all
peak_all_df = pd.concat([peak_cue_df, peak_flick_df], ignore_index=True)

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

# Compute AUC DataFrames
auc_cue_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(0, 1))
auc_flick_df = compute_auc(avgflicktrace_df, timerange_lick, interval=(0, 2))
# Plot AUC for each type
plot_auc(auc_cue_df, ylabel='Cue AUC', ylim=(-5,10))
plot_auc(auc_flick_df, ylabel='First Lick AUC',ylim=(-5,20))


# -------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------
colors10 = sns.color_palette("husl", 10)

# -------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------
def categorize_lick_bout(bout_length):
    if bout_length < 20:
        return 0
    elif bout_length < 40:
        return 1
    elif bout_length < 60:
        return 2
    else:
        return 3


# ---------------------- PEAK HEIGHT & TIME TO BASELINE: LICK BOUTS ---------------------- 

# Categorize bouts
avglickbouttrace_df['BoutCategory'] = avglickbouttrace_df['BoutLength'].apply(categorize_lick_bout)

# Compute metrics
time_to_baseline_dict = {}
peakheight_dict = {}

for _, row in avglickbouttrace_df.iterrows():
    mouse, session, trial, boutlength, baseline = row['Mouse'], row['Session'], row['Trial'], row['BoutLength'], row['BaselineMean']
    trace = np.array(row['Trace'])
    time = np.linspace(timerange_lick[0], timerange_lick[1], len(trace))
    
    # Focus on post-event period
    start_time, end_time = 0, timerange_lick[1]
    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)
    trace_segment = trace[start_idx:end_idx]
    time_segment = time[start_idx:end_idx]
    
    # Boolean array: True when trace > baseline
    above = trace_segment > baseline
    #above = trace_segment > 0

    # Detect transitions
    transitions = np.diff(above.astype(int))

    # +1 → rising through baseline (below → above)
    # -1 → falling through baseline (above → below)
    rise_indices = np.where(transitions == 1)[0]
    fall_indices = np.where(transitions == -1)[0]

    baselinetime = np.nan  # default in case no valid crossing

    if len(rise_indices) > 0:
        if above[0]:
            # Trace starts above baseline
            if len(fall_indices) > 0:
                # Find first fall, then look for next rise *after* that fall
                first_fall = fall_indices[0]
                later_rises = rise_indices[rise_indices > first_fall]
                if len(later_rises) > 0:
                    baselinetime = time_segment[later_rises[0]]
            else:
                # Stays above baseline the whole time
                baselinetime = np.nan #time_segment[0]
        else:
            # Trace starts below baseline — take first rise
            baselinetime = np.nan #time_segment[rise_indices[0]]

    # Store result
    time_to_baseline_dict[(mouse, session, trial, boutlength)] = baselinetime

    
    # Peak height
    trace = np.array(row['Trace'])
    time = np.linspace(timerange_lick[0], timerange_lick[1], len(trace))
    start_time, end_time = 0, 2
    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)
    trace_segment = trace[start_idx:end_idx]
    time_segment = time[start_idx:end_idx]
    peak_val = trace_segment.max()
    peakheight_dict[(mouse, session, trial, boutlength,)] = peak_val

# Convert to DataFrames
time_to_baseline_df = pd.DataFrame([
    {'Mouse': k[0], 'Session': k[1], 'Trial': k[2], 'BoutLength': k[3], 'TimeToBaseline': v}
    for k, v in time_to_baseline_dict.items()
])
peakheight_df = pd.DataFrame([
    {'Mouse': k[0], 'Session': k[1], 'Trial': k[2], 'BoutLength': k[3], 'PeakHeight': v}
    for k, v in peakheight_dict.items()
])

# -------------------------------------------------------------
# 5. VISUALIZATION
# -------------------------------------------------------------
# Time to baseline by sessions
plt.figure(figsize=(5, 6))
for session in sorted(time_to_baseline_df['Session'].unique()):
    # Get all time-to-baseline values for this session
    session_values = time_to_baseline_df[time_to_baseline_df['Session'] == session]['TimeToBaseline'].values
    clean_data = np.array(session_values)[~np.isnan(session_values)]
    
    # Scatter individual points (slightly jittered for visibility)
    jitter = np.random.normal(0, 0.05, size=len(clean_data))
    plt.scatter(np.full(len(clean_data), session) + jitter, clean_data,
                color=colors10[session], alpha=0.1, label='_nolegend_')
    
    # Plot mean ± SEM
    if len(clean_data) > 0:
        plt.errorbar(session, np.nanmean(clean_data), 
                     yerr=sem(clean_data, nan_policy='omit'),
                     fmt='o', color=colors10[session], capsize=4, markersize=6)

plt.xlabel('Session', fontsize=12)
plt.ylabel('Time to Baseline (s)', fontsize=12)
plt.title(f'Time to Baseline in Noncontingenet EtOH by Session', fontsize=14)
plt.tight_layout()
plt.show()


# Time to baseline by categories
plt.figure(figsize=(10,6))
for cat in sorted(avglickbouttrace_df['BoutCategory'].unique()):
    cat_data = time_to_baseline_df[time_to_baseline_df['BoutLength'].apply(categorize_lick_bout) == cat]
    for session in sorted(cat_data['Session'].unique()):
        session_values = cat_data[cat_data['Session'] == session]['TimeToBaseline'].values
        clean_data = np.array(session_values)[~np.isnan(session_values)]
        plt.scatter([session]*len(session_values)+cat/5-0.1, session_values,
                       color=colors10[cat], alpha=0.1)
        plt.errorbar(session+cat/5-0.1, clean_data.mean(), yerr=sem(clean_data),fmt='o',
                        color=colors10[cat], capsize=3)
plt.xlabel('Session')
plt.ylabel('Time to Baseline (s)')
plt.title(f'Time to Baseline in Noncontingenet EtOH by Bout Category')
plt.tight_layout()
plt.show()

# Peak height
plt.figure(figsize=(10,6))
for cat in sorted(avglickbouttrace_df['BoutCategory'].unique()):
    cat_data = peakheight_df[peakheight_df['BoutLength'].apply(categorize_lick_bout) == cat]
    for session in sorted(cat_data['Session'].unique()):
        session_values = cat_data[cat_data['Session'] == session]['PeakHeight'].values
        plt.scatter([session]*len(session_values)+cat/5-0.1, session_values,
                       color=colors10[cat], alpha=0.05)
        plt.errorbar(session+cat/5-0.1, session_values.mean(), yerr=sem(session_values),fmt='o',
                        color=colors10[cat], capsize=3)
plt.xlabel('Session')
plt.ylabel('Peak Height (z-score)')
plt.title(f'Peak Height in Noncontingenet EtOH by Bout Category')
plt.tight_layout()
plt.ylim(-10,20)
plt.show()

# Merge DataFrames on common columns
combined_df = pd.merge(time_to_baseline_df, peakheight_df, 
                        on=['Mouse', 'Session', 'Trial', 'BoutLength'])
plt.figure(figsize=(10, 6))

# Create scatter plot with line of best fit
for session in sorted(combined_df['Session'].unique()):
    session_data = combined_df[combined_df['Session'] == session]
    
    # Scatter plot
    sns.scatterplot(data=session_data, 
                    x='PeakHeight', 
                    y='BoutLength', 
                    color=colors10[session], 
                    label=f'Session {session}', 
                    alpha=0.5)
    
    # Line of best fit
    sns.regplot(data=session_data,
                x='PeakHeight', 
                y='BoutLength', 
                scatter=False,  # Do not plot points again
                color=colors10[session], 
                line_kws={"alpha": 0.7, "lw": 2})
    
# Add labels and title
plt.xlabel('Peak Height (z-score)')
plt.ylabel('Time to Baseline (s)')
plt.title('Lick Bouts: Bout Length vs Peak Height with Best Fit Line')
plt.legend(title='Session') #, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import statsmodels.api as sm

# Loop over each session to perform linear regression
for session in sorted(combined_df['Session'].unique()):
    session_data = combined_df[combined_df['Session'] == session]
    
    # Prepare the data
    X = session_data['PeakHeight']
    y = session_data['BoutLength']
    
    # Add a constant to our predictor
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X, missing='drop').fit()  # Drop any missing values
    
    # Print the statistics
    print(f'Session {session}')
    print(model.summary())
    print('\n' + '='*50 + '\n')
