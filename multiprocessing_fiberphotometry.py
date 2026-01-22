import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import sem, kruskal
from scipy.integrate import simpson

# -------------------------- ALL FILES --------------------------

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'


# ------- D2 MEDIUM SPINY NEURONS (ALCOHOL) -------
mice = ['7098','7099','7107','7108', '7310', '7311', '7319', '7321','8729','8730','8731','8732','9299','9302','9325','9326','9327'] #'7296',
# mice = ['9299','9302','9325','9326','9327']
experiment = 'D2_EtOHLearning'
# experiment = 'D2_1WeekWD'

# mice = ['7098','7099','7108', '7311', '7319', '7321','8729','8730','8731','8732'] #'7296',
# experiment = 'D2_EtOHExtinction'

# ------- D2 MEDIUM SPINY NEURONS (SUCROSE) -------

mice = ['7678', '7680', '8733','8742','8743','8747','8748','8750']
experiment = 'D2_SucLearning'
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# experiment = 'D2_SucExtinction'
# experiment = 'D2_SuctoEtOH_EtOHLearning'
# experiment = 'D2_SuctoEtOH_AlcExtinction'
# experiment = 'D2_SuctoEtOH_SucExtinction2'


# ------- D1 MEDIUM SPINY NEURONS -------
# mice = ['676', '679', '849', '873', '874', '917']
# experiment = 'D1_EtOHLearning'
# experiment = 'D1_1WeekWD'
# experiment = 'D1_EtOHExtinction'
# experiment = 'D1_SucLearning'
# experiment = 'D1_SucExtinction'

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

# def downsample_trial(trial, N):
#     trial = np.array(trial)
#     trim = len(trial) % N
#     trial = trial[:len(trial)-trim] if trim != 0 else trial
#     return trial.reshape(-1, N).mean(axis=1)

def identify_firstlicks(track_lever, track_licks):
    firstlicks = []
    alllick = []
    for i, press in enumerate(track_lever):
        if i < len(track_lever)-1:
            post_licks = (np.array(track_licks) > press) & (np.array(track_licks) < track_lever[i+1])
            indices = np.where(post_licks)[0]
            if len(indices) > 0:
                firstlicks.append(indices[0])
                alllick.append(len(indices))
        elif i == len(track_lever)-1:
            post_licks = np.array(track_licks) > press
            indices = np.where(post_licks)[0]
            if len(indices) > 0:
                firstlicks.append(indices[0])
                alllick.append(len(indices))
    return firstlicks, alllick

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

import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filtfilt(data, fs, cutoff=3, order=4):
    """
    Zero-phase low-pass filter using filtfilt.

    Parameters
    ----------
    data : array-like
        1D signal to filter
    fs : float
        Sampling frequency (Hz)
    cutoff : float
        Low-pass cutoff frequency (Hz)
    order : int
        Butterworth filter order

    Returns
    -------
    filtered_data : ndarray
        Filtered signal
    """

    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype='low')
    filtered_data = filtfilt(b, a, data)

    return filtered_data

# -------------------------- PARAMETERS --------------------------
events = ['Cue', 'Press', 'Licks', 'Timeout Press']
epocs = ['Po0_', 'Po6_', 'Po4_', 'Po2_']

timerange_cue = [-2, 5]
timerange_lever = [-2, 5]
timerange_lick = [-2, 10]
active_time = [-2,30]
N = 100
max_interval = 0.5  # Example max interval for lick bouts, adjust as needed

# -------------------------- INIT STORAGE --------------------------
all_cue_trials = []
all_iti_trials = []
all_lever_trials = []
all_firstlick_trials = []
all_firstlick_cue_aligned = []
all_lickbouts = []
all_trials = []
lickspersession = []
responsepersession = []
timeoutpersession = []

# -------------------------- MAIN LOOP --------------------------
for mouse in tqdm(mice, desc="Processing mice"):
    mouse_dir = os.path.join(folder, experiment, mouse)
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
        dff = (sig465 - sig405) # / sig465
        fs = round(data.streams._465B.fs)
        filtered_trace = lowpass_filtfilt(dff, fs)
       
        df = pd.DataFrame({'Sig405': sig405, 'Sig465': sig465, 'Dff': dff, 'Filtered': filtered_trace})

        # ------------------- EXTRACT EVENTS -------------------
        fp_df = extract_events(data, events, epocs)

        track_cue = fp_df.loc[fp_df['Event'] == 'Cue', 'Timestamp'].to_numpy()
        track_lever = fp_df.loc[fp_df['Event'] == 'Press', 'Timestamp'].to_numpy()
        track_licks = fp_df.loc[fp_df['Event'] == 'Licks', 'Timestamp'].to_numpy()
        track_to = fp_df.loc[fp_df['Event'] == 'Timeout Press', 'Timestamp'].to_numpy()
        
        # plt.figure(figsize=(16,6))
        # plt.plot(dff)
        # plt.eventplot(track_cue*fs, colors = 'gold')
        # plt.eventplot(track_lever*fs, colors = 'blueviolet')
        # plt.eventplot(track_licks*fs, colors = 'orchid')
        # plt.eventplot(track_to*fs, colors = 'red')
        # #plt.ylim(-2,2)
        # plt.title(f'Mouse: {mouse} Date: {date}')
        # plt.show()
        

        # filtered_trace = lowpass_filtfilt(dff, fs)
        # plt.figure(figsize=(16,6))
        # plt.plot(filtered_trace)
        # plt.eventplot(track_cue*fs, colors = 'gold')
        # plt.eventplot(track_lever*fs, colors = 'blueviolet')
        # plt.eventplot(track_licks*fs, colors = 'orchid')
        # plt.eventplot(track_to*fs, colors = 'red')
        # #plt.ylim(-2,2)
        # plt.title(f'Mouse: {mouse} Date: {date}')
        # plt.show()

        # ------------------- SESSION COUNTS -------------------
        if len(track_cue) > 10:
            totallicks = track_licks[track_licks < track_cue[10]]
            totalresponse = track_lever[track_lever < track_cue[10]]
            totaltimeout = track_to[track_to < track_cue[10]]
            track_cue = track_cue[:10]
        else:
            totallicks = track_licks
            totalresponse = track_lever
            totaltimeout = track_to
        
        lickspersession.append({'Mouse': mouse, 'Session:': session_idx, 'Licks':len(totallicks)})
        responsepersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totalresponse)})
        timeoutpersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totaltimeout)})

        # ------------------- CUE ALIGNMENT -------------------
        cue_baselines = []
        for trial_num, cue_time in enumerate(track_cue):
            cue_zero = round(cue_time * fs)
            cue_baseline = cue_zero + timerange_cue[0] * fs
            cue_end = cue_zero + timerange_cue[1] * fs

            baseline_mean = np.mean(df['Filtered'].iloc[cue_baseline:cue_zero])
            baseline_std = np.std(df['Filtered'].iloc[cue_baseline:cue_zero])
            cue_baselines.append((baseline_mean, baseline_std))

            trial_signal = (df['Filtered'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std

            all_cue_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'Trace': trial_signal,
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
            trial_signal = (df['Filtered'].iloc[lever_baseline:lever_end] - baseline_mean) / baseline_std
            
            all_lever_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'LeverTime': lever_time,
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
            })

        # ------------------- FIRST LICK ALIGNMENT -------------------
        firstlick_indices, licks = identify_firstlicks(track_lever, track_licks)
        track_flicks = track_licks[firstlick_indices]

        for i,flick_time in enumerate(track_flicks):
            cue_trial_idx = np.where((flick_time - track_cue > 0) & (flick_time - track_cue < 30))[0]
            if len(cue_trial_idx) == 0:
                continue

            cue_trial = cue_trial_idx[0]
            baseline_mean, baseline_std = cue_baselines[cue_trial]

            flick_zero = round(flick_time * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs

            trial_signal = (df['Filtered'].iloc[flick_baseline:flick_end] - baseline_mean) / baseline_std
            
            all_firstlick_trials.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'FlickTime': flick_time,
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std,
                'Licks': licks[i]
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
            
            # baseline_mean = np.mean(df['Dff'].iloc[lickb_baseline:lickb_zero])
            # baseline_std = np.std(df['Dff'].iloc[lickb_baseline:lickb_zero])

            trial_signal = (df['Filtered'].iloc[lickb_baseline:lickb_end] - baseline_mean) / baseline_std

            all_lickbouts.append({
                'Mouse': mouse,
                'Session': session_idx,
                'Trial': cue_trial,
                'BoutLength': bout_len,
                'StartTime': start_time,
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std
            })

        # ------------------- ALL TRIAL ALIGNMENT -------------------
        for trial_num, cue_time in enumerate(track_cue):
            #baseline_mean, baseline_std = cue_baselines[trial_num]

            cue_zero = round(cue_time * fs)
            cue_baseline = cue_zero + active_time[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            
            baseline_mean = np.mean(df['Filtered'].iloc[cue_baseline:cue_end])
            baseline_std = np.std(df['Filtered'].iloc[cue_baseline:cue_end])
            
            trial_signal = (df['Filtered'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std


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
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std,
                'Latency': flick_time-cue_time
            })
        # ------------------- ITI ALIGNMENT -------------------
        timerange_iti = [-40,-10]
        for trial_num, cue_time in enumerate(track_cue):
            if trial_num > 0:
                #baseline_mean, baseline_std = cue_baselines[trial_num-1]
                cue_zero = round(cue_time * fs)
                cue_baseline = cue_zero + timerange_iti[0] * fs
                cue_end = cue_zero + timerange_iti[1] * fs
                baseline_mean = np.mean(df['Filtered'].iloc[cue_baseline:cue_end])
                baseline_std = np.std(df['Filtered'].iloc[cue_baseline:cue_end])
                trial_signal = (df['Filtered'].iloc[cue_baseline:cue_end] - baseline_mean) / baseline_std

                find_to = track_to*fs
                trial_to = []
                
                for tos in find_to:
                    if cue_baseline <= tos <= cue_end:
                        trial_to.append((tos-cue_baseline)/fs)
                
    
                all_iti_trials.append({
                    'Mouse': mouse,
                    'Session': session_idx,
                    'Trial': trial_num,
                    'CueTime': cue_time,
                    'Trace': trial_signal,
                    'TimeOuts': trial_to,
                    'TimeoutLength': len(trial_to)
                })
# -------------------------- CONVERT TO DATAFRAMES --------------------------
avgcuetrace_df = pd.DataFrame(all_cue_trials)
avglevertrace_df = pd.DataFrame(all_lever_trials)
avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
avglickbouttrace_df = pd.DataFrame(all_lickbouts)
alltrialtrace_df = pd.DataFrame(all_trials)
allititrace_df = pd.DataFrame(all_iti_trials)


# ------------------------- Plotting Cue Trace By Individual Trials -------------------------
colors = sns.color_palette("husl", 10)
timerange = [-2,5]
session_list = [0]
plt.figure(figsize=(7, 8))
for session in session_list:
    traces = np.stack(avgcuetrace_df.loc[avgcuetrace_df['Session']==session, 'Trace'])
    time = np.linspace(timerange[0], timerange[1], len(traces[0]))

    start_idx = np.searchsorted(time, timerange[0])
    end_idx = np.searchsorted(time, timerange[1])
    timesegment = time[start_idx:end_idx]
    # mean_trace = traces.mean(axis=0)
    # sem_trace = sem(traces)
    for k in range(len(traces)):
        plt.plot(traces[k], color=colors[session % len(colors)], label=f'Session {session}', alpha = 0.2)

plt.xlabel('Time (samples)')
plt.xlim(start_idx,end_idx)
plt.xticks(np.arange(0, len(timesegment)+1,
                     len(timesegment)/(timerange[1]-timerange[0])),
           np.arange(timerange[0], timerange[1]+1, 1, dtype=int))
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(timesegment)/(timerange[1]-timerange[0])*(0-timerange[0]),
            linewidth=1, color='black')
plt.ylabel('z-score')
plt.tight_layout()
plt.show()

# -------------------------- by response at the trial --------------------------
responsetrials = []
noresponsetrials = []
for _, row in avgcuetrace_df.iterrows():
    mouse, session, trial, trace = row["Mouse"], row["Session"], row["Trial"], row['Trace']
    
    # Check for matching row in avgflicktrace_df
    matching_row = avgflicktrace_df[
        (avgflicktrace_df["Mouse"] == mouse) & (avgflicktrace_df["Session"] == session) & (avgflicktrace_df["Trial"] == trial)]
    
    if not matching_row.empty:
        responsetrials.append({'Mouse': mouse,'Session': session, 'Trial': trial, 'Trace': trace})
    else:
        noresponsetrials.append({'Mouse': mouse, 
                               'Session': session, 
                               'Trial': trial,
                               'Trace': trace})
        
avgcuebyresponsetrace_df = pd.DataFrame(responsetrials)
avgcuebynoresponsetrace_df= pd.DataFrame(noresponsetrials)

# ------------------------- Plotting Cue Trace By Individual Trials -------------------------

df_of_interest = avgcuebyresponsetrace_df
colors = sns.color_palette("husl", 10)
timerange = [-2,5]
session_list = [3,7]
plt.figure(figsize=(7, 8))
for session in session_list:
    traces = np.stack(df_of_interest.loc[df_of_interest['Session']==session, 'Trace'])
    time = np.linspace(timerange[0], timerange[1], len(traces[0]))

    start_idx = np.searchsorted(time, timerange[0])
    end_idx = np.searchsorted(time, timerange[1])
    timesegment = time[start_idx:end_idx]
    # mean_trace = traces.mean(axis=0)
    # sem_trace = sem(traces)
    for k in range(len(traces)):
        plt.plot(traces[k], color=colors[session % len(colors)], label=f'Session {session}', alpha = 0.2)

plt.xlabel('Time (samples)')
plt.xlim(start_idx,end_idx)
plt.xticks(np.arange(0, len(timesegment)+1,
                     len(timesegment)/(timerange[1]-timerange[0])),
           np.arange(timerange[0], timerange[1]+1, 1, dtype=int))
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(timesegment)/(timerange[1]-timerange[0])*(0-timerange[0]),
            linewidth=1, color='black')
plt.ylabel('z-score')
plt.tight_layout()
plt.title(f'{df_of_interest}')
plt.show()


# -------------------------- BEHAVIORAL DATA --------------------------

alltrialtrace_df['LeverLate'] = alltrialtrace_df['LeverTime'] - alltrialtrace_df['CueTime']
leverlatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LeverLate", aggfunc="first") 
alltrialtrace_df['LickLate'] = alltrialtrace_df['FlickTime'] - alltrialtrace_df['LeverTime']
licklatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LickLate", aggfunc="first") 
behavioral_licks_df = pd.DataFrame(lickspersession)
behavioral_response_df = pd.DataFrame(responsepersession)
behavioral_timeout_df = pd.DataFrame(timeoutpersession)

# ----------------------------------------------------------------------------
# ------------------------- NOW THE FUN STUFF!!!!!!! -------------------------
# ----------------------------------------------------------------------------

# colors = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen', 'mediumseagreen', 'mediumturquoise', 'deepskyblue', 'dodgerblue', 'slateblue', 'darkorchid', 'purple']
colors = sns.color_palette("husl", 10)
colors10 = sns.color_palette("husl", 10)

# ------------------------- PLOTTING HEATMAPS OF FULL TRIALS -------------------------
cmapcolor = 'vlag' #sns.diverging_palette(250, 30, l=60, as_cmap=True)

# ---- INTERTRIAL INTERVALS SORTED BY LATENCY TO PRESS DURING TIME OUT ----
session_list = [0]

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
timerange_iti_new = [0,timerange_iti[1]-timerange_iti[0]]
for ax_i, session in enumerate(session_list):
    
    session_df = allititrace_df.loc[allititrace_df['Session'] == session].copy()
    
    sorted_session_df = session_df.sort_values("TimeoutLength")
    traces = np.vstack(sorted_session_df["Trace"].values)
    n_trials, n_timepoints = traces.shape
    time = np.linspace(timerange_iti_new[0], timerange_iti_new[1], n_timepoints)

    ax = axs[ax_i]
    sns.heatmap(traces, cmap=cmapcolor, vmin=-4, vmax=4, cbar=True, ax=ax)
    ax.set_xticks(np.arange(0, len(time)+1, 2*len(time)/(timerange_iti_new[1]-timerange_iti_new[0])),
        np.arange(timerange_iti_new[0], timerange_iti_new[1]+1, 2, dtype=int))

    # Convert each timeout time → heatmap x index
    def convert_list_to_xcoords(timeout_list):
        return [(t - timerange_iti_new[0]) / (timerange_iti_new[1] - timerange_iti_new[0]) * n_timepoints for t in timeout_list]

    events = sorted_session_df["TimeOuts"].apply(convert_list_to_xcoords).tolist()
    ax.eventplot(events,colors='black',lineoffsets=np.arange(n_trials) + 0.5,linelengths=0.8,linewidths=1)
    ax.set_title(f"Session {session}")

plt.xlabel("Time (sec)")
plt.tight_layout()
plt.savefig('/Users/kristineyoon/Documents/fullititraceheatmap_bylatency.pdf', transparent=True)
plt.show()

# ---- ACTIVE TRIALS SORTED BY LATENCY TO LICK ----
fig, axs = plt.subplots(1, 2, figsize=(10, 8))

for ax_i, session in enumerate(session_list):
    session_df = alltrialtrace_df.loc[alltrialtrace_df['Session'] == session]
    session_sorted_df = session_df.sort_values(by='Latency')
    traces = np.vstack(session_sorted_df['Trace'])
    latency_values = session_sorted_df['Latency'].values  # shape (n_trials,)
    n_trials, n_timepoints = traces.shape

    time = np.linspace(active_time[0], active_time[1], n_timepoints)

    # --- HEATMAP ---
    sns.heatmap(traces, cmap=cmapcolor, vmin=-4, vmax=4, cbar=True, ax=axs[ax_i])
    ax = axs[ax_i]
    ax.set_xticks(np.arange(0, n_timepoints + 1, 2 * n_timepoints / (active_time[1] - active_time[0])), np.arange(active_time[0], active_time[1] + 1, 2, dtype=int))
    ax.axvline(x=n_timepoints * (0 - active_time[0]) / (active_time[1] - active_time[0]),linewidth=1, color='black')

    # --- EVENTPLOT OVERLAY ---
    # Convert latencies (sec) → column positions (indices)
    latency_x = (latency_values - active_time[0]) / (active_time[1] - active_time[0]) * n_timepoints

    events = [[x] for x in latency_x]
    ax.eventplot(events,colors='black',lineoffsets=np.arange(n_trials) + 0.5, linelengths=0.9, linewidths=1)
    ax.set_title(f"Session {session}")

plt.xlabel('Time (sec)')
plt.ylabel('Trials')
plt.tight_layout()
plt.savefig('/Users/kristineyoon/Documents/fulltraceheatmap_bylatency.pdf', transparent=True)
plt.show()




# ------------------------- Plotting Traces -------------------------

def plot_traces_bysession(df, label,ogtime, timerange, ylim, savepath):
    session_list = sorted(df['Session'].unique())
    plt.figure(figsize=(timerange[1]-timerange[0], 8))
    for i, session in enumerate(session_list):
        traces = np.stack(df.loc[df['Session']==session, 'Trace'])
        time = np.linspace(ogtime[0], ogtime[1], len(traces[0]))

        start_idx = np.searchsorted(time, timerange[0])
        end_idx = np.searchsorted(time, timerange[1])
        timesegment = time[start_idx:end_idx]
        mean_trace = traces.mean(axis=0)
        sem_trace = sem(traces)
        plt.plot(mean_trace, color=colors[i % len(colors)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)),
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[i % len(colors)], alpha=0.1)
    
    plt.xlabel('Time (samples)')
    plt.xlim(start_idx,end_idx)
    plt.xticks(np.arange(0, len(timesegment)+1,
                         len(timesegment)/(timerange[1]-timerange[0])),
               np.arange(timerange[0], timerange[1]+1, 1, dtype=int))
    plt.axhline(y=0, linestyle=':', color='black')
    plt.axvline(x=len(timesegment)/(timerange[1]-timerange[0])*(0-timerange[0]),
                linewidth=1, color='black')
    plt.ylabel('z-score')
    plt.title(f'Average {label}-Aligned Trace in {experiment} by Session')
    plt.legend()
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.show()


plot_traces_bysession(avgcuetrace_df, 'Cue', timerange_cue, [-2,5], (-6,8), f'/Users/kristineyoon/Documents/{experiment}_cuebysessions.pdf')
plot_traces_bysession(avglevertrace_df, 'Lever', timerange_lever,timerange_lever, (-6,8), f'/Users/kristineyoon/Documents/{experiment}_leverbysessions.pdf')
plot_traces_bysession(avgflicktrace_df, 'First Lick', timerange_lick, [-2,10], (-6,8), f'/Users/kristineyoon/Documents/{experiment}_firstlickbysessions.pdf')

# # ---- by response ----
plot_traces_bysession(avgcuebyresponsetrace_df, 'Response Cue', timerange_cue, [-2,10], (-2,4), f'/Users/kristineyoon/Documents/{experiment}_cuebyresponse.pdf')
plot_traces_bysession(avgcuebynoresponsetrace_df, 'No Response Cue', timerange_cue, [-2,10], (-2,4), f'/Users/kristineyoon/Documents/{experiment}_cuebynoresponse.pdf')


# ------------------------- Finding Peak Height  -------------------------

totalsessions = len(sorted(avgcuetrace_df['Session'].unique()))

def compute_peak_df(df, timerange, analysis_window, label):
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

peak_cue_df = compute_peak_df(avgcuetrace_df, timerange_cue, [0,1], "Cue")
peak_cue_matrix_df = peak_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 



# # ------------------------- Finding Peak Height Using SCIPY FIND PEAKS  -------------------------
# import numpy as np
# import pandas as pd
# from scipy.signal import find_peaks

# def compute_peak_df(df, timerange, analysis_window, label, min_height=None, min_prominence=None):
#     results = []

#     for _, row in df.iterrows():
#         mouse   = row["Mouse"]
#         session = row["Session"]
#         trial   = row["Trial"]
#         trace   = row["Trace"]
#         time = np.linspace(timerange[0], timerange[1], len(trace))

#         start_idx = np.searchsorted(time, analysis_window[0])
#         end_idx   = np.searchsorted(time, analysis_window[1])
#         trace_segment = trace[start_idx:end_idx].reset_index(drop=True)
#         time_segment  = time[start_idx:end_idx]
#         peaks, properties = find_peaks(trace_segment, height=min_height, prominence=min_prominence)

#         if len(peaks) == 0:
#             peakheight = np.nan
#             peak_time  = np.nan
#         else:
#             # take the largest peak
#             peak_idx = peaks[np.argmax(properties["prominences"])]
#             peakheight = trace_segment[peak_idx]
#             peak_time  = time_segment[peak_idx]
#         results.append({"Alignment": label, "Mouse": mouse, "Session": session, "Trial": trial, "PeakHeight": peakheight, "PeakTime": peak_time})
#     return pd.DataFrame(results)

# # ------------------------- LOOKING AT PEAK HEIGHT -------------------------
# peak_cue_df = compute_peak_df(df = avgcuetrace_df, timerange = timerange_cue, analysis_window=[0,2], label="Cue", min_height=-5, min_prominence=0.1)                                                
# peak_cue_matrix_df = peak_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
# peak_lever_df = compute_peak_df(avglevertrace_df, timerange_lever, [-0.5,0.5], "Lever", min_height=-5, min_prominence=0.2)                   
# peak_lever_matrix_df = peak_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
# peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, [0,2], "FirstLick", min_height=-5, min_prominence=0.2)                   
# peak_flick_matrix_df = peak_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 


# ------------------------- Running Nested Statistics on Cue -------------------------
from scipy.stats import ttest_rel
import statsmodels.formula.api as smf
import itertools
from scipy.stats import ttest_rel
from scipy.stats import chi2

sessions_of_interest = [3,7]

# Average across trials within mouse
mouse_means = (peak_cue_matrix_df[sessions_of_interest].groupby(level="Mouse").mean())
mouse_means_clean = mouse_means.dropna()
t_stat, p_val = ttest_rel(mouse_means_clean[sessions_of_interest[0]],mouse_means_clean[sessions_of_interest[1]])
print(f"Nested paired t-test (Session {sessions_of_interest[0]} vs {sessions_of_interest[1]})")
print(f"n mice = {len(mouse_means_clean)}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")

# ------------------------- Plotting Cue Trace By Individual Trials -------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

colors = sns.color_palette("husl", 10)
timerange = [-2, 6]
session_list = [2,6]

plt.figure(figsize=(7, 8))

for session in session_list:
    # Get all traces for this session
    traces = np.stack(avgcuetrace_df.loc[avgcuetrace_df['Session']==session, 'Trace'])
    # Get corresponding peak heights and times
    peakheights = peak_cue_df.loc[peak_cue_df['Session']==session, 'PeakHeight'].values
    peaktime = peak_cue_df.loc[peak_cue_df['Session']==session, 'PeakTime'].values

    time = np.linspace(timerange[0], timerange[1], traces.shape[1])

    for k in range(len(traces)):
        # Plot trace
        plt.plot(time, traces[k], color=colors[session % len(colors)], alpha=0.2)

        # Plot peak
        if not np.isnan(peakheights[k]):
            plt.scatter(peaktime[k], peakheights[k],
                        color=colors[session % len(colors)],
                        alpha=0.6)

plt.xlabel('Time (s)')
plt.ylabel('z-score')
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=0, linestyle=':', color='black')  # alignment point
plt.xlim(timerange)
plt.tight_layout()
plt.show()


# ------------------------- LOOKING AT LEVER AND FIRST LICK PEAK HEIGHT -------------------------
peak_lever_df = compute_peak_df(avglevertrace_df, timerange_lever, [-0.5,0.5], "Lever")                   
peak_lever_matrix_df = peak_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, [0,1], "FirstLick")                   
peak_flick_matrix_df = peak_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 

sessions_of_interest = [2, 5]
# Average across trials within mouse
mouse_means = (peak_cue_matrix_df[sessions_of_interest].groupby(level="Mouse").mean())
mouse_means_clean = mouse_means.dropna()
t_stat, p_val = ttest_rel(mouse_means_clean[sessions_of_interest[0]],mouse_means_clean[sessions_of_interest[1]])
print('----Looking at Cue Peak Height----')
print(f"Nested paired t-test (Session {sessions_of_interest[0]} vs {sessions_of_interest[1]})")
print(f"n mice = {len(mouse_means_clean)}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")
print('------------------------------------')

# Average across trials within mouse
mouse_means = (peak_lever_matrix_df[sessions_of_interest].groupby(level="Mouse").mean())
mouse_means_clean = mouse_means.dropna()
t_stat, p_val = ttest_rel(mouse_means_clean[sessions_of_interest[0]],mouse_means_clean[sessions_of_interest[1]])
print('----Looking at Lever Peak Height----')
print(f"Nested paired t-test (Session {sessions_of_interest[0]} vs {sessions_of_interest[1]})")
print(f"n mice = {len(mouse_means_clean)}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")
print('------------------------------------')

# Average across trials within mouse
mouse_means = (peak_flick_matrix_df[sessions_of_interest].groupby(level="Mouse").mean())
mouse_means_clean = mouse_means.dropna()
t_stat, p_val = ttest_rel(mouse_means_clean[sessions_of_interest[0]],mouse_means_clean[sessions_of_interest[1]])
print('----Looking at First Lick Peak Height----')
print(f"Nested paired t-test (Session {sessions_of_interest[0]} vs {sessions_of_interest[1]})")
print(f"n mice = {len(mouse_means_clean)}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")
print('------------------------------------')


# ---------------------- Plotting Peak Height ----------------------

def plot_peak_by_session(df, label, ylabel, ylim, analysis_window): #, savepath):
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
    plt.title(f'{label}-Aligned Peak Height ({analysis_window[0]}–{analysis_window[1]} s) in {experiment} ')
    plt.ylim(ylim)
    plt.tight_layout()
    # plt.savefig(savepath, transparent=True)
    plt.show()

plot_peak_by_session(peak_cue_df, 'Cue', 'Peak Height (z)', (-2, 8),[0,1]) # ,'/Users/kristineyoon/Documents/peakheight_cue.pdf')
plot_peak_by_session(peak_lever_df, 'Lever', 'Peak Height (z)', (-2, 10),[-0.5,0.5]) # ,'/Users/kristineyoon/Documents/peakheight_lever.pdf')
plot_peak_by_session(peak_flick_df, 'First Lick', 'Peak Height (z)', (-4, 10),[0,1]) # ,'/Users/kristineyoon/Documents/peakheight_firstlick.pdf')


# plot_peak_by_session(peak_cue_response_df, 'Response Cue', 'Peak Height (z)', (-2, 8)) # ,'/Users/kristineyoon/Documents/peakheight_cue.pdf')
# plot_peak_by_session(peak_cue_noresponse_df, 'No REsponse Cue', 'Peak Height (z)', (-2, 8)) # ,'/Users/kristineyoon/Documents/peakheight_cue.pdf')


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
    plt.title(f'{ylabel} in {experiment} by Session')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Compute AUC DataFrames
auc_cue_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(0, 2))
auc_cue_matrix_df = auc_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 

auc_lever_df = compute_auc(avglevertrace_df, timerange_lever, interval=(-1, 1))
auc_flick_df = compute_auc(avgflicktrace_df, timerange_lick, interval=(0, 2))
auc_lever_matrix_df = auc_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 
auc_flick_matrix_df = auc_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 

# auc_responsecue_df = compute_auc(avgcuebyresponsetrace_df, timerange_cue, interval=(0, 2))
# auc_noresponsecue_df = compute_auc(avgcuebynoresponsetrace_df, timerange_cue, interval=(0, 2))


# Plot AUC for each type
plot_auc(auc_cue_df, ylabel='Cue AUC', ylim=(-5,10))
plot_auc(auc_lever_df, ylabel='Lever AUC', ylim=(-5,10))
plot_auc(auc_flick_df, ylabel='First Lick AUC',ylim=(-5,20))


from scipy.stats import ttest_rel
import statsmodels.formula.api as smf
import itertools
from scipy.stats import ttest_rel
from scipy.stats import chi2


# Extract sessions of interest
sessions_of_interest = [3, 7]

# Average across trials within mouse
mouse_means = (auc_cue_matrix_df[sessions_of_interest].groupby(level="Mouse").mean())
mouse_means_clean = mouse_means.dropna()
t_stat, p_val = ttest_rel(mouse_means_clean[3],mouse_means_clean[7])
print(f"Nested paired t-test (Session 3 vs 7)")
print(f"n mice = {len(mouse_means_clean)}")
print(f"t = {t_stat:.3f}, p = {p_val:.4g}")


df_sub = auc_cue_df.loc[auc_cue_df["Session"].isin([0, 3, 7])].copy()
df_sub["Session"] = df_sub["Session"].astype("category")
model = smf.mixedlm("AUC ~ C(Session)", df_sub, groups=df_sub["Mouse"])
result = model.fit()
print(result.summary())

# Null model: no Session effect
null_model = smf.mixedlm("AUC ~ 1", df_sub, groups=df_sub["Mouse"]).fit()
lr_stat = 2 * (result.llf - null_model.llf)
p_lr = chi2.sf(lr_stat, df=2)  # 3 levels → 2 df
print(f"Nested one-way ANOVA (Session 0,3,7): "f"χ²(2) = {lr_stat:.3f}, p = {p_lr:.4g}")

mouse_means = (df_sub.groupby(["Mouse", "Session"])["AUC"].mean().unstack())
pairs = [(0, 3), (0, 7), (3, 7)]
for s1, s2 in pairs:
    data = mouse_means[[s1, s2]].dropna()
    t, p = ttest_rel(data[s1], data[s2])
    print(f"Session {s1} vs {s2}: t = {t:.3f}, p = {p:.4g}")


# ---------------------- Categorizing Lick Bouts ---------------------- 
def categorize_lick_bout(bout_length):
    if bout_length <= 20 and bout_length > 10:
        return 0
    elif bout_length <= 30 and bout_length > 20:
        return 1
    elif bout_length <= 40 and bout_length > 30:
        return 2
    elif bout_length <= 50 and bout_length > 40:
        return 3
    elif bout_length <= 60 and bout_length > 50:
        return 4
    elif bout_length > 60:
        return 3


# ---------------------- Time to Baseline by Lick Bouts ---------------------- 

# Categorize bouts
avglickbouttrace_df['BoutCategory'] = avglickbouttrace_df['BoutLength'].apply(categorize_lick_bout)

time_to_baseline_dict = {}
for _, row in avglickbouttrace_df.iterrows():
    mouse, session, trial, boutlength, baseline = row['Mouse'], row['Session'], row['Trial'], row['BoutLength'], row['BaselineMean']
    trace = np.array(row['Trace'])
    time = np.linspace(timerange_lick[0], timerange_lick[1], len(trace))
    
    # Focus on post-event period
    start_time, end_time = 1, timerange_lick[1]
    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)
    trace_segment = trace[start_idx:end_idx]
    time_segment = time[start_idx:end_idx]
    
    # Boolean array: True when trace > baseline
    above = (trace_segment > baseline + 0.2) | (trace_segment < baseline - 0.4)
    # above = trace_segment > baseline
    #above = trace_segment > 0

    # Detect transitions
    transitions = np.diff(above.astype(int))

    # +1 → rising through baseline (below → above)
    # -1 → falling through baseline (above → below)
    rise_indices = np.where(transitions == 1)[0]

    
    fall_indices = np.where(transitions == -1)[0]

    baselinetime = np.nan  # default in case no valid crossing
    
    if len(rise_indices) > 0:
        baselinetime = time_segment[rise_indices[0]]
        
    # Store result
    time_to_baseline_dict[(mouse, session, trial, boutlength)] = baselinetime, rise_indices

# Convert to DataFrames
time_to_baseline_df = pd.DataFrame([
    {'Mouse': k[0], 'Session': k[1], 'Trial': k[2], 'BoutLength': k[3], 'TimeToBaseline': v[0], 'Rise Indices': v[1]}
    for k, v in time_to_baseline_dict.items()])


# ---------------------- Plotting Time to Baseline by Sessions ----------------------
plt.figure(figsize=(5, 6))
for session in sorted(time_to_baseline_df['Session'].unique()):
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
plt.title(f'Time to Baseline in {experiment} by Session', fontsize=14)
plt.tight_layout()
plt.show()


# ---------------------- Plotting Time to Baseline by Lick Bouts and By Sessions ----------------------
plt.figure(figsize=(10,6))
for cat in sorted(avglickbouttrace_df['BoutCategory'].unique()):
    cat_data = time_to_baseline_df[time_to_baseline_df['BoutLength'].apply(categorize_lick_bout) == cat]
    for session in sorted(cat_data['Session'].unique()):
        session_values = cat_data[cat_data['Session'] == session]['TimeToBaseline'].values
        clean_data = np.array(session_values)[~np.isnan(session_values)]
        plt.scatter([session]*len(session_values)+cat/5-0.1, session_values,
                       color=colors10[int(cat)], alpha=0.1)
        plt.errorbar(session+cat/5-0.1, clean_data.mean(), yerr=sem(clean_data),fmt='o',
                        color=colors10[int(cat)], capsize=3)
plt.xlabel('Session')
plt.ylabel('Time to Baseline (s)')
plt.title(f'Time to Baseline in {experiment} by Bout Category')
plt.tight_layout()


# ---------------------- Peak Heights by Lick Bouts ---------------------- 
peakheight_dict = {}
for _, row in avglickbouttrace_df.iterrows():
    mouse, session, trial, boutlength, baseline = row['Mouse'], row['Session'], row['Trial'], row['BoutLength'], row['BaselineMean']
    if boutlength >10:
        # Peak height
        trace = np.array(row['Trace'])
        time = np.linspace(timerange_lick[0], timerange_lick[1], len(trace))
        start_time, end_time = 0, 2
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)
        trace_segment = trace[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]
        peak_val = trace_segment.max()
        cat = categorize_lick_bout(boutlength)
        peakheight_dict[(mouse, session, trial, boutlength,)] = peak_val, cat


peakheight_df = pd.DataFrame([
    {'Mouse': k[0], 'Session': k[1], 'Trial': k[2], 'BoutLength': k[3], 'PeakHeight': v[0], 'BoutCategory': v[1]}
    for k, v in peakheight_dict.items()])

# ---------------------- Plotting Peak Heighst by Lick Bouts and By Sessions ----------------------
plt.figure(figsize=(10,6))
for cat in sorted(avglickbouttrace_df['BoutCategory'].unique()):
    cat_data = peakheight_df[peakheight_df['BoutLength'].apply(categorize_lick_bout) == cat]
    for session in sorted(cat_data['Session'].unique()):
        session_values = cat_data[cat_data['Session'] == session]['PeakHeight'].values
        plt.scatter([session]*len(session_values)+cat/5-0.1, session_values,
                       color=colors10[int(cat)], alpha=0.05)
        plt.errorbar(session+cat/5-0.1, session_values.mean(), yerr=sem(session_values),fmt='o',
                        color=colors10[int(cat)], capsize=3)
plt.xlabel('Session')
plt.ylabel('Peak Height (z-score)')
plt.title(f'Peak Height in {experiment} by Bout Category')
plt.tight_layout()
plt.ylim(-10,20)
plt.show()

# ---------------------- Plotting Peak Heights by Lick Bouts in Each Sessions ----------------------
plt.figure(figsize=(10,8))
#for session in sorted(combined_df['Session'].unique()):
for session in [1,3,7]:
    session_data = peakheight_df[peakheight_df['Session'] == session]
    
    # Scatter plot
    sns.scatterplot(data=session_data, 
                    x='PeakHeight', 
                    y='BoutLength', 
                    color=colors10[session], 
                    label=f'Session {session}', 
                    s=100,
                    alpha=0.5)
    
    # Line of best fit
    sns.regplot(data=session_data,
                x='PeakHeight', 
                y='BoutLength', 
                scatter=False,  # Do not plot points again
                color=colors10[session], 
                line_kws={"alpha": 0.7, "lw": 0.7})
    
# Add labels and title
plt.xlabel('Peak Height (z-score)')
plt.ylabel('Bouth Length')
plt.title('Lick Bouts: Bout Length vs Peak Height with Best Fit Line')
plt.legend(title='Session') #, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'/Users/kristineyoon/Documents/{experiment}_boutsvpeakheight.pdf', transparent=True)
plt.show()

# ---------------------- Statistics for Lick Bouts and Peak Heights ----------------------
import statsmodels.api as sm

# Loop over each session to perform linear regression
for session in sorted(peakheight_df['Session'].unique()):
    session_data = peakheight_df[peakheight_df['Session'] == session]
    
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

# ---------------------- Plotting All Mice Traces by Lick Bouts in Specific Sessions ----------------------
plt.figure(figsize=(10,8))
session = 3
newtimerange = [-2,12]
for cat in sorted(avglickbouttrace_df['BoutCategory'].unique()):
    cat_data = avglickbouttrace_df[(avglickbouttrace_df["BoutCategory"] == cat)]
    session_values = cat_data[cat_data['Session'] == session]['Trace'].values
    if len(session_values) > 0:
        time = np.linspace(timerange_lick[0], timerange_lick[1], len(session_values[0]))
    
        start_idx = np.searchsorted(time, newtimerange[0])
        end_idx = np.searchsorted(time, newtimerange[1])
        
        timesegment = time[start_idx:end_idx]
        mean_trace = session_values.mean(axis=0)
        sem_trace = sem(session_values)
        plt.plot(mean_trace, color=colors[int(cat) % len(colors)], label=f'Category {int(cat)}')
        plt.fill_between(range(len(mean_trace)),
                         mean_trace - sem_trace,
                         mean_trace + sem_trace,
                         color=colors[int(cat) % len(colors)], alpha=0.1)

plt.xlabel('Time (samples)')
plt.xlim(start_idx,end_idx)
plt.xticks(np.arange(0, len(timesegment)+1,
                     len(timesegment)/(newtimerange[1]-newtimerange[0])),
           np.arange(newtimerange[0], newtimerange[1]+1, 1, dtype=int))
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(timesegment)/(newtimerange[1]-newtimerange[0])*(0-newtimerange[0]),
            linewidth=1, color='black')
plt.ylabel('z-score')
plt.title(f'Average Lick-Aligned Trace in {experiment} in Session {session} by Lickbout')
plt.legend()
plt.ylim(-6,10)
plt.tight_layout()
plt.savefig(f'/Users/kristineyoon/Documents/{experiment}_session{session}_firstlickbycat.pdf', transparent=True)
plt.show()

# # ---------------------- Plotting All Mice Traces by Lick Bouts in Specific Sessions ----------------------
# plt.figure(figsize=(10,8))
# session = 7
# newtimerange = [-2,12]
# for cat in [2]:
#     cat_data = avglickbouttrace_df[(avglickbouttrace_df["BoutCategory"] == cat)]
#     session_values = cat_data[cat_data['Session'] == session]['Trace'].values
#     for i in range (len(session_values)):
#         time = np.linspace(timerange_lick[0], timerange_lick[1], len(session_values[0]))
        
#         plt.plot(session_values[i], color=colors[int(cat) % len(colors)], label=f'Category {int(cat)}')
        

# plt.xlabel('Time (samples)')
# plt.xlim(start_idx,end_idx)
# plt.xticks(np.arange(0, len(timesegment)+1,
#                      len(timesegment)/(newtimerange[1]-newtimerange[0])),
#            np.arange(newtimerange[0], newtimerange[1]+1, 1, dtype=int))
# plt.axhline(y=0, linestyle=':', color='black')
# plt.axvline(x=len(timesegment)/(newtimerange[1]-newtimerange[0])*(0-newtimerange[0]),
#             linewidth=1, color='black')
# plt.ylabel('z-score')
# plt.title(f'Average Lick-Aligned Trace in {experiment} in Session {session} by Lickbout')
# plt.ylim(-6,10)
# plt.tight_layout()
# plt.show()


from scipy.stats import sem
plt.figure(figsize=(5, 6))
df7 = peak_cue_df.loc[peak_cue_df["Session"] == 7]
df3 = peak_cue_df.loc[peak_cue_df["Session"] == 3]

for trial in range(10):
    trial_data7 = df7.loc[df7['Trial'] == trial, "PeakHeight"]
    trial_data3 = df3.loc[df3['Trial'] == trial, "PeakHeight"]

    # ---- Session 7 ----
    if not trial_data7.empty:
        plt.scatter([trial] * len(trial_data7),trial_data7, color=colors[7 % len(colors)], alpha=0.05)
        mean_val = trial_data7.mean()
        err_val = sem(trial_data7)

        plt.scatter(trial, mean_val, color=colors[7 % len(colors)], s=40, label='Session 7' if trial == 0 else None)
        plt.errorbar(trial, mean_val, yerr=err_val, ecolor=colors[7 % len(colors)], capsize=3)


    # ---- Session 3 ----
    if not trial_data3.empty:
        plt.scatter( [trial] * len(trial_data3), trial_data3, color=colors[3 % len(colors)],  alpha=0.05)
        mean_val = trial_data3.mean()
        err_val = sem(trial_data3)
        plt.scatter( trial,  mean_val, color=colors[3 % len(colors)], s=40, label='Session 3' if trial == 0 else None)
        plt.errorbar( trial, mean_val, yerr=err_val,  ecolor=colors[3 % len(colors)], capsize=3)

plt.xlabel('Trial')
plt.ylabel('Peak Height (z)')
plt.legend()
plt.show()


plt.figure(figsize=(5,6))
df = peak_flick_df.loc[peak_flick_df["Session"] == 7]
for trial in range(10):
    trial_data = df.loc[df['Trial'] == trial, "PeakHeight"]

    if not trial_data.empty:
        plt.scatter([trial]*len(trial_data), trial_data,
                    color=colors[i % len(colors)], alpha=0.05)
        mean_val = trial_data.mean()
        err_val = sem(trial_data)
        plt.scatter(trial, mean_val, color=colors[i % len(colors)], s=40, label=f'Session {session}')
        plt.errorbar(trial, mean_val, yerr=err_val,
                     ecolor=colors[i % len(colors)], capsize=3)

plt.xlabel('Trial')
plt.ylabel('Peak Height (z)')
plt.show()
