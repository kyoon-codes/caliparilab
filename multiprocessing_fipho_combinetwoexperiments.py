import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import sem, kruskal
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.stats import linregress

# -------------------------- ALL FILES --------------------------

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'


# ------- D2 MEDIUM SPINY NEURONS (ALCOHOL) -------
mice = ['7098','7099','7107','7108', '7310', '7311', '7319', '7321','8729','8730','8731','8732','9299','9302','9325','9326','9327'] #'7296','9325','9326',
male = ['7107','7108','7319', '7321','8729','8730','8731','8732','9299','9302']
female = ['7098','7099','7310', '7311','9325','9326','9327' ]
experiment = 'D2_EtOHLearning'


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

    if mouse in male:
        sex = 'M'
    elif mouse in female:
        sex = 'F'
    print(sex)
        
    for date in dates:
        session_idx = session_map[date]
        date_dir = os.path.join(mouse_dir, date)
        data = tdt.read_block(date_dir)

        # ------------------- SIGNAL PROCESSING BEFORE-------------------
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

        # ------------------- SESSION COUNTS -------------------
        if len(track_cue) > 10:
            totallicks = track_licks[track_licks < track_cue[10]]
            totalresponse = track_lever[track_lever < track_cue[10]]
            totaltimeout = track_to[track_to < track_cue[9]]
            track_cue = track_cue[:10]
        else:
            totallicks = track_licks
            totalresponse = track_lever
            totaltimeout = track_to[track_to < track_cue[-1]]
        
            
        lickspersession.append({'Mouse': mouse, 'Session': session_idx, 'Licks':len(totallicks), 'Sex': sex})
        responsepersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totalresponse), 'Sex': sex})
        timeoutpersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totaltimeout), 'Sex': sex})

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
            
            all_cue_trials.append({'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'Licks': licks[i],
                'Sex': sex})

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
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'Latency': flick_time-cue_time,
                'Sex': sex
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
                    'TimeoutLength': len(trial_to),
                    'Sex': sex
                })
# -------------------------- CONVERT TO DATAFRAMES --------------------------
avgcuetrace_df = pd.DataFrame(all_cue_trials)
avglevertrace_df = pd.DataFrame(all_lever_trials)
avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
avglickbouttrace_df = pd.DataFrame(all_lickbouts)
alltrialtrace_df = pd.DataFrame(all_trials)
allititrace_df = pd.DataFrame(all_iti_trials)


# -------------------------- BEHAVIORAL DATA --------------------------

alltrialtrace_df['LeverLate'] = alltrialtrace_df['LeverTime'] - alltrialtrace_df['CueTime']
leverlatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LeverLate", aggfunc="first") 
alltrialtrace_df['LickLate'] = alltrialtrace_df['FlickTime'] - alltrialtrace_df['LeverTime']
licklatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LickLate", aggfunc="first") 
behavioral_licks_df = pd.DataFrame(lickspersession)
behavioral_response_df = pd.DataFrame(responsepersession)
behavioral_timeout_df = pd.DataFrame(timeoutpersession)

behavioral_licks_matrix_df = behavioral_licks_df.pivot_table(index=["Mouse"], columns="Session", values="Licks", aggfunc="first") 
behavioral_response_matrix_df = behavioral_response_df.pivot_table(index=["Mouse"], columns="Session:", values="Responses", aggfunc="first") 
behavioral_timeout_matrix_df = behavioral_timeout_df.pivot_table(index=["Mouse"], columns="Session:", values="Responses", aggfunc="first") 

behavioral_licklatency_matrix_df= alltrialtrace_df.pivot_table(index=["Mouse"], columns="Session", values="LickLate", aggfunc="mean") 


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
auc_baseline_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(-2, 0))
auc_baseline_matrix_df =  auc_baseline_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 
auc_cue_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(0, 1))
auc_cue_matrix_df = auc_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 

auc_lever_df = compute_auc(avglevertrace_df, timerange_lever, interval=(-.5, .5))
auc_flick_df = compute_auc(avgflicktrace_df, timerange_lick, interval=(0,1))
auc_lever_matrix_df = auc_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 
auc_flick_matrix_df = auc_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 


auc_baseline_matrix2_df =  auc_baseline_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_cue_matrix2_df = auc_cue_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_lever_matrix2_df = auc_lever_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_flick_matrix2_df = auc_flick_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 



# ------------------------- Finding Peak Height Using SCIPY FIND PEAKS  -------------------------
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def compute_peak_df(df, timerange, analysis_window, label, min_height=None, min_prominence=None):
    results = []

    for _, row in df.iterrows():
        mouse   = row["Mouse"]
        session = row["Session"]
        trial   = row["Trial"]
        trace   = row["Trace"]
        time = np.linspace(timerange[0], timerange[1], len(trace))

        start_idx = np.searchsorted(time, analysis_window[0])
        end_idx   = np.searchsorted(time, analysis_window[1])
        trace_segment = trace[start_idx:end_idx].reset_index(drop=True)
        time_segment  = time[start_idx:end_idx]
        peaks, properties = find_peaks(trace_segment, height=min_height, prominence=min_prominence)

        if len(peaks) == 0:
            peakheight = np.nan
            peak_time  = np.nan
        else:
            # take the largest peak
            peak_idx = peaks[np.argmax(properties["prominences"])]
            peakheight = trace_segment[peak_idx]
            peak_time  = time_segment[peak_idx]
        results.append({"Alignment": label, "Mouse": mouse, "Session": session, "Trial": trial, "PeakHeight": peakheight, "PeakTime": peak_time})
    return pd.DataFrame(results)

# ------------------------- LOOKING AT PEAK HEIGHT -------------------------
peak_cue_df = compute_peak_df(df = avgcuetrace_df, timerange = timerange_cue, analysis_window=[0,2], label="Cue", min_height=-5, min_prominence=0.1)                                                
peak_cue_matrix_df = peak_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
peak_lever_df = compute_peak_df(avglevertrace_df, timerange_lever, [-1,1], "Lever", min_height=-5, min_prominence=0.2)                   
peak_lever_matrix_df = peak_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, [0,2], "FirstLick", min_height=-5, min_prominence=0.2)                   
peak_flick_matrix_df = peak_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 

peak_cue_matrix2_df = peak_cue_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 
peak_lever_matrix2_df = peak_lever_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 
peak_flick_matrix2_df = peak_flick_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 


#---------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("\nBuilding Master Trial-by-Trial DataFrame (Using Peak Heights)...")

# 1. Start with the base trial timing and latencies
master_df_etoh = alltrialtrace_df[['Mouse', 'Session', 'Trial', 'LeverLate', 'LickLate', 'Latency']].copy()

# 2. Add Binary Outcomes (1 = happened, 0 = didn't happen)
master_df_etoh['Pressed'] = master_df_etoh['LeverLate'].notna().astype(int)
master_df_etoh['Licked'] = master_df_etoh['Latency'].notna().astype(int)

# 3. Add Neural Features (Using your Peak Height dataframes instead of AUC)
master_df_etoh = pd.merge(master_df_etoh, peak_cue_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Cue_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh = pd.merge(master_df_etoh, peak_lever_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Lever_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh = pd.merge(master_df_etoh, peak_flick_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Lick_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh = pd.merge(master_df_etoh, auc_cue_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Cue_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh = pd.merge(master_df_etoh, auc_lever_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Lever_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh = pd.merge(master_df_etoh, auc_flick_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Lick_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')

# 4. Add Behavioral Features (Max Bout Length per Trial)
trial_bouts = avgflicktrace_df.groupby(['Mouse', 'Session', 'Trial'])['Licks'].max().reset_index()
master_df_etoh = pd.merge(master_df_etoh, trial_bouts, on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh['Licks'] = master_df_etoh['Licks'].fillna(0) 
master_df_etoh = master_df_etoh.sort_values(['Mouse','Session','Trial'])
grouped = master_df_etoh.groupby(['Mouse','Session'])
master_df_etoh['Cumulative_Licks'] = grouped['Licks'].cumsum().shift(1, fill_value=0)
master_df_etoh['Prev_Licks'] = grouped['Licks'].shift(1, fill_value=0)

# 5. Add Timeout Presses (Amount of ITI timeout presses before this trial)
timeout_counts = allititrace_df[['Mouse', 'Session', 'Trial', 'TimeoutLength']].copy()
timeout_counts.rename(columns={'TimeoutLength': 'TimeoutCount'}, inplace=True)
master_df_etoh = pd.merge(master_df_etoh, timeout_counts, on=['Mouse', 'Session', 'Trial'], how='left')
master_df_etoh['TimeoutCount'] = master_df_etoh['TimeoutCount'].fillna(0)

# Drop trials where photometry baseline failed
master_df_etoh = master_df_etoh.dropna(subset=['Cue_Peak'])



#---------------------------
#----------------------------
# ------- D2 MEDIUM SPINY NEURONS (SUCROSE) -------

mice = ['7678', '7680', '8733','8742','8743','8747','8748','8750', '7899']
experiment = 'D2_SucLearning' 
male = ['7678', '7680','8742','8743','8747','8748']
female = ['7899', '8733','8750']

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

    if mouse in male:
        sex = 'M'
    elif mouse in female:
        sex = 'F'
    print(sex)
        
    for date in dates:
        session_idx = session_map[date]
        date_dir = os.path.join(mouse_dir, date)
        data = tdt.read_block(date_dir)

        # ------------------- SIGNAL PROCESSING BEFORE-------------------
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

        # ------------------- SESSION COUNTS -------------------
        if len(track_cue) > 10:
            totallicks = track_licks[track_licks < track_cue[10]]
            totalresponse = track_lever[track_lever < track_cue[10]]
            totaltimeout = track_to[track_to < track_cue[9]]
            track_cue = track_cue[:10]
        else:
            totallicks = track_licks
            totalresponse = track_lever
            totaltimeout = track_to[track_to < track_cue[-1]]
        
            
        lickspersession.append({'Mouse': mouse, 'Session': session_idx, 'Licks':len(totallicks), 'Sex': sex})
        responsepersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totalresponse), 'Sex': sex})
        timeoutpersession.append({'Mouse': mouse,'Session:': session_idx, 'Responses':len(totaltimeout), 'Sex': sex})

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
            
            all_cue_trials.append({'Mouse': mouse,
                'Session': session_idx,
                'Trial': trial_num,
                'CueTime': cue_time,
                'Trace': trial_signal,
                'BaselineMean': baseline_mean,
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'Licks': licks[i],
                'Sex': sex})

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
                'BaselineStd': baseline_std,
                'Sex': sex})

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
                'Latency': flick_time-cue_time,
                'Sex': sex
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
                    'TimeoutLength': len(trial_to),
                    'Sex': sex
                })
# -------------------------- CONVERT TO DATAFRAMES --------------------------
avgcuetrace_df = pd.DataFrame(all_cue_trials)
avglevertrace_df = pd.DataFrame(all_lever_trials)
avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
avglickbouttrace_df = pd.DataFrame(all_lickbouts)
alltrialtrace_df = pd.DataFrame(all_trials)
allititrace_df = pd.DataFrame(all_iti_trials)


# -------------------------- BEHAVIORAL DATA --------------------------

alltrialtrace_df['LeverLate'] = alltrialtrace_df['LeverTime'] - alltrialtrace_df['CueTime']
leverlatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LeverLate", aggfunc="first") 
alltrialtrace_df['LickLate'] = alltrialtrace_df['FlickTime'] - alltrialtrace_df['LeverTime']
licklatency_matrix_df = alltrialtrace_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="LickLate", aggfunc="first") 
behavioral_licks_df = pd.DataFrame(lickspersession)
behavioral_response_df = pd.DataFrame(responsepersession)
behavioral_timeout_df = pd.DataFrame(timeoutpersession)

behavioral_licks_matrix_df = behavioral_licks_df.pivot_table(index=["Mouse"], columns="Session", values="Licks", aggfunc="first") 
behavioral_response_matrix_df = behavioral_response_df.pivot_table(index=["Mouse"], columns="Session:", values="Responses", aggfunc="first") 
behavioral_timeout_matrix_df = behavioral_timeout_df.pivot_table(index=["Mouse"], columns="Session:", values="Responses", aggfunc="first") 

behavioral_licklatency_matrix_df= alltrialtrace_df.pivot_table(index=["Mouse"], columns="Session", values="LickLate", aggfunc="mean") 


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
auc_baseline_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(-2, 0))
auc_baseline_matrix_df =  auc_baseline_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 
auc_cue_df = compute_auc(avgcuetrace_df, timerange_cue, interval=(0, 1))
auc_cue_matrix_df = auc_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 

auc_lever_df = compute_auc(avglevertrace_df, timerange_lever, interval=(-.5, .5))
auc_flick_df = compute_auc(avgflicktrace_df, timerange_lick, interval=(0,1))
auc_lever_matrix_df = auc_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 
auc_flick_matrix_df = auc_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="AUC", aggfunc="first") 


auc_baseline_matrix2_df =  auc_baseline_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_cue_matrix2_df = auc_cue_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_lever_matrix2_df = auc_lever_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 
auc_flick_matrix2_df = auc_flick_df.pivot_table(index=["Mouse"], columns="Session", values="AUC", aggfunc="mean") 



# ------------------------- Finding Peak Height Using SCIPY FIND PEAKS  -------------------------
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def compute_peak_df(df, timerange, analysis_window, label, min_height=None, min_prominence=None):
    results = []

    for _, row in df.iterrows():
        mouse   = row["Mouse"]
        session = row["Session"]
        trial   = row["Trial"]
        trace   = row["Trace"]
        time = np.linspace(timerange[0], timerange[1], len(trace))

        start_idx = np.searchsorted(time, analysis_window[0])
        end_idx   = np.searchsorted(time, analysis_window[1])
        trace_segment = trace[start_idx:end_idx].reset_index(drop=True)
        time_segment  = time[start_idx:end_idx]
        peaks, properties = find_peaks(trace_segment, height=min_height, prominence=min_prominence)

        if len(peaks) == 0:
            peakheight = np.nan
            peak_time  = np.nan
        else:
            # take the largest peak
            peak_idx = peaks[np.argmax(properties["prominences"])]
            peakheight = trace_segment[peak_idx]
            peak_time  = time_segment[peak_idx]
        results.append({"Alignment": label, "Mouse": mouse, "Session": session, "Trial": trial, "PeakHeight": peakheight, "PeakTime": peak_time})
    return pd.DataFrame(results)

# ------------------------- LOOKING AT PEAK HEIGHT -------------------------
peak_cue_df = compute_peak_df(df = avgcuetrace_df, timerange = timerange_cue, analysis_window=[0,2], label="Cue", min_height=-5, min_prominence=0.1)                                                
peak_cue_matrix_df = peak_cue_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
peak_lever_df = compute_peak_df(avglevertrace_df, timerange_lever, [-1,1], "Lever", min_height=-5, min_prominence=0.2)                   
peak_lever_matrix_df = peak_lever_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 
peak_flick_df = compute_peak_df(avgflicktrace_df, timerange_lick, [0,2], "FirstLick", min_height=-5, min_prominence=0.2)                   
peak_flick_matrix_df = peak_flick_df.pivot_table(index=["Mouse", "Trial"], columns="Session", values="PeakHeight", aggfunc="first") 

peak_cue_matrix2_df = peak_cue_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 
peak_lever_matrix2_df = peak_lever_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 
peak_flick_matrix2_df = peak_flick_df.pivot_table(index=["Mouse"], columns="Session", values="PeakHeight", aggfunc="mean") 


#---------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("\nBuilding Master Trial-by-Trial DataFrame (Using Peak Heights)...")

# 1. Start with the base trial timing and latencies
master_df_suc = alltrialtrace_df[['Mouse', 'Session', 'Trial', 'LeverLate', 'LickLate', 'Latency']].copy()

# 2. Add Binary Outcomes (1 = happened, 0 = didn't happen)
master_df_suc['Pressed'] = master_df_suc['LeverLate'].notna().astype(int)
master_df_suc['Licked'] = master_df_suc['Latency'].notna().astype(int)

# 3. Add Neural Features (Using your Peak Height dataframes instead of AUC)
master_df_suc = pd.merge(master_df_suc, peak_cue_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Cue_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc = pd.merge(master_df_suc, peak_lever_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Lever_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc = pd.merge(master_df_suc, peak_flick_df[['Mouse', 'Session', 'Trial', 'PeakHeight']].rename(columns={'PeakHeight': 'Lick_Peak'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc = pd.merge(master_df_suc, auc_cue_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Cue_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc = pd.merge(master_df_suc, auc_lever_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Lever_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc = pd.merge(master_df_suc, auc_flick_df[['Mouse', 'Session', 'Trial', 'AUC']].rename(columns={'AUC': 'Lick_AUC'}), on=['Mouse', 'Session', 'Trial'], how='left')

# 4. Add Behavioral Features (Max Bout Length per Trial)
trial_bouts = avgflicktrace_df.groupby(['Mouse', 'Session', 'Trial'])['Licks'].max().reset_index()
master_df_suc = pd.merge(master_df_suc, trial_bouts, on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc['Licks'] = master_df_suc['Licks'].fillna(0) 
master_df_suc = master_df_suc.sort_values(['Mouse','Session','Trial'])
grouped = master_df_suc.groupby(['Mouse','Session'])
master_df_suc['Cumulative_Licks'] = grouped['Licks'].cumsum().shift(1, fill_value=0)
master_df_suc['Prev_Licks'] = grouped['Licks'].shift(1, fill_value=0)
# 5. Add Timeout Presses (Amount of ITI timeout presses before this trial)
timeout_counts = allititrace_df[['Mouse', 'Session', 'Trial', 'TimeoutLength']].copy()
timeout_counts.rename(columns={'TimeoutLength': 'TimeoutCount'}, inplace=True)
master_df_suc = pd.merge(master_df_suc, timeout_counts, on=['Mouse', 'Session', 'Trial'], how='left')
master_df_suc['TimeoutCount'] = master_df_suc['TimeoutCount'].fillna(0)

# Drop trials where photometry baseline failed
master_df_suc = master_df_suc.dropna(subset=['Cue_Peak'])


master_df_etoh['Reinforcer'] = 'Alcohol'
master_df_suc['Reinforcer'] = 'Sucrose'
master_df_combined = pd.concat([master_df_etoh, master_df_suc], ignore_index=True)


# ===========================
# Probabiliyt of licking or lever pressing
# ===========================
df = master_df_combined.copy()

# A) Did they lick?
import statsmodels.api as sm
import statsmodels.formula.api as smf

lick_model = smf.glm(
    "Licked ~ Reinforcer + Cue_Peak + Trial + Session",
    data=df,
    family=sm.families.Binomial()
).fit()
print(lick_model.summary())

# B) Latency given that they licked
dfL = df[df["Licked"] == 1].copy()
lat_model = smf.mixedlm(
    "Latency ~ Reinforcer + Cue_Peak + Trial + Session",
    data=dfL,
    groups=dfL["Mouse"]
).fit(reml=False)
print(lat_model.summary())


# ===========================
# session effects

dfL = dfL.sort_values(["Mouse","Session","Trial"])
dfL["Trial_idx"] = dfL.groupby(["Mouse","Session"]).cumcount()  # 0..n-1
dfL["Trial_z"] = dfL.groupby(["Mouse","Session"])["Trial_idx"].transform(
    lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
)


lat_model2 = smf.mixedlm(
    "Latency ~ Reinforcer + Cue_Peak  + Prev_Licks + TimeoutCount + Trial_z + C(Session)",
    data=dfL,
    groups=dfL["Mouse"]
).fit(reml=False)
print(lat_model2.summary())

# ===========================
# test reinfocer neural relationships
lat_int = smf.mixedlm(
    "Latency ~ Reinforcer*(Cue_Peak) + Prev_Licks + TimeoutCount + Trial_z",
    data=dfL,
    groups=dfL["Mouse"]
).fit(reml=False)
print(lat_int.summary())

lat_rs = smf.mixedlm(
    "Latency ~ Reinforcer + Cue_Peak + Prev_Licks + TimeoutCount + Trial_z",
    data=dfL,
    groups=dfL["Mouse"],
    re_formula="~Cue_Peak"
).fit(reml=False, method="lbfgs")
print(lat_rs.summary())

# ===========================
# 
# ===========================
import statsmodels.formula.api as smf
# Copy to avoid SettingWithCopyWarning
df_onlylicked = master_df_combined[master_df_combined['Licked']==1].dropna(
    subset=['Cue_Peak','Lever_Peak','Lick_Peak','Licks','Latency','Cumulative_Licks'] #'LeverLate',
).copy()


# Mixed model: latency ~ neural peaks + reinforcer + random intercept per mouse
model = smf.mixedlm(
    "Licks ~ Cue_Peak + Lever_Peak + Lick_Peak + Reinforcer",
    data=df_onlylicked,
    groups=df_onlylicked["Mouse"]
)
result = model.fit()
print(result.summary())

# Optional: visualize predicted vs actual
sns.scatterplot(x=result.fittedvalues, y=df_onlylicked['Licks'], hue=df_onlylicked['Reinforcer'])
plt.xlabel("Predicted Licks")
plt.ylabel("Observed Licks")
plt.title("Mixed Model Predictions vs Observed")
plt.show()

# Mixed model: latency ~ neural peaks + reinforcer + random intercept per mouse
model = smf.mixedlm(
    "Latency ~ Cue_Peak + Lever_Peak + Lick_Peak + Reinforcer",
    data=df_onlylicked,
    groups=df_onlylicked["Mouse"]
)
result = model.fit()
print(result.summary())

# Optional: visualize predicted vs actual
sns.scatterplot(x=result.fittedvalues, y=df_onlylicked['Latency'], hue=df_onlylicked['Reinforcer'])
plt.xlabel("Predicted Latency")
plt.ylabel("Observed Latency")
plt.title("Mixed Model Predictions vs Observed")
plt.show()


# ===========================
# PCA: Neural vs Behavioral Features with Biplot
# ===========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Copy to avoid SettingWithCopyWarning
df_onlylicked = master_df_combined[master_df_combined['Licked']==1].dropna(
    subset=['Cue_Peak','Lever_Peak','Lick_Peak','Licks','Latency','Cumulative_Licks'] #'LeverLate',
).copy()

# ---------------------------
# 1️⃣ Define feature sets
# ---------------------------
neural_features = ['Cue_Peak','Lever_Peak','Lick_Peak'] #
behavioral_features = ['Cumulative_Licks','Licks','Latency'] #'LeverLate',
all_features = neural_features + behavioral_features

X = df_onlylicked[all_features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 2️⃣ Run PCA
# ---------------------------
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

df_onlylicked['PC1'] = pca_result[:,0]
df_onlylicked['PC2'] = pca_result[:,1]
df_onlylicked['PC3'] = pca_result[:,2]

# ---------------------------
# 3️⃣ Feature contributions
# ---------------------------
# Each feature's contribution to PC1 and PC2
loadings = pd.DataFrame(pca.components_.T, index=all_features, columns=['PC1','PC2'])
loadings['Magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
print("Feature Loadings:\n", loadings)

# ---------------------------
# 4️⃣ PCA Scatterplot (Biplot)
# ---------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_onlylicked,
    x='PC1',
    y='PC2',
    hue='Reinforcer',
    size='Licks',
    alpha=0.7,
    s=50,
    sizes=(10,100)
)
# plt.show()

# plt.figure(figsize=(8,6))
# Add arrows for feature contributions
for i, feature in enumerate(loadings.index):
    plt.arrow(0, 0, loadings.PC1[i]*3, loadings.PC2[i]*3, color='red', alpha=0.7, head_width=0.15)
    plt.text(loadings.PC1[i]*3.2, loadings.PC2[i]*3.2, feature, color='black', fontsize=10)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA Biplot: Neural & Behavioral Features")
plt.grid(True)
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.show()

# ===========================
# 3D PCA Biplot: Neural & Behavioral Features
# ===========================
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Features
neural_features = ['Cue_Peak','Lever_Peak','Lick_Peak']
behavioral_features = ['Cumulative_Licks','Licks','Latency']
all_features = neural_features + behavioral_features

# Standardize
X = df_onlylicked[all_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

df_onlylicked['PC1'] = pca_result[:,0]
df_onlylicked['PC2'] = pca_result[:,1]
df_onlylicked['PC3'] = pca_result[:,2]

# ---------------------------
# Feature loadings
# ---------------------------
loadings = pd.DataFrame(pca.components_.T, index=all_features, columns=['PC1','PC2','PC3'])
loadings['Magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2 + loadings['PC3']**2)
print("Feature Loadings:\n", loadings)

# ---------------------------
# 3D Biplot
# ---------------------------
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

# Scatter points (trials)
colors = df_onlylicked['Reinforcer'].map({'Alcohol':'blue','Sucrose':'orange'})
sizes = df_onlylicked['Licks']*2  # scale by Licks
scatter = ax.scatter(df_onlylicked['PC1'], df_onlylicked['PC2'], df_onlylicked['PC3'],
                     c=colors, s=sizes, alpha=0.1)

# Arrows for features
arrow_scale = 3  # scale factor for visibility
for feature in neural_features:
    ax.quiver(0,0,0,
              loadings.loc[feature,'PC1']*arrow_scale,
              loadings.loc[feature,'PC2']*arrow_scale,
              loadings.loc[feature,'PC3']*arrow_scale,
              color='blue', linewidth=2, alpha=0.8)
    ax.text(loadings.loc[feature,'PC1']*arrow_scale*1.1,
            loadings.loc[feature,'PC2']*arrow_scale*1.1,
            loadings.loc[feature,'PC3']*arrow_scale*1.1,
            feature, color='blue', fontsize=10)

for feature in behavioral_features:
    ax.quiver(0,0,0,
              loadings.loc[feature,'PC1']*arrow_scale,
              loadings.loc[feature,'PC2']*arrow_scale,
              loadings.loc[feature,'PC3']*arrow_scale,
              color='green', linewidth=2, alpha=0.8)
    ax.text(loadings.loc[feature,'PC1']*arrow_scale*1.1,
            loadings.loc[feature,'PC2']*arrow_scale*1.1,
            loadings.loc[feature,'PC3']*arrow_scale*1.1,
            feature, color='green', fontsize=10)

# Axes labels with explained variance
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D PCA Biplot: Neural (blue) & Behavioral (green) Features")

# Legend for Reinforcer
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', label='Alcohol', markerfacecolor='blue', markersize=8),
    Line2D([0],[0], marker='o', color='w', label='Sucrose', markerfacecolor='orange', markersize=8)
]
ax.legend(handles=legend_elements, title="Reinforcer")

plt.show()