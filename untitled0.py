import tdt
import os
import pandas as pd
from tqdm import tqdm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# -------------------------- PARAMETERS --------------------------
events = ['Cue', 'Press', 'Licks', 'Timeout Press']
epocs = ['Po0_', 'Po6_', 'Po4_', 'Po2_']
N_PTS = 1000          # common interpolation length
PRE_CUE_SEC = 10      # seconds before cue[0] to use as baseline window
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'
experiments = ['D2_EtOHLearning', 'D2_SucLearning']
reinforcer_map = {'D2_EtOHLearning': 'etoh','D2_SucLearning': 'sucrose'}

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


from scipy.signal import butter, filtfilt

def lowpass_filtfilt(data, fs, cutoff=3, order=4):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def get_precue_baseline(row, pre_cue_sec=PRE_CUE_SEC):
    """
    Extract mean signal in [cue[0] - pre_cue_sec, cue[0]] from the raw trace.
    
    The stored Trace starts at cue[0] (sample = round(cue[0]*fs)), so we cannot
    slice backwards from it directly. We therefore use the raw stream stored in
    the TDT block. Since you've already loaded everything into alltrialtrace_df,
    the cleanest approach is to store the pre-cue mean at load time (see Step 3
    below for how to add this to your main loop). For now this function shows
    the logic assuming PreCueMean is already a column.
    """
    return row['PreCueMean']   # added in the modified main loop below




# -------------------------- INIT STORAGE --------------------------

all_trials = []
lickspersession = []
responsepersession = []
timeoutpersession = []

# -------------------------- ALL FILES --------------------------

for experiment in experiments:
    if experiment == 'D2_EtOHLearning':
        mice = ['7098','7099','7107','7108', '7310', '7311', '7319', '7321','8729','8730','8731','8732','9299','9302','9325','9326','9327'] #'7296','9325','9326',
        male = ['7107','7108','7319', '7321','8729','8730','8731','8732','9299','9302']
        female = ['7098','7099','7310', '7311','9325','9326','9327' ]
    elif experiment == 'D2_SucLearning' :
        mice = ['7678', '7680', '8733','8742','8743','8747','8748','8750', '7899']
        male = ['7678', '7680','8742','8743','8747','8748']
        female = ['7899', '8733','8750']
    else:
        print('EXPERIMENT IS NOT LISTED')
        
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
            
            # ── PRE-CUE BASELINE ──────────────────────────────────────────────────────
            cue_zero_samp = round(track_cue[0] * fs)          # first cue in samples
            baseline_start = max(0, cue_zero_samp - round(PRE_CUE_SEC * fs))
            baseline_end   = cue_zero_samp                    # up to but not including cue onset
            precue_mean    = df['Filtered'].iloc[baseline_start:baseline_end].mean()
          
            # ── SESSION TRACE (unchanged) ─────────────────────────────────────────────
            cue_zero = round(track_cue[0] * fs)
            cue_end  = round(track_cue[9] * fs)               # or track_cue[-1] if <10 cues
            trial_signal = df['Filtered'].iloc[cue_zero:cue_end]
              
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
    

            all_trials.append({
                'Mouse': mouse, 'Session': session_idx,
                'Trace': trial_signal,
                'PreCueMean': precue_mean,           # <── new field
                'Reinforcer': reinforcer_map[experiment],
                'Licks': len(totallicks),
                'Response': len(totalresponse),
                'Timeout': len(totaltimeout),
                'CueTimeStamp': track_cue,
                'LeverTimeStamp': track_lever,
                'LicksTimeStamp': track_licks,
                'TOTimeStamp': track_to,
                'Sex': sex
            })
            
# -------------------------- CONVERT TO DATAFRAMES --------------------------

alltrialtrace_df = pd.DataFrame(all_trials)
