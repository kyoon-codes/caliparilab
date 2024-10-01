import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

def sem(arr):
    """Calculate the Standard Error of the Mean (SEM)."""
    return np.std(arr, axis=0) / np.sqrt(len(arr))

def load_data(folder, mice):
    """Load data for each mouse and session."""
    data_dict = {}
    for mouse in mice:
        mouse_dir = os.path.join(folder, mouse)
        dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
        dates.sort()
        data_dict[mouse] = []
        for date in dates:
            date_dir = os.path.join(mouse_dir, date)
            data = tdt.read_block(date_dir)
            data_dict[mouse].append((date, data))
    return data_dict

def process_fp_data(data, timerange_cue, fs):
    """Process fiber photometry data to calculate dFF and z-scores."""
    df = pd.DataFrame({
        'Sig465': data.streams._465B.data,
        'Sig405': data.streams._405B.data
    })
    df['Dff'] = (df['Sig465'] - df['Sig405']) / df['Sig465']

    cue_times = extract_event_times(data, ['Cue', 'Press', 'Licks', 'Timeout Press'])
    align_cue = align_events(cue_times['Cue'], df, timerange_cue, fs)
    
    sample_cue = downsample_traces(align_cue, 100)
    zscore_cue, baselinedict = calculate_zscores(sample_cue, timerange_cue, fs)
    
    return zscore_cue, baselinedict

def extract_event_times(data, events):
    """Extract event times from TDT data."""
    epocs = ['Po0_', 'Po6_', 'Po4_', 'Po2_']
    fp_df = pd.DataFrame(columns=['Event', 'Timestamp'])
    for event, epoc in zip(events, epocs):
        if epoc in data.epocs:
            event_df = pd.DataFrame({
                'Timestamp': data.epocs[epoc].onset,
                'Event': event
            })
            fp_df = pd.concat([fp_df, event_df])
    return fp_df

def align_events(event_times, df, timerange, fs):
    """Align events based on timestamps."""
    aligned_events = []
    for time in event_times:
        zero_point = round(time * fs)
        baseline = zero_point + timerange[0] * fs
        end = zero_point + timerange[1] * fs
        trial = np.array(df.iloc[baseline:end, 2])
        aligned_events.append(trial)
    return aligned_events

def downsample_traces(traces, N):
    """Downsample traces using a moving window mean."""
    downsampled_traces = []
    for trace in traces:
        downsampled = [np.mean(trace[i:i+N-1]) for i in range(0, len(trace), N)]
        downsampled_traces.append(downsampled)
    return downsampled_traces

def calculate_zscores(traces, timerange, fs):
    """Calculate z-scores for each trace based on the baseline period."""
    zscored_traces = []
    baselinedict = {}
    baseline_end = round(-timerange[0] * fs / 100)
    for trace in traces:
        baseline_mean = np.mean(trace[:baseline_end])
        baseline_std = np.std(trace[:baseline_end])
        baselinedict[trace] = (baseline_mean, baseline_std)
        zscored = (trace - baseline_mean) / baseline_std
        zscored_traces.append(zscored)
    return zscored_traces, baselinedict

def plot_traces(traces, timerange, title, filename):
    """Plot average traces with SEM."""
    session_data = {}
    for key, value in traces.items():
        mouse, session = key
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(value)

    mean_traces = {session: np.mean(traces, axis=0) for session, traces in session_data.items()}
    sem_traces = {session: sem(traces) for session, traces in session_data.items()}

    plt.figure(figsize=(10, 6))
    for session, mean_trace in mean_traces.items():
        sem_trace = sem_traces[session]
        plt.plot(mean_trace, color=sns.color_palette("light:teal")[int(session)+1], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("light:teal")[int(session)+1], alpha=0.3)

    plt.xlabel('Time (samples)')
    plt.xticks(np.arange(0, len(mean_trace)+1, len(mean_trace)/(timerange[1]-timerange[0])), 
               np.arange(timerange[0], timerange[1]+1, 1, dtype=int), rotation=0)
    plt.axhline(y=0, linestyle=':', color='black')
    plt.axvline(x=len(mean_traces[0])/(timerange[1]-timerange[0])*(0-timerange[0]), linewidth=1, color='black')
    plt.ylabel('Z-score')
    plt.title(title)
    plt.legend()
    plt.savefig(filename, transparent=True)
    plt.show()

# Example usage
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'
mice = ['7098', '7099', '7107', '7108']
timerange_cue = [-2, 5]


data_dict = load_data(folder, mice)
fs = round(data_dict[0].streams._465B.fs)  # Example sampling rate

avgcuetrace_dict = {}
for mouse, sessions in data_dict.items():
    for date, data in sessions:
        zscore_cue, baselinedict = process_fp_data(data, timerange_cue, fs)
        avgcuetrace_dict[(mouse, date)] = np.mean(zscore_cue, axis=0)  # Example averaging, adjust as needed

plot_traces(avgcuetrace_dict, timerange_cue, 'Average Cue-aligned Trace with SEM by Session', '/Users/kristineyoon/Documents/cue.pdf')
