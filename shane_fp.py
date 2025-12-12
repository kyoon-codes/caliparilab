import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alltraces = {}
N = 100

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Shane_Amph/'
mice = ['27fi1x48fi2','30fi2']

# -------------------------
# Load and organize all traces
# -------------------------
for mouse in mice:
    mouse_dir = os.path.join(folder, mouse)
    Dates = sorted([x for x in os.listdir(mouse_dir) if x.isnumeric()])

    for date_i, date in enumerate(Dates):
        data = tdt.read_block(os.path.join(mouse_dir, date))

        # Extract raw data
        sig405_b = data.streams._405B.data
        sig465_b = data.streams._465B.data
        sig405_c = data.streams._405C.data
        sig465_c = data.streams._465C.data

        # Compute dF/F (vectorized)
        dff_b = (sig465_b - sig405_b) / sig465_b
        dff_c = (sig465_c - sig405_c) / sig465_c

        fs = round(data.streams._465B.fs)

        # Assign traces
        if mouse == '27fi1x48fi2':
            alltraces[(27, date_i)] = dff_b
            alltraces[(48, date_i)] = dff_c
        elif mouse == '30fi2':
            alltraces[(30, date_i)] = dff_c

# -------------------------
# Vectorized Moving Average
# -------------------------
kernel = np.ones(N) / N

def fast_moving_average(x, kernel):
    return np.convolve(x, kernel, mode="valid")


# -------------------------
# Define function to calculate SEM
# 

def sem(arr):
    return np.std(arr, axis=0) / np.sqrt(len(arr))

# -------------------------
# Downsample baseline + amph
# -------------------------
refined_traces = {}
zscored_traces = {}

for (mouse, session), trace in alltraces.items():
    baseline = trace[fs : 5*60*fs + fs]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    amph = trace[-20*60*fs :]
    
    # Vectorized moving average (100× faster)
    down_baseline = fast_moving_average(baseline, kernel)
    down_amph = fast_moving_average(amph, kernel)
    trace = np.concatenate([down_baseline, down_amph])

    refined_traces[(mouse, session)] = trace
    zscored_traces[(mouse, session)] = (trace-np.mean(down_baseline))/np.std(down_baseline)




session_ranges = 5
colors10 = ['indianred',  'gold', 'mediumseagreen',  'deepskyblue', 'darkorchid']

session_data = {}  
for (mouse, session), trace in zscored_traces.items():
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(trace)
    
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)

plt.figure(figsize=(16, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    #if session in [1,6]:
    if session in range(session_ranges):
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')


fig, axs = plt.subplots(session_ranges, 1, figsize=(12, 2*session_ranges),sharex=True)
# Sort sessions so they appear in order
for ax_i, session in enumerate(sorted(session_data.keys())):
    mat = np.vstack(session_data[session])

    sns.heatmap(mat, cmap='vlag', vmin=-4, vmax = 4, cbar=True, ax=axs[ax_i])

    axs[ax_i].set_title(f"Session {session}: Heatmap of Refined Traces")
    axs[ax_i].set_ylabel("Mouse")
    axs[ax_i].set_xlabel("Time")
    
    axis_len = len(session_data[session][0])
    # Define the x axis in "minutes"
    xtick_labels = np.arange(-5, 26, 5)       # -5,0,5,10,15,20,25
    # Map those labels into indices in the trace
    xtick_positions = np.linspace(0, axis_len-1, len(xtick_labels))
    
    axs[ax_i].set_xticks(xtick_positions)
    axs[ax_i].set_xticklabels(xtick_labels, rotation=0)

plt.tight_layout()
plt.show()
