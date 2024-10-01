import tdt
# from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/Volumes/Kristine/All TDT Data/4887-230130-125550'
data = tdt.read_block(path)

df = pd.DataFrame()
df['Sig490'] = data.streams._490R.data
df['Sig405'] = data.streams._405R.data
df['Dff'] = (df['Sig490']-df['Sig405'])/df['Sig490']
fs = round(data.streams._490R.fs)

fp_df = pd.DataFrame(columns=['Event','Timestamp'])
events = ['Cue', 'Press', 'Licks']
epocs = ['Po0_','Po6_','Po4_']
for a, b in zip(events, epocs):
    event_df = pd.DataFrame(columns=['Event','Timestamp'])
    event_df['Timestamp'] = data.epocs[b].onset
    event_df['Event'] = a
    fp_df = pd.concat([fp_df, event_df])

########################## CUE ALIGNMENT ##########################
align_cue = []

#### CHANGE THIS FOR SAMPLING -- 9, 113 FOR DOWNSAMPLING
sample = 113

#### CHANGE THIS FOR TIME RANGE BEFORE EVENT THAT YOU WANT TO SEE/ZSCORE
timerange_cue = [-2, 20]
for i in range(len(fp_df)):
    if fp_df.iloc[i,0] == 'Cue':
        cue_zero = round(fp_df.iloc[i,1] * fs)
        cue_baseline = cue_zero + timerange_cue[0] * fs
        cue_end = cue_zero + timerange_cue[1] * fs
        trial = np.array(df.iloc[cue_baseline:cue_end,2])
        align_cue.append(trial)

zscore_cue = []
for i in range(len(align_cue)):
    trial = align_cue[i]
    if len(trial) != (-timerange_cue[0]+timerange_cue[1])*fs:
        continue
    zb = np.mean(trial[0:(-timerange_cue[0]*fs)])
    zsd = np.std(trial[0:(-timerange_cue[0]*fs)])
    trial = np.mean(trial.reshape(-1, sample), axis=1)
    trial = (trial - zb)/zsd
    zscore_cue.append(trial)

########################## FIRST LICK ALIGNMENT ##########################
lickbout = []
for i in range(len(fp_df)):
    if fp_df.iloc[i,0] == 'Licks':
        lickbout.append(fp_df.iloc[i,1])

firstlick = []
firstlick.append(lickbout[0])
for i in range(len(lickbout)):
    if i == len(lickbout)-1:
        break
    elif lickbout[i+1] - lickbout[i] < 10:
        continue
    elif lickbout[i+1] - lickbout[i] > 10:
        firstlick.append(lickbout[i+1])

align_lick = []
timerange_lick = [-1,10]
for i in range (len(firstlick)):
    lick_zero = round(firstlick[i] * fs)
    lick_baseline = lick_zero + timerange_lick[0] * fs
    lick_end = lick_zero + timerange_lick[1] * fs
    trial = np.array(df.iloc[lick_baseline:lick_end,2])
    align_lick.append(trial)

zscore_lick = []
for i in range(len(align_lick)):
    trial = align_lick[i]
    if len(trial) != (-timerange_lick[0]+timerange_lick[1])*fs:
        continue
    zb = np.mean(trial[0:(-timerange_lick[0]*fs)])
    zsd = np.std(trial[0:(-timerange_lick[0]*fs)])
    trial = (trial - zb)/zsd
    trial = np.mean(trial.reshape(-1, sample), axis=1)
    zscore_lick.append(trial)
    
################################## PLOTTING HEATMAP WITH EVENTPLOT ##################################
heightratio = [len(zscore_cue),len(zscore_lick)]
fig, axs = plt.subplots(2, 1, figsize=(8,10), gridspec_kw={'height_ratios':heightratio})
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
fig.tight_layout(h_pad=0.8)

axs[0] = fig.add_subplot(2,1,1)
heatmap_rescue = axs[0].imshow(zscore_cue, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_cue[0], timerange_cue[1], len(zscore_cue), 0])
cbar = fig.colorbar(heatmap_rescue, pad=0.08, fraction=0.02)
axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
axs[0].set_ylabel('Trials')
axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[0].set_yticks(np.arange(0, len(zscore_cue)+1,1, dtype=int))

axs[1] = fig.add_subplot(2,1,2)
heatmap_flick = axs[1].imshow(zscore_lick, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_lick[0], timerange_lick[1], len(zscore_lick), 0])
cbar = fig.colorbar(heatmap_flick, pad=0.08, fraction=0.02)
axs[1].axvline(x=0, linewidth=2, color='black', label='Lick Onset')
axs[1].set_ylabel('Trials')
axs[1].set_xlabel('Seconds from First Lick Onset')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[1].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
axs[1].set_yticks(np.arange(0, len(zscore_lick)+1,1, dtype=int))

fig.tight_layout(h_pad=0.45)

###################################### PLOTTING AVG LINE PLOTS ######################################
fig, axs = plt.subplots(2,1,figsize=(10,20))
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)

axs[0] = fig.add_subplot(2,1,1)
axs0_time = np.linspace(timerange_cue[0], timerange_cue[1], int(len(align_cue[0])/sample))
axs[0].plot(axs0_time, np.mean(zscore_cue, axis=0), linewidth=2, color='orange')
axs[0].fill_between(axs0_time, np.mean(zscore_cue, axis=0)+np.std(zscore_cue)
                      ,np.mean(zscore_cue, axis=0)-np.std(zscore_cue), facecolor='orange', alpha=0.2)
axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
axs[0].set_ylabel('z-Score', labelpad = 2)
axs[0].set_xlim(timerange_cue[0], timerange_cue[1])
axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[0].set_ylim(-3, 3)

axs[1] = fig.add_subplot(2,1,2)
axs1_time = np.linspace(timerange_lick[0], timerange_lick[1], int(len(align_lick[0])/sample))
axs[1].plot(axs1_time, np.mean(zscore_lick, axis=0), linewidth=2, color='indigo')
axs[1].fill_between(axs1_time, np.mean(zscore_lick, axis=0)+np.std(zscore_lick)
                      ,np.mean(zscore_lick, axis=0)-np.std(zscore_lick), facecolor='indigo', alpha=0.2)
axs[1].axvline(x=0, linewidth=2, color='black', label='First Lick')
axs[1].set_xlabel('Seconds from First Lick Onset')
axs[1].set_ylabel('z-Score', labelpad = 2)
axs[1].set_xlim(timerange_lick[0], timerange_lick[1])
axs[1].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
axs[1].set_ylim(-3, 3)
