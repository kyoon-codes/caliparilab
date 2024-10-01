## SORTED BY ORDER OF TRIAL 

import tdt
# from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/Volumes/Kristine/All TDT Data/4885-230111-133851'
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
sample = 113
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

########################## LEVER PRESS ALIGNMENT ##########################
align_lever = []
timerange_lever = [-1,10]
for i in range(len(fp_df)):
    if fp_df.iloc[i,0] == 'Press':
        lever_zero = round(fp_df.iloc[i,1] * fs)
        lever_baseline = lever_zero + timerange_lever[0] * fs
        lever_end = lever_zero + timerange_lever[1] * fs
        trial = np.array(df.iloc[lever_baseline:lever_end,2])
        align_lever.append(trial)

zscore_lever = []
for i in range(len(align_lever)):
    trial = align_lever[i]
    if len(trial) != (-timerange_lever[0]+timerange_lever[1])*fs:
        continue
    zb = np.mean(trial[0:(-timerange_lever[0]*fs)])
    zsd = np.std(trial[0:(-timerange_lever[0]*fs)])
    trial = np.mean(trial.reshape(-1, sample), axis=1)
    trial = (trial - zb)/zsd
    zscore_lever.append(trial)
    
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
    
    
########################## ADD LATENCY ##########################
cue_time = 20
lick_time= 10

track_cue = []
track_lever = []
track_licks  =[]
latency= []
leverpermice = []
for i in range(len(fp_df)):
    if fp_df.iloc[i,0] == 'Cue':
        track_cue.append(fp_df.iloc[i,1])
    if fp_df.iloc[i,0] == 'Press':
        track_lever.append(fp_df.iloc[i,1])
    if fp_df.iloc[i,0] == 'Licks':
        track_licks.append(fp_df.iloc[i,1])

lever_latency = {}
for i in range(len(track_cue)):
    lever_list=[]
    for k in range (len(track_lever)):
        if track_lever[k] - track_cue[i] <= cue_time and track_lever[k] - track_cue[i] > 0:
            lever_list.append(track_lever[k] - track_cue[i])
        lever_latency[i] = lever_list

lick_latency = {}
for i in range(len(track_lever)):
    lick_list = []
    for k in range (len(track_licks)):
        if track_licks[k] - track_lever[i] <= lick_time and track_licks[k] - track_lever[i] > 0:
            lick_list.append(track_licks[k] - track_lever[i])
        lick_latency[i] = lick_list

########################## SPLIT DATA TO RESPONSE V. NONRESPONSE ##########################
response_cue = []
nonresponse_cue = []
rp_response = []
for key in lever_latency:
    if len(lever_latency[key]) == 0:
        nonresponse_cue.append(zscore_cue[key])
    else:
        response_cue.append(zscore_cue[key])
        rp_response.append(lever_latency[key])
        
cue_df = pd.Series(rp_response)
lever_df = pd.Series(lick_latency)


################################## PLOTTING HEATMAP WITH EVENTPLOT ##################################

heightratio = [len(response_cue),len(nonresponse_cue), len(zscore_lever), len(zscore_lick), 0, 0]
fig, axs = plt.subplots(6, 1, figsize=(12,20), gridspec_kw={'height_ratios':heightratio})
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
fig.tight_layout(h_pad=0.8)

axs[0] = fig.add_subplot(6,1,1)
heatmap_rescue = axs[0].imshow(response_cue, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_cue[0], timerange_cue[1], len(response_cue), 0])
cbar = fig.colorbar(heatmap_rescue, pad=0.08, fraction=0.02)
axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
axs[0].set_ylabel('Trials')
axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[0].set_yticks(np.arange(0, len(response_cue)+1,1, dtype=int))

axs[1] = fig.add_subplot(6,1,2)
heatmap_norescue = axs[1].imshow(nonresponse_cue, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_cue[0], timerange_cue[1], len(nonresponse_cue), 0])
cbar = fig.colorbar(heatmap_norescue, pad=0.08, fraction=0.02)
axs[1].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
axs[1].set_ylabel('Trials')
axs[1].set_xlabel('Seconds from Cue Onset (No Lever Press)')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[1].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[1].set_yticks(np.arange(0, len(nonresponse_cue)+1,1, dtype=int))

axs[2] = fig.add_subplot(6,1,3)
heatmap_lever = axs[2].imshow(zscore_lever, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_lever[0], timerange_lever[1], len(zscore_lever), 0])
cbar = fig.colorbar(heatmap_lever, pad=0.08, fraction=0.02)
axs[2].axvline(x=0, linewidth=2, color='black', label='Lever Onset')
axs[2].set_ylabel('Trials')
axs[2].set_xlabel('Seconds from Lever Onset')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[2].set_xticks(np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int))
axs[2].set_yticks(np.arange(0, len(zscore_lever)+1,1, dtype=int))
 
axs[3] = fig.add_subplot(6,1,4)
heatmap_flick = axs[3].imshow(zscore_lick, 
                cmap='RdBu', 
                vmin = -3, vmax = 3, 
                interpolation='none', aspect="auto",
                extent=[timerange_lick[0], timerange_lick[1], len(zscore_lick), 0])
cbar = fig.colorbar(heatmap_flick, pad=0.08, fraction=0.02)
axs[3].axvline(x=0, linewidth=2, color='black', label='Lick Onset')
axs[3].set_ylabel('Trials')
axs[3].set_xlabel('Seconds from First Lick Onset')
cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
axs[3].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
axs[3].set_yticks(np.arange(0, len(zscore_lick)+1,1, dtype=int))

fig.tight_layout(h_pad=0.45)

axs[4] = axs[0].twinx()
leveroff = np.arange(0.5,len(cue_df)+0.5)
axs[4].eventplot(cue_df, colors='black', lineoffsets=leveroff)
axs[4].set(xlim=(timerange_cue[0], timerange_cue[1]), 
           ylim=(len(cue_df),0), 
           yticks=np.arange(0.5,len(cue_df)+0.5))

axs[5] = axs[2].twinx()
lickoff = np.arange(0.5,len(lever_df)+0.5)
axs[5].eventplot(lever_df, colors='black', lineoffsets=lickoff)
axs[5].set(xlim=(timerange_lever[0], timerange_lever[1]), 
           ylim=(len(lever_df),0), 
           yticks=np.arange(0.5,len(lever_df)+0.5))

###################################### PLOTTING AVG LINE PLOTS ######################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

fig, axs = plt.subplots(4,1,figsize=(10,20))
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)

axs[0] = fig.add_subplot(4,1,1)
axs0_time = np.linspace(timerange_cue[0], timerange_cue[1], int(len(align_cue[0])/sample))
axs[0].plot(axs0_time, np.mean(response_cue, axis=0), linewidth=2, color='orange')
axs[0].fill_between(axs0_time, np.mean(response_cue, axis=0)+np.std(response_cue)
                      ,np.mean(response_cue, axis=0)-np.std(response_cue), facecolor='orange', alpha=0.2)
axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
axs[0].set_ylabel('z-Score', labelpad = 2)
axs[0].set_xlim(timerange_cue[0], timerange_cue[1])
axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[0].set_ylim(-3, 3)

axs[1] = fig.add_subplot(4,1,2)
axs1_time = np.linspace(timerange_cue[0], timerange_cue[1], int(len(align_cue[0])/sample))
axs[1].plot(axs1_time, np.mean(nonresponse_cue, axis=0), linewidth=2, color='gold')
axs[1].fill_between(axs1_time, np.mean(nonresponse_cue, axis=0)+np.std(nonresponse_cue)
                      ,np.mean(response_cue, axis=0)-np.std(response_cue), facecolor='gold', alpha=0.2)
axs[1].axvline(x=0, linewidth=2, color='black', label='Cue')
axs[1].set_xlabel('Seconds from Cue Onset (No Lever Press)')
axs[1].set_ylabel('z-Score', labelpad = 2)
axs[1].set_xlim(timerange_cue[0], timerange_cue[1])
axs[1].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
axs[1].set_ylim(-3, 3)

axs[2] = fig.add_subplot(4,1,3)
axs2_time = np.linspace(timerange_lever[0], timerange_lever[1], int(len(align_lever[0])/sample))
axs[2].plot(axs2_time, np.mean(zscore_lever, axis=0), linewidth=2, color='teal')
axs[2].fill_between(axs2_time, np.mean(zscore_lever, axis=0)+np.std(zscore_lever)
                      ,np.mean(zscore_lever, axis=0)-np.std(zscore_lever), facecolor='teal', alpha=0.2)
axs[2].axvline(x=0, linewidth=2, color='black', label='Lever')
axs[2].set_xlabel('Seconds from Lever Onset')
axs[2].set_ylabel('z-Score', labelpad = 2)
axs[2].set_xlim(timerange_lever[0], timerange_lever[1])
axs[2].set_xticks(np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int))
axs[2].set_ylim(-3, 3)

axs[3] = fig.add_subplot(4,1,4)
axs3_time = np.linspace(timerange_lick[0], timerange_lick[1], int(len(align_lick[0])/sample))
axs[3].plot(axs3_time, np.mean(zscore_lick, axis=0), linewidth=2, color='indigo')
axs[3].fill_between(axs3_time, np.mean(zscore_lick, axis=0)+np.std(zscore_lick)
                      ,np.mean(zscore_lick, axis=0)-np.std(zscore_lick), facecolor='indigo', alpha=0.2)
axs[3].axvline(x=0, linewidth=2, color='black', label='First Lick')
axs[3].set_xlabel('Seconds from First Lick Onset')
axs[3].set_ylabel('z-Score', labelpad = 2)
axs[3].set_xlim(timerange_lick[0], timerange_lick[1])
axs[3].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
axs[3].set_ylim(-3, 3)
