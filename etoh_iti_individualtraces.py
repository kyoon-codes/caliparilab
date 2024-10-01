import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folder = '/Volumes/SAMSUNG USB/'
files = os.listdir(folder)
files.sort()
print(files)

foi = '4413R-240530-135500'
path = str(folder+foi)
data = tdt.read_block(path)

df = pd.DataFrame()
df['Sig490'] = data.streams._490R.data
df['Sig405'] = data.streams._405R.data
df['Dff'] = (df['Sig490']-df['Sig405'])/df['Sig490']
fs = round(data.streams._490R.fs)

split1 = str(data.epocs).split('\t')
y = []
for elements in split1:
    x = elements.split('\n')
    if '[struct]' in x:
        x.remove('[struct]')
    y.append(x)
z= [item for sublist in y for item in sublist]

fp_df = pd.DataFrame(columns=['Event','Timestamp'])
events = ['Cue', 'Press', 'Licks']
epocs = ['Po0_','Po6_','Po4_']
for a, b in zip(events, epocs):
    if b in z:
        event_df = pd.DataFrame(columns=['Event','Timestamp'])
        event_df['Timestamp'] = data.epocs[b].onset
        event_df['Event'] = a
        fp_df = pd.concat([fp_df, event_df])


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

            
########################## CUE ALIGNMENT ##########################    
        
cuestolevers={}
for i in range(len(track_cue)):
    for k in range(len(track_lever)):
        if track_lever[k]- track_cue[i]<= cue_time and track_lever[k]- track_cue[i]>0:
            cuestolevers[track_cue[i]]=track_lever[k]
            
            
align_cue = []
timerange_cue = [-2, 5]
for i in range(len(track_cue)):
    cue_zero = round(track_cue[i] * fs)
    cue_baseline = cue_zero + timerange_cue[0] * fs
    cue_end = cue_zero + timerange_cue[1] * fs
    trial = np.array(df.iloc[cue_baseline:cue_end,2])
    align_cue.append(trial)
        
### DOWNSAMPLING
N = 100
sample_cue=[]
for lst in align_cue: 
    small_lst = []
    for i in range(0, len(trial), N):
        small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    sample_cue.append(small_lst)
    
zscore_cue = []

baselinedict = {}
for i in range(len(sample_cue)):
    trial = sample_cue[i]
    zb = np.mean(trial[0:round((-timerange_cue[0]*fs/N))])
    zsd = np.std(trial[0:round((-timerange_cue[0]*fs/N))])
    baselinedict[track_cue[i]] = zb, zsd
    trial = (trial - zb)/zsd
    zscore_cue.append(trial)

plt.plot(zscore_cue[0])
#### ALIGN TO CUE: AVERAGE LINE PLOT OF TRIALS ####
plt.figure(figsize=(8,4))
plt.plot(np.mean(zscore_cue, axis=0), color='teal')
plt.fill_between(np.arange(0,len(zscore_cue[0]),1),
                np.mean(zscore_cue, axis=0)+np.std(zscore_cue, axis=0), 
                np.mean(zscore_cue, axis=0)-np.std(zscore_cue, axis=0),
                facecolor='teal',
                alpha=0.2)
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Cue Onset (s)')


####
### TO TRANSFER TO PRISM ###
#TRACES
fptrace_df = pd.DataFrame()
for i in range(0,10):
    fptrace_df[i]=zscore_cue[i]
#PEAK HEIGHT IN THE FIRST SECOND
calc_range = 1
peakheight_df = pd.DataFrame()
for i in range(0,10):
    peakheight = max(fptrace_df[i][round((-timerange_cue[0])*fs/N):round((calc_range-timerange_cue[0])*fs/N)])
    peakheight_df.loc[0,i]=peakheight

##### RESPONDING CUES ALIGNMENT ONLY #####
respondingcue = []
for cuetime, traces in zip(track_cue, sample_cue):
    for cue,lever in cuestolevers.items():
        if cuetime == cue:
            zb = baselinedict[cue][0]
            zsd = baselinedict[cue][1]
            newtrial = (traces - zb)/zsd
            respondingcue.append(newtrial)

#### GRAPHING ####
plt.figure(figsize=(8,4))
plt.plot(np.mean(respondingcue, axis=0), color='limegreen')
plt.fill_between(np.arange(0,len(respondingcue[0]),1),
                np.mean(respondingcue, axis=0)+np.std(respondingcue, axis=0), 
                np.mean(respondingcue, axis=0)-np.std(respondingcue, axis=0),
                facecolor='limegreen',
                alpha=0.2)
plt.xticks(np.arange(0,len(respondingcue[0])+1,len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Cue Onset (s)')
plt.ylim(-2,5)
plt.tight_layout()
plt.savefig('/Users/kristineyoon/Documents/response.pdf', transparent=True)

##### RESPONDING CUES ALIGNMENT ONLY #####
nonrespondingcue = []
for cuetime, traces in zip(track_cue, sample_cue):
    for cue,lever in cuestolevers.items():
        if cuetime != cue:
            zb = baselinedict[cue][0]
            zsd = baselinedict[cue][1]
            newtrial = (traces - zb)/zsd
            nonrespondingcue.append(newtrial)

#### GRAPHING ####
plt.figure(figsize=(8,3))
plt.plot(np.mean(nonrespondingcue, axis=0), color='limegreen')
plt.fill_between(np.arange(0,len(nonrespondingcue[0]),1),
                np.mean(nonrespondingcue, axis=0)+np.std(nonrespondingcue, axis=0), 
                np.mean(nonrespondingcue, axis=0)-np.std(nonrespondingcue, axis=0),
                facecolor='limegreen',
                alpha=0.2)
plt.xticks(np.arange(0,len(nonrespondingcue[0])+1,len(nonrespondingcue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(nonrespondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Cue Onset (s)')
plt.ylim(-2,5)
plt.tight_layout()
plt.savefig('/Users/kristineyoon/Documents/noresponse.pdf', transparent=True)

########################## LEVER PRESS ALIGNMENT ##########################
align_lever = []
timerange_lever = [-2, 5]
for i in range(len(track_lever)):
    lever_zero = round(track_lever[i] * fs)
    lever_baseline = lever_zero + timerange_lever[0] * fs
    lever_end = lever_zero + timerange_lever[1] * fs
    trial = np.array(df.iloc[lever_baseline:lever_end,2])
    align_lever.append(trial)

### DOWNSAMPLING
sample_lever=[]
for lst in align_lever: 
    small_lst = []
    for i in range(0, len(trial), N):
        small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    sample_lever.append(small_lst)
    
zscore_lever = []
for levertime, trace in zip(track_lever, sample_lever):
    for cue,lever in cuestolevers.items():
        if levertime == lever:
            zb = baselinedict[cue][0]
            zsd = baselinedict[cue][1]
            newtrial = (trace - zb)/zsd
            zscore_lever.append(newtrial)

#### ALIGN TO LEVER PRESS: AVERAGE LINE PLOT OF TRIALS ####
plt.plot(np.mean(zscore_lever, axis=0), color='orangered')
plt.fill_between(np.arange(0,len(zscore_lever[0]),1),
                 np.mean(zscore_lever, axis=0)+np.std(zscore_lever, axis=0), 
                np.mean(zscore_lever, axis=0)-np.std(zscore_lever, axis=0),
                facecolor='orangered',
                alpha=0.2)
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Lever Press Onset (s)')


################## FIRST LICK ALIGNMENT ################
N=10
track_flicks = []
firstlicks = []
for press in track_lever:
    lickyes = np.array(track_licks) > press
    firstlicktime = np.where(lickyes == True)
    if len(firstlicktime[0]) > 0:
        firstlicks.append(firstlicktime[0][0])
firstlicks = list(set(firstlicks))
for index in firstlicks:
    track_flicks.append(track_licks[index])

cuestolicks={}
for i in range(len(track_cue)):
    for k in range(len(track_flicks)):
        if track_flicks[k]- track_cue[i] <= (cue_time + lick_time) and track_flicks[k]- track_cue[i]>0:
            cuestolicks[track_cue[i]]=track_flicks[k]
            
align_flick = []
timerange_lick = [-2, 10]
for i in range(len(track_flicks)):
    flick_zero = round(track_flicks[i] * fs)
    flick_baseline = flick_zero + timerange_lick[0] * fs
    flick_end = flick_zero + timerange_lick[1] * fs
    trial = np.array(df.iloc[flick_baseline:flick_end,2])
    align_flick.append(trial)

### DOWNSAMPLING
sample_flick=[]
for lst in align_flick: 
    small_lst = []
    for i in range(0, len(lst), N):
        small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    sample_flick.append(small_lst)

zscore_flick = []
for flicktime, trace in zip(track_flicks, sample_flick):
    for cue,lick in cuestolicks.items():
        if flicktime == lick:
            zb = baselinedict[cue][0]
            zsd = baselinedict[cue][1]
            newtrial = (trace - zb)/zsd
            zscore_flick.append(newtrial)

#### ALIGN TO FIRSTLICK: AVERAGE LINE PLOT OF TRIALS ####

for i in range(len(zscore_flick)):
    plt.plot((zscore_flick[i]), color=sns.color_palette("light:deeppink")[i],label=i)
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('First Lick Onset (s)')

plt.plot(np.mean(zscore_flick, axis=0), color='deeppink')
plt.fill_between(np.arange(0,len(zscore_flick[0]),1),
                 np.mean(zscore_flick, axis=0)+np.std(zscore_flick, axis=0), 
                np.mean(zscore_flick, axis=0)-np.std(zscore_flick, axis=0),
                facecolor='deeppink',
                alpha=0.2)
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('First Lick Onset (s)')

## AREA UNDER CURVE ##
auc_df = pd.DataFrame()
#cue
def trapezoidal_rule(x, y):
    area = 0.5 * sum((y[i] + y[i+1]) * ((x[i+1] - x[i])/(fs/N)) for i in range(len(x)-1))
    return area

timerange_auc = [0,2]
N = 10

auc_cue_range=[]
for i in range (len(zscore_cue)):
    trial = zscore_cue[i][int((timerange_auc[0]-timerange_cue[0])*(fs/N)):int((timerange_auc[1]-timerange_cue[0])*(fs/N))]
    auc_cue_range.append(trial)

auc_cue=[]
for i in range(len(auc_cue_range)):
    plt.plot(np.arange(0,len(auc_cue_range[i]),1),auc_cue_range[i])
    area = trapezoidal_rule(np.arange(0,len(auc_cue_range[i]),1), auc_cue_range[i])
    auc_cue.append(area)
    auc_df.at[i,'Cue'] = area
    
# responding cue
auc_respcue_range=[]
for i in range (len(respondingcue)):
    trial = respondingcue[i][int((timerange_auc[0]-timerange_cue[0])*(fs/N)):int((timerange_auc[1]-timerange_cue[0])*(fs/N))]
    auc_respcue_range.append(trial)

for i in range(len(auc_respcue_range)):
    plt.plot(np.arange(0,len(auc_respcue_range[i]),1),auc_respcue_range[i])
    area = trapezoidal_rule(np.arange(0,len(auc_respcue_range[i]),1), auc_respcue_range[i])
    print(area)
    auc_df.at[i,'Respond'] = area
    
# lever press
auc_levpress_range=[]
for i in range (len(zscore_lever)):
    trial = zscore_lever[i][len(zscore_lever[i])//2:]
    trial = trial[0:int(1*fs/N)]
    auc_levpress_range.append(trial)

for i in range(len(auc_levpress_range)):
    plt.plot(np.arange(0,len(auc_levpress_range[i]),1),auc_levpress_range[i])
    area = trapezoidal_rule(np.arange(0,len(auc_levpress_range[i]),1), auc_levpress_range[i])
    print(area/(fs/N))
    auc_df.at[i,'LeverPress'] = area/(fs/N)

# flick
auc_flick_range=[]
for i in range (len(zscore_flick)):
    trial = zscore_flick[i][len(zscore_flick[i])//2:]
    trial = trial[0:int(1*fs/N)]
    auc_flick_range.append(trial)

for i in range(len(auc_flick_range)):
    plt.plot(np.arange(0,len(auc_flick_range[i]),1),auc_flick_range[i])
    area = trapezoidal_rule(np.arange(0,len(auc_flick_range[i]),1), auc_flick_range[i])
    print(area/(fs/N))
    auc_df.at[i,'Lick'] = area/(fs/N)


#### PLOT EVERYTHING ALL TOGETHER ####
fig, axs = plt.subplots(2, 2, figsize=(8,6))
plt.suptitle(f'Recording {(np.where(np.array(files) == foi))[0][0]}')
    #### ALIGN TO CUE: AVERAGE LINE PLOT OF TRIALS ####
axs[0,0].plot(np.mean(nonrespondingcue, axis=0), color='teal')
axs[0,0].fill_between(np.arange(0,len(nonrespondingcue[0]),1),
                np.mean(nonrespondingcue, axis=0)+np.std(nonrespondingcue, axis=0), 
                np.mean(nonrespondingcue, axis=0)-np.std(nonrespondingcue, axis=0),
                facecolor='teal',
                alpha=0.2)
axs[0,0].title.set_text('Align to Cues')
axs[0,0].set_ylim([-5,6])
plt.sca(axs[0,0])
plt.xticks(np.arange(0,len(nonrespondingcue[0])+1,len(nonrespondingcue[0])/((timerange_cue[1]-timerange_cue[0]))), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(nonrespondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Seconds from Cue Onset (s)')
fig.tight_layout(h_pad=0.5)
    #### ALIGN TO RESPONDING CUES ####
axs[0,1].plot(np.mean(respondingcue, axis=0), color='limegreen')
axs[0,1].fill_between(np.arange(0,len(respondingcue[0]),1),
                np.mean(respondingcue, axis=0)+np.std(respondingcue, axis=0), 
                np.mean(respondingcue, axis=0)-np.std(respondingcue, axis=0),
                facecolor='limegreen',
                alpha=0.2)
axs[0,1].title.set_text('Align to Responded Cues')
axs[0,1].set_ylim([-5,6])
plt.sca(axs[0,1])
plt.xticks(np.arange(0,len(respondingcue[0])+1,len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Seconds from Cue Onset (s)')
fig.tight_layout(h_pad=0.5)
    #### ALIGN TO LEVER PRESS: AVERAGE LINE PLOT OF TRIALS ####
axs[1,0].plot(np.mean(zscore_lever, axis=0), color='orangered')
axs[1,0].fill_between(np.arange(0,len(zscore_lever[0]),1),
                 np.mean(zscore_lever, axis=0)+np.std(zscore_lever, axis=0), 
                np.mean(zscore_lever, axis=0)-np.std(zscore_lever, axis=0),
                facecolor='orangered',
                alpha=0.2)
axs[1,0].title.set_text('Align to Lever Presses')
axs[1,0].set_ylim([-5,6])
plt.sca(axs[1,0])
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Seconds from Lever Press Onset (s)')
fig.tight_layout(h_pad=0.5)
    #### ALIGN TO FIRSTLICK: AVERAGE LINE PLOT OF TRIALS ####
axs[1,1].plot(np.mean(zscore_flick, axis=0), color='deeppink')
axs[1,1].fill_between(np.arange(0,len(zscore_flick[0]),1),
                 np.mean(zscore_flick, axis=0)+np.std(zscore_flick, axis=0), 
                np.mean(zscore_flick, axis=0)-np.std(zscore_flick, axis=0),
                facecolor='deeppink',
                alpha=0.2)
axs[1,1].title.set_text('Align to First Licks')
axs[1,1].set_ylim([-5,6])
plt.sca(axs[1,1])
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Seconds from First Lick Onset (s)')
fig.tight_layout(h_pad=0.5)

plt.savefig('/Users/kristineyoon/Documents/all curves.pdf', transparent=True)

######### INDIVIDUAL TRACES IN THE TRIALS WITH CUE-LEVER-LICK ######### 
align_cue = []
timerange_cue = [-5, 20]
for i in range(len(track_cue)):
    cue_zero = round(track_cue[i] * fs)
    cue_baseline = cue_zero + timerange_cue[0] * fs
    cue_end = cue_zero + timerange_cue[1] * fs
    trial = np.array(df.iloc[cue_baseline:cue_end,2])
    align_cue.append(trial)
        
### DOWNSAMPLING
sample_cue=[]
for lst in align_cue: 
    small_lst = []
    for i in range(0, len(trial), N):
        small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    sample_cue.append(small_lst)
    
zscore_cue = []

baselinedict = {}
for i in range(len(sample_cue)):
    trial = sample_cue[i]
    zb = np.mean(trial[0:round((-timerange_cue[0]*fs/N))])
    zsd = np.std(trial[0:round((-timerange_cue[0]*fs/N))])
    baselinedict[track_cue[i]] = zb, zsd
    trial = (trial - zb)/zsd
    zscore_cue.append(trial)

zscoreindex=[]
for i in range (len(track_cue)):
    truefalse = np.where(track_cue[i] in cuestolicks.keys(), True, False)
    zscoreindex.append(truefalse)
    
selectcuetrials = []
for i in range(len(zscoreindex)):
    if zscoreindex[i] == True:
        selectcuetrials.append(zscore_cue[i])
        
leverlatencytocue=[]
licklatencytocue=[]
for i in range(len(track_cue)):
    for k in range(len(track_lever)):
        for j in range(len(track_flicks)):
            if track_lever[k]- track_cue[i] <= cue_time and track_lever[k]- track_cue[i]>0 and track_flicks[j]- track_cue[i] <= (cue_time + lick_time) and track_flicks[j]- track_cue[i]>0:
                leverlatencytocue.append(track_lever[k]- track_cue[i])
                licklatencytocue.append(track_flicks[j]- track_cue[i])
            
ntrials = len(zscore_flick)

fig, axs = plt.subplots(ntrials, 1, figsize=(12,ntrials*2))
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

for i in range(0,ntrials):
    axs[i] = fig.add_subplot(ntrials,1,i+1)
    axs[i].axvline(x=0, linewidth=3, color='teal', label='Cue', alpha=0.6)
    axs[i].axvline(x=leverlatencytocue[i], linewidth=3, color='orangered', label='Lever', alpha=0.6)
    axs[i].axvline(x=licklatencytocue[i], linewidth=3, color='deeppink', label='First Lick', alpha=0.6)
    axs[i].legend(loc='upper right')
    axs_time = np.linspace(timerange_cue[0], timerange_cue[1], len(selectcuetrials[i]))
    axs[i].plot(axs_time, np.array(selectcuetrials[i]), linewidth=2, color = 'cornflowerblue')
    axs[i].axhline(y=0, linewidth=1, linestyle=':', color='black')
    axs[i].set_ylabel(f'Trial {i} \n\n z-Score', labelpad = 1)
    axs[i].set_xlabel('seconds')
    axs[i].set_xlim(timerange_cue[0], timerange_cue[1])
    axs[i].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
    axs[i].set_ylim(-5,8)
plt.suptitle(f'Recording {(np.where(np.array(files) == foi))[0][0]}')
fig.tight_layout(h_pad=0.45)
plt.savefig('/Users/kristineyoon/Documents/trialbytrial.pdf', transparent=True)



# ########################## CUE ALIGNMENT ##########################    
        
# cuestolevers={}
# for i in range(len(track_cue)):
#     for k in range(len(track_lever)):
#         if track_lever[k]- track_cue[i]<= cue_time and track_lever[k]- track_cue[i]>0:
#             cuestolevers[track_cue[i]]=track_lever[k]
            
# from scipy.signal import find_peaks           
# signalaftercue = []

# timerange_cue = [-5, 10]
# for i in range(len(track_cue)):
#     cue_zero = round(track_cue[i] * fs)
#     cue_baseline = cue_zero + timerange_cue[0] * fs
#     cue_end = cue_zero + timerange_cue[1] * fs
#     trial = np.array(df.iloc[cue_baseline:cue_end,2])
#     signalaftercue = np.array(df.iloc[cue_zero:cue_end,2])
#     peaks, properties = find_peaks(signalaftercue, width=100)
#     plt.plot(signalaftercue)
#     plt.plot(peaks[0],signalaftercue[peaks[0]], "x")
#     print(peaks)
# print (signalaftercue)
        
# ### DOWNSAMPLING
# N = 10
# sample_cue=[]
# for lst in align_cue: 
#     small_lst = []
#     for i in range(0, len(trial), N):
#         small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
#     sample_cue.append(small_lst)
    
# zscore_cue = []

# baselinedict = {}
# for i in range(len(sample_cue)):
#     trial = sample_cue[i]
#     zb = np.mean(trial[0:round((-timerange_cue[0]*fs/N))])
#     zsd = np.std(trial[0:round((-timerange_cue[0]*fs/N))])
#     baselinedict[track_cue[i]] = zb, zsd
#     trial = (trial - zb)/zsd
#     zscore_cue.append(trial)


# #### ALIGN TO CUE: AVERAGE LINE PLOT OF TRIALS ####
# plt.plot(np.mean(zscore_cue, axis=0), color='teal')
# plt.fill_between(np.arange(0,len(zscore_cue[0]),1),
#                 np.mean(zscore_cue, axis=0)+np.std(zscore_cue, axis=0), 
#                 np.mean(zscore_cue, axis=0)-np.std(zscore_cue, axis=0),
#                 facecolor='teal',
#                 alpha=0.2)
# plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
#            np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
# plt.axhline(y=0, linestyle=':', color='black')
# plt.xlabel('Cue Onset (s)')
    
# ### ALIGN TO FIRST PEAK AFTER CUE ###

# zscore_cuepeak = []
# baselinedict_alignpeakcue = {}
# for i in range(len(sample_cue)):
#     trial = sample_cue[i]
#     zb = np.mean(trial[0:round((-timerange_cue[0]*fs/N))])
#     zsd = np.std(trial[0:round((-timerange_cue[0]*fs/N))])
#     baselinedict_alignpeakcue[track_cue[i]] = zb, zsd
#     trial = (trial - zb)/zsd
#     zscore_cue.append(trial)

    
    
    
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
# ########################## ADD LATENCY ##########################
# cue_time = 30
# lick_time= 10

# track_cue = []
# track_lever = []
# track_licks  =[]
# latency= []
# leverpermice = []
# for i in range(len(fp_df)):
#     if fp_df.iloc[i,0] == 'Cue':
#         track_cue.append(fp_df.iloc[i,1])
#     if fp_df.iloc[i,0] == 'Press':
#         track_lever.append(fp_df.iloc[i,1])
#     if fp_df.iloc[i,0] == 'Licks':
#         track_licks.append(fp_df.iloc[i,1])

# lever_latency = {}
# for i in range(len(track_cue)):
#     lever_list=[]
#     for k in range (len(track_lever)):
#         if track_lever[k] - track_cue[i] <= cue_time and track_lever[k] - track_cue[i] > 0:
#             lever_list.append(track_lever[k] - track_cue[i])
#         lever_latency[i] = lever_list
# lick_latency = {}
# for i in range(len(lever_latency)):
#     lick_list = []
#     if len(lever_latency[i]) != 0:
#         for k in range (len(track_licks)):
#             if track_licks[k] - track_cue[i] <= cue_time and track_licks[k] - track_cue[i] > 0:
#                 lick_list.append(track_licks[k] - track_cue[i])
#             lick_latency[i] = lick_list
#     else:
#         lick_latency[i] = []

# df_lever = pd.Series(lever_latency)
# df_lick = pd.Series(lick_latency)

  
# ########################## INDIVIDUAL TRIAL TRACES TO CUE ##########################
# ntrials = len(zscore_cue)
# colors = sns.color_palette("tab20")
# listall = []
# graphit = 0

# fig, axs = plt.subplots(ntrials, 1, figsize=(12,ntrials))
# all_axes = fig.get_axes()
# for ax in all_axes:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

# for i in range(0,ntrials):
    
#     axs[i] = fig.add_subplot(ntrials,1,i+1)
#     axs_time = np.linspace(timerange_cue[0], timerange_cue[1], len(zscore_cue[i]))
#     axs[i].axvline(x=0, linewidth=1, color='black')
#     axs[i].axhline(y=0, linewidth=1, color='lightgrey')
#     if i < 20:
#         axs[i].plot(axs_time, np.array(zscore_cue[i]), linewidth=2, color=colors[i])
#     else:
#         axs[i].plot(axs_time, np.array(zscore_cue[i]), linewidth=2, color=colors[round(i-20)])
#     axs[i].set_ylabel('z-Score', labelpad = 2)
#     axs[i].set_xlim(timerange_cue[0], timerange_cue[1])
#     axs[i].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
#     axs[i].set_ylim(-5,15)

# fig.tight_layout(h_pad=0.45)

  
# ########################## INDIVIDUAL TRIAL TRACES TO LEVER ##########################
# ntrials = len(zscore_lever)
# colors = sns.color_palette("tab20")
# listall = []
# graphit = 0

# df_all = pd.DataFrame(columns=["Lever","Lick"])
# for i in range(0,ntrials):
#     df_all.loc[i]=[df_lever.iloc[i], df_lick.iloc[i]]

# fig, axs = plt.subplots(ntrials, 1, figsize=(12,ntrials))
# all_axes = fig.get_axes()
# for ax in all_axes:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

# for i in range(0,ntrials):
    
#     axs[i] = fig.add_subplot(ntrials,1,i+1)
#     axs_time = np.linspace(timerange_lever[0], timerange_lever[1], len(zscore_lever[i]))
#     axs[i].axvline(x=0, linewidth=1, color='black')
#     axs[i].axhline(y=0, linewidth=1, color='lightgrey')
#     if i < 20:
#         axs[i].plot(axs_time, np.array(zscore_lever[i]), linewidth=2, color=colors[i])
#     else:
#         axs[i].plot(axs_time, np.array(zscore_lever[i]), linewidth=2, color=colors[round(i-20)])
#     axs[i].set_ylabel('z-Score', labelpad = 2)
#     axs[i].set_xlim(timerange_lever[0], timerange_lever[1])
#     axs[i].set_xticks(np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int))
#     axs[i].set_ylim(-5,5)

# fig.tight_layout(h_pad=0.45)

#### TRIALS ON HEATMAP ALIGNED TO CUE ####
plt.figure(figsize=(5,3))
sns.heatmap(zscore_cue, cmap='RdBu', vmin=-5, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
            np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
            rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
plt.ylabel('Trials')
plt.xlabel('Cue Onset')


#### TRIALS ON HEATMAP ALIGNED TO LEVERPRESS ####
plt.figure(figsize=(5,3))
sns.heatmap(zscore_lever, cmap='RdBu', vmin=-5, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
            np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
            rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
plt.ylabel('Trials')
plt.xlabel('Lever Press Onset')

#### TRIALS ON HEATMAP ALIGNED TO LEVERPRESS ####
plt.figure(figsize=(5,3))
sns.heatmap(zscore_flick, cmap='RdBu', vmin=-5, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lever[1]-timerange_lever[0])), 
            np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
            rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
plt.ylabel('Trials')
plt.xlabel('Lever Press Onset')

# #### ALIGN TO CUE: AVERAGE LINE PLOT OF TRIALS ####
# plt.plot(np.mean(zscore_cue, axis=0), color='blue')
# plt.fill_between(np.arange(0,len(zscore_cue[0]),1),
#                 np.mean(zscore_cue, axis=0)+np.std(zscore_cue, axis=0), 
#                 np.mean(zscore_cue, axis=0)-np.std(zscore_cue, axis=0),
#                 facecolor='blue',
#                 alpha=0.2)
# plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
#            np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
# plt.xlabel('Cue Onset')

# #### ALIGN TO LEVER PRESS: AVERAGE LINE PLOT OF TRIALS ####
# plt.plot(np.mean(zscore_lever, axis=0), color='orange')
# plt.fill_between(np.arange(0,len(zscore_lever[0]),1),
#                  np.mean(zscore_lever, axis=0)+np.std(zscore_lever, axis=0), 
#                 np.mean(zscore_lever, axis=0)-np.std(zscore_lever, axis=0),
#                 facecolor='orange',
#                 alpha=0.2)
# plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
#            np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
# plt.xlabel('Lever Press Onset')


# #### RASTER PLOT OF EACH TRIAL ####
# fig, axs = plt.subplots(ntrials, 1, figsize=(12,ntrials))
# all_axes = fig.get_axes()
# for ax in all_axes:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

# for i in range(0,ntrials):
#     axs[i] = fig.add_subplot(ntrials,1,i+1)
#     axs[i].eventplot(df_all.loc[i], colors=['black','red'], lineoffsets=[0,0])
#     axs[i].set_ylabel('Events', labelpad = 2)
#     axs[i].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
#     axs[i].set(xlim=(timerange_cue[0], timerange_cue[1]), 
#                ylim=(-1,1))

# fig.tight_layout(h_pad=0.45)

######################################

# axs[4].eventplot(cue_df, colors='black', lineoffsets=leveroff)
# axs[4].set(xlim=(timerange_cue[0], timerange_cue[1]), 
#            ylim=(len(cue_df),0), 
#            yticks=np.arange(0.5,len(cue_df)+0.5))

# axs[5] = axs[2].twinx()
# lickoff = np.arange(0.5,len(lever_df)+0.5)
# axs[5].eventplot(lever_df, colors='black', lineoffsets=lickoff)
# axs[5].set(xlim=(timerange_lever[0], timerange_lever[1]), 
#            ylim=(len(lever_df),0), 
#            yticks=np.arange(0.5,len(lever_df)+0.5))



# lever_latency = sorted(lever_latency.items(), key=lambda x:x[1])
# lick_latency = sorted(lick_latency.items(), key=lambda x:x[1])

# ########################## SPLIT DATA TO RESPONSE V. NONRESPONSE ##########################
# response_cue = []
# nonresponse_cue = []
# for key,value in lever_latency:
#     if len(value) == 0:
#         nonresponse_cue.append(zscore_cue[key])
#     else:
#         response_cue.append(zscore_cue[key])

# ########################## CONVERTING LATENCY DATA TO DATAFRAMES ##########################

# rp_response = []
# for key,value in lever_latency:
#     if len(value) != 0:
#         rp_response.append(value)

# cue_df = pd.Series(rp_response)
# lever_df = pd.Series(rp_lick)

#####################################################################################################
################################## PLOTTING HEATMAP WITH EVENTPLOT ##################################
#####################################################################################################


# heightratio = [len(response_cue),len(nonresponse_cue), len(sorted_lick), len(zscore_lick), 0, 0]
# fig, axs = plt.subplots(6, 1, figsize=(12,20), gridspec_kw={'height_ratios':heightratio})
# all_axes = fig.get_axes()
# for ax in all_axes:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
# fig.tight_layout(h_pad=0.8)

# axs[0] = fig.add_subplot(6,1,1)
# heatmap_rescue = axs[0].imshow(response_cue, 
#                 cmap='RdBu', 
#                 vmin = -3, vmax = 3, 
#                 interpolation='none', aspect="auto",
#                 extent=[timerange_cue[0], timerange_cue[1], len(response_cue), 0])
# cbar = fig.colorbar(heatmap_rescue, pad=0.08, fraction=0.02)
# axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
# axs[0].set_ylabel('Trials')
# axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
# cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
# axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
# axs[0].set_yticks(np.arange(0, len(response_cue)+1,1, dtype=int))

# axs[1] = fig.add_subplot(6,1,2)
# heatmap_norescue = axs[1].imshow(nonresponse_cue, 
#                 cmap='RdBu', 
#                 vmin = -3, vmax = 3, 
#                 interpolation='none', aspect="auto",
#                 extent=[timerange_cue[0], timerange_cue[1], len(nonresponse_cue), 0])
# cbar = fig.colorbar(heatmap_norescue, pad=0.08, fraction=0.02)
# axs[1].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
# axs[1].set_ylabel('Trials')
# axs[1].set_xlabel('Seconds from Cue Onset (No Lever Press)')
# cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
# axs[1].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
# axs[1].set_yticks(np.arange(0, len(nonresponse_cue)+1,1, dtype=int))

# axs[2] = fig.add_subplot(6,1,3)
# heatmap_lever = axs[2].imshow(sorted_lick, 
#                 cmap='RdBu', 
#                 vmin = -3, vmax = 3, 
#                 interpolation='none', aspect="auto",
#                 extent=[timerange_lever[0], timerange_lever[1], len(sorted_lick), 0])
# cbar = fig.colorbar(heatmap_lever, pad=0.08, fraction=0.02)
# axs[2].axvline(x=0, linewidth=2, color='black', label='Lever Onset')
# axs[2].set_ylabel('Trials')
# axs[2].set_xlabel('Seconds from Lever Onset')
# cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
# axs[2].set_xticks(np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int))
# axs[2].set_yticks(np.arange(0, len(sorted_lick)+1,1, dtype=int))
 
# axs[3] = fig.add_subplot(6,1,4)
# heatmap_flick = axs[3].imshow(zscore_lick, 
#                 cmap='RdBu', 
#                 vmin = -3, vmax = 3, 
#                 interpolation='none', aspect="auto",
#                 extent=[timerange_lick[0], timerange_lick[1], len(zscore_lick), 0])
# cbar = fig.colorbar(heatmap_flick, pad=0.08, fraction=0.02)
# axs[3].axvline(x=0, linewidth=2, color='black', label='Lick Onset')
# axs[3].set_ylabel('Trials')
# axs[3].set_xlabel('Seconds from First Lick Onset')
# cbar.ax.set_ylabel('Delta F/F From Baseline', rotation=90)
# axs[3].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
# axs[3].set_yticks(np.arange(0, len(zscore_lick)+1,1, dtype=int))

# fig.tight_layout(h_pad=0.45)

# axs[4] = axs[0].twinx()
# leveroff = np.arange(0.5,len(cue_df)+0.5)
# axs[4].eventplot(cue_df, colors='black', lineoffsets=leveroff)
# axs[4].set(xlim=(timerange_cue[0], timerange_cue[1]), 
#            ylim=(len(cue_df),0), 
#            yticks=np.arange(0.5,len(cue_df)+0.5))

# axs[5] = axs[2].twinx()
# lickoff = np.arange(0.5,len(lever_df)+0.5)
# axs[5].eventplot(lever_df, colors='black', lineoffsets=lickoff)
# axs[5].set(xlim=(timerange_lever[0], timerange_lever[1]), 
#            ylim=(len(lever_df),0), 
#            yticks=np.arange(0.5,len(lever_df)+0.5))

# #####################################################################################################
# #####################################################################################################
# #####################################################################################################
# ###################################### PLOTTING AVG LINE PLOTS ######################################
# #####################################################################################################
# #####################################################################################################
# #####################################################################################################

# fig, axs = plt.subplots(4,1,figsize=(10,20))
# all_axes = fig.get_axes()
# for ax in all_axes:
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)

# axs[0] = fig.add_subplot(4,1,1)
# axs0_time = np.linspace(timerange_cue[0], timerange_cue[1], int(len(align_cue[0])/sample))
# axs[0].plot(axs0_time, np.mean(response_cue, axis=0), linewidth=2, color='orange')
# axs[0].fill_between(axs0_time, np.mean(response_cue, axis=0)+np.std(response_cue)
#                       ,np.mean(response_cue, axis=0)-np.std(response_cue), facecolor='orange', alpha=0.2)
# axs[0].axvline(x=0, linewidth=2, color='black', label='Cue Onset')
# axs[0].set_xlabel('Seconds from Cue Onset (Responsive)')
# axs[0].set_ylabel('z-Score', labelpad = 2)
# axs[0].set_xlim(timerange_cue[0], timerange_cue[1])
# axs[0].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
# axs[0].set_ylim(-3, 3)

# axs[1] = fig.add_subplot(4,1,2)
# axs1_time = np.linspace(timerange_cue[0], timerange_cue[1], int(len(align_cue[0])/sample))
# axs[1].plot(axs1_time, np.mean(nonresponse_cue, axis=0), linewidth=2, color='gold')
# axs[1].fill_between(axs1_time, np.mean(nonresponse_cue, axis=0)+np.std(nonresponse_cue)
#                       ,np.mean(response_cue, axis=0)-np.std(response_cue), facecolor='gold', alpha=0.2)
# axs[1].axvline(x=0, linewidth=2, color='black', label='Cue')
# axs[1].set_xlabel('Seconds from Cue Onset (No Lever Press)')
# axs[1].set_ylabel('z-Score', labelpad = 2)
# axs[1].set_xlim(timerange_cue[0], timerange_cue[1])
# axs[1].set_xticks(np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int))
# axs[1].set_ylim(-3, 3)

# axs[2] = fig.add_subplot(4,1,3)
# axs2_time = np.linspace(timerange_lever[0], timerange_lever[1], int(len(align_lever[0])/sample))
# axs[2].plot(axs2_time, np.mean(zscore_lever, axis=0), linewidth=2, color='teal')
# axs[2].fill_between(axs2_time, np.mean(zscore_lever, axis=0)+np.std(zscore_lever)
#                       ,np.mean(zscore_lever, axis=0)-np.std(zscore_lever), facecolor='teal', alpha=0.2)
# axs[2].axvline(x=0, linewidth=2, color='black', label='Lever')
# axs[2].set_xlabel('Seconds from Lever Onset')
# axs[2].set_ylabel('z-Score', labelpad = 2)
# axs[2].set_xlim(timerange_lever[0], timerange_lever[1])
# axs[2].set_xticks(np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int))
# axs[2].set_ylim(-3, 3)

# axs[3] = fig.add_subplot(4,1,4)
# axs3_time = np.linspace(timerange_lick[0], timerange_lick[1], int(len(align_lick[0])/sample))
# axs[3].plot(axs3_time, np.mean(zscore_lick, axis=0), linewidth=2, color='indigo')
# axs[3].fill_between(axs3_time, np.mean(zscore_lick, axis=0)+np.std(zscore_lick)
#                       ,np.mean(zscore_lick, axis=0)-np.std(zscore_lick), facecolor='indigo', alpha=0.2)
# axs[3].axvline(x=0, linewidth=2, color='black', label='First Lick')
# axs[3].set_xlabel('Seconds from First Lick Onset')
# axs[3].set_ylabel('z-Score', labelpad = 2)
# axs[3].set_xlim(timerange_lick[0], timerange_lick[1])
# axs[3].set_xticks(np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int))
# axs[3].set_ylim(-3, 3)
