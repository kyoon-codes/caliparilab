import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy


# Define function to calculate SEM
def sem(arr):
    return np.std(arr, axis=0) / np.sqrt(len(arr))

# Define function to find lick bouts
max_interval = 0.4
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


# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/Omission/'
# mice = ['7098','7107','7108','7296', '7310', '7311', '7319', '7321']

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/All/'
mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/SucrosePretreat'
# mice = ['7098','7107','7108']


files = os.listdir(folder)
files.sort()
print(files)
lick_rasterdata = {}
avgcuetrace_dict = {}
avgresptrace_dict = {}
avglevertrace_dict = {}
avgflicktrace_dict = {}
avglickbouttrace_dict = {}
lickbouttracedict ={}
allcuetraces = {}
bout_traces = {}
bout_peak_heights = {}
bout_peak_indices = {}
timerange_cue = [-2, 5]
timerange_lever = [-2, 5]
timerange_lick = [-2, 10]
totalactivetrace_dict = {}
active_time = [-2,20]
cue_time = 20
lick_time= 10
N = 100

colors10 = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61','#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']

auc_lickbout = {}

for mouse in mice:
    mouse_dir = os.path.join(folder, mouse)

    Dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
    Dates.sort()
    for date in Dates:
        date_dir = os.path.join(mouse_dir, date)
        data = tdt.read_block(date_dir)
        print(date_dir)
        df = pd.DataFrame()
        df['Sig465'] = data.streams._465B.data
        df['Sig405'] = data.streams._405B.data
        df['Dff'] = (df['Sig465']-df['Sig405'])/df['Sig465']
        fs = round(data.streams._465B.fs)

        split1 = str(data.epocs).split('\t')
        y = []
        for elements in split1:
            x = elements.split('\n')
            if '[struct]' in x:
                x.remove('[struct]')
            y.append(x)
        z= [item for sublist in y for item in sublist]

        fp_df = pd.DataFrame(columns=['Event','Timestamp'])
        events = ['Cue', 'Press', 'Licks', 'Timeout Press']
        epocs = ['Po0_','Po6_','Po4_','Po2_']
        
        for a, b in zip(events, epocs):
            if b in z:
                event_df = pd.DataFrame(columns=['Event','Timestamp'])
                event_df['Timestamp'] = data.epocs[b].onset
                event_df['Event'] = a
                fp_df = pd.concat([fp_df, event_df])
        
        track_cue = []
        track_lever = []
        track_licks = []
        track_to = []
        latency= []
        leverpermice = []
        for i in range(len(fp_df)):
            if fp_df.iloc[i,0] == 'Cue':
                track_cue.append(fp_df.iloc[i,1])
            if fp_df.iloc[i,0] == 'Press':
                track_lever.append(fp_df.iloc[i,1])
            if fp_df.iloc[i,0] == 'Licks':
                track_licks.append(fp_df.iloc[i,1])
            if fp_df.iloc[i,0] == 'Timeout Press':
                track_to.append(fp_df.iloc[i,1])
                
        

        ########################## CUE ALIGNMENT ##########################    

        for i in range(len(track_cue)):
            cue_zero = round(track_cue[i] * fs)
            cue_baseline = cue_zero + timerange_cue[0] * fs
            cue_end = cue_zero + timerange_cue[1] * fs
            
            aligntobase = np.mean(df.iloc[cue_baseline:cue_zero,2])
            rawtrial = np.array(df.iloc[cue_baseline:cue_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            avgcuetrace_dict[mouse,Dates.index(date),i]= track_cue[i], sampletrial

        
        ########################## LEVER ALIGNMENT ##########################
        for i in range(len(track_lever)):
            lever_zero = round(track_lever[i] * fs)
            lever_baseline = lever_zero + timerange_lever[0] * fs
            lever_end = lever_zero + timerange_lever[1] * fs
            
            aligntobase = np.mean(df.iloc[lever_baseline:lever_zero,2])
            rawtrial = np.array(df.iloc[lever_baseline:lever_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for trial_num in range(len(track_cue)):
                if track_lever[i] - track_cue[trial_num] > 0 and track_lever[i] - track_cue[trial_num] < 20:
                    cue_trial = trial_num
            
            avglevertrace_dict[mouse,Dates.index(date),cue_trial]= track_lever[i], sampletrial

        ################## FIRST LICK ALIGNMENT ################3
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

        for i in range(len(track_flicks)):
            flick_zero = round(track_flicks[i] * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs
            
            aligntobase = np.mean(df.iloc[flick_baseline:flick_zero,2])
            rawtrial = np.array(df.iloc[flick_baseline:flick_end,2])

            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for trial_num in range(len(track_cue)):
                if track_flicks[i] - track_cue[trial_num] > 0 and track_flicks[i] - track_cue[trial_num] < 30:
                    cue_trial = trial_num
            avgflicktrace_dict[mouse,Dates.index(date),cue_trial]= track_flicks[i], sampletrial
        
        
        ############ LICKBOUT ALIGNMENT #################
        # Identify and sort lick bouts by length
        sorted_lick_bouts = identify_and_sort_lick_bouts(track_licks, max_interval)

        for length, bout in sorted_lick_bouts:
            start_time = bout[0]
            lickbout_each = []
            for items in bout:
                licklick = items-start_time
                lickbout_each.append(licklick)
            lick_rasterdata[mouse,Dates.index(date), length]=lickbout_each
            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero + timerange_lick[0] * fs
            lickb_end = lickb_zero + timerange_lick[1] * fs
            
            aligntobase = np.mean(df.iloc[lickb_baseline:lickb_zero,2])
            rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for i in range(len(track_cue)):
                if start_time - track_cue[i] > 0 and start_time - track_cue[i] < 30:
                    trial_num = i
            bout_traces[mouse, Dates.index(date),trial_num, length]=(sampletrial)
            avglickbouttrace_dict[mouse,Dates.index(date),trial_num]= start_time, length, sampletrial


        ########################## ALL ALIGNMENT ##########################    
        
        for i in range(len(track_cue)):
            cue_zero = round(track_cue[i] * fs)
            cue_baseline = cue_zero + timerange_cue[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            levertime = np.nan
            flicktime = np.nan
            aligntobase = np.mean(df.iloc[cue_baseline:cue_zero,2])
            rawtrial = np.array(df.iloc[cue_baseline:cue_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for l in range(len(track_lever)):
                if  track_lever[l] - track_cue[i] > 0 and track_lever[l] - track_cue[i] < 20:
                    levertime = track_lever[l]

            for m in range(len(track_flicks)):
                if track_flicks[m] - track_cue[i] > 0 and track_flicks[m] - track_cue[i] < 30:
                    flicktime = track_flicks[m]
                    
            totalactivetrace_dict[mouse,Dates.index(date),i]= track_cue[i], levertime, flicktime, sampletrial

        


##############################################################
### FIRST CHECK POINT
##############################################################
#by session individual
# ['7098','7099','7107','7108','7296','7310','7311','7319','7321','7329']

fig, axs = plt.subplots(8, sharex=True, figsize=(10,12))
for i in range(8):
    for mouse,session,trial in avgcuetrace_dict:
        if mouse in ['7329']:
            if session == i:
                axs[i].plot(avgcuetrace_dict[str(mouse),session,trial][1],color=colors10[int(session)],label=session)
                axs[i].axvline(x=len(avgcuetrace_dict[str(mouse),session,trial][1])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=0.5, color='black')
plt.xticks(np.arange(0,len(avgcuetrace_dict[str(mouse),session,trial][1])+1,len(avgcuetrace_dict[str(mouse),session,trial][1])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.xlabel('Cue Onset (s)')
fig.tight_layout()

##############################################################
#by session individual
# ['7098','7099','7107','7108','7296','7310','7311','7319','7321','7329']

fig, axs = plt.subplots(8, sharex=True)
for i in range(8):
    for mouse,session,trial in avglevertrace_dict:
        if mouse in ['7296']:
            if session == i:
                axs[i].plot(avglevertrace_dict[str(mouse),session,trial][1],color=colors10[int(session)],label=session)
                axs[i].axvline(x=len(avglevertrace_dict[str(mouse),session,trial][1])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=0.5, color='black')
plt.xticks(np.arange(0,len(avglevertrace_dict[str(mouse),session,trial][1])+1,len(avglevertrace_dict[str(mouse),session,trial][1])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.legend()
plt.xlabel('Lever Press Onset (s)')

##############################################################
#by session individual
# ['7098','7099','7107','7108','7296','7310','7311','7319','7321','7329']

fig, axs = plt.subplots(8, sharex=True)
for i in range(8):
    for mouse,session,trial in avgflicktrace_dict:
        if mouse in ['7296']:
            if session == i:
                axs[i].plot(avgflicktrace_dict[str(mouse),session,trial][1],color=colors10[int(session)],label=session)
                axs[i].axvline(x=len(avgflicktrace_dict[str(mouse),session,trial][1])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.xticks(np.arange(0,len(avgflicktrace_dict[str(mouse),session,trial][1])+1,len(avgflicktrace_dict[str(mouse),session,trial][1])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.legend()
plt.xlabel('First Lick Onset (s)')




##############################################################
### HEAT MAP
##############################################################


#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by latency to lick ####

for mouse in mice:
    alltraces=[]
    for i in range(8):
        bysession = {}
        for subj,session,trial in totalactivetrace_dict:
            if session == i and subj == mouse and trial in range (10):
                if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
                    bysession[totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
                else:
                    bysession[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
        sorted_bysession=[]
        for i in sorted(bysession.keys()):
            #print(i)
            sorted_bysession.append(bysession[i])
        alltraces.append(sorted_bysession)
    
    fig, axs = plt.subplots(8, sharex=True)
    for i in range (8):
        bysessiondf= pd.DataFrame(alltraces[i])    
        sns.heatmap(bysessiondf, cmap='RdBu', vmin=-5, vmax=5, ax=axs[i])
        axs[i].axvline(x=(bysessiondf.shape[1])/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black', label='Cue Onset')
    plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2), 
                np.arange(active_time[0], active_time[1],2, dtype=int),
                rotation=0)
    
    plt.ylabel('Trials')
    plt.xlabel('Time (sec)')
    plt.show()

#### TRIALS ON HEATMAP ALIGNED TO CUE of all session sorted by latency to lick ####

for mouse in mice:
    alltraces={}
    for i in range(8):
        for subj,session,trial in totalactivetrace_dict:
            if session == i and subj == mouse and trial in range (10):
                if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
                    continue
                else:
                    alltraces[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
    
    sorted_bylatency=[]
    for i in sorted(alltraces.keys()):
        sorted_bylatency.append(alltraces[i])
    
    plt.figure()
    bysessiondf= pd.DataFrame(sorted_bylatency)    
    sns.heatmap(bysessiondf, cmap='RdBu', vmin=-5, vmax=5, )
    plt.axvline(x=(bysessiondf.shape[1])/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black', label='Cue Onset')
    plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2), 
                np.arange(active_time[0], active_time[1],2, dtype=int),
                rotation=0)
    plt.title(f'{mouse}')
    plt.ylabel('Trials')
    plt.xlabel('Time (sec)')
    plt.show()


#### TRIALS ON HEATMAP ALIGNED TO CUE of all session sorted by latency to lick ####

for mouse in mice:
    alltraces={}
    for i in range(8):
        for subj,session,trial in totalactivetrace_dict:
            if session == i and subj == mouse and trial in range (10):
                if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
                    continue
                else:
                    alltraces[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
    
    sorted_bylatency=[]
    for i in sorted(alltraces.keys()):
        sorted_bylatency.append(alltraces[i])

    
    plt.figure()
    bysessiondf= pd.DataFrame(sorted_bylatency)    
    sns.heatmap(bysessiondf, cmap='RdBu', vmin=-5, vmax=5, )
    plt.axvline(x=(bysessiondf.shape[1])/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black', label='Cue Onset')
    plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2), 
                np.arange(active_time[0], active_time[1],2, dtype=int),
                rotation=0)
    plt.title(f'{mouse}')
    plt.ylabel('Trials')
    plt.xlabel('Time (sec)')
    plt.show()

#### TRIALS ON HEATMAP ALIGNED TO CUE of all mouse sorted by latency to lick ####


alltraces = {}

for subj, session, trial in totalactivetrace_dict:
    if trial in range(10):
        if np.isnan(totalactivetrace_dict[subj, session, trial][2]):
            continue
        else:
            alltraces[totalactivetrace_dict[subj, session, trial][2]-totalactivetrace_dict[subj,
                                                                                           session, trial][0]] = totalactivetrace_dict[subj, session, trial][3]

sorted_bylatency = []
for i in reversed(sorted(alltraces.keys())):
    sorted_bylatency.append(alltraces[i])


plt.figure(figsize=(10, 12))
bysessiondf = pd.DataFrame(sorted_bylatency)
sns.heatmap(bysessiondf, cmap=sns.diverging_palette(255, 28, l=68, as_cmap=True), vmin=-5, vmax=5 )
plt.axvline(x=(bysessiondf.shape[1])/(active_time[1]-active_time[0])
            * (0-active_time[0]), linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0, bysessiondf.shape[1]+1, (bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2),
           np.arange(active_time[0], active_time[1], 2, dtype=int),
           rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time (sec)')
plt.show()
plt.savefig(
    '/Users/kristineyoon/Documents/heatmaptracesallmice.pdf', transparent=True)



##############################################################
### BEHAVIORAL PARADIGMS
##############################################################
# Latency to press

latency_to_leverpress = {}
for  cue_mouse, cue_session, cue_trial in avgcuetrace_dict:
    cue_time = avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0]
    for lever_mouse, lever_session, lever_trial in avglevertrace_dict:
        if cue_mouse == lever_mouse and cue_session == lever_session:
            if lever_trial - cue_time > 0 and lever_trial - cue_time < 20:
                latency_to_leverpress[cue_mouse, cue_session, cue_trial] = lever_trial - cue_time

plt.figure()
for i in range (8):
    mean_session=[]
    for mouse, session, trial in latency_to_leverpress:
        if session == i:
            mean_session.append(latency_to_leverpress[mouse, session,trial])
            plt.scatter(x=session,y= latency_to_leverpress[mouse, session, trial],color=colors10[i], alpha=0.1)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Latency to Lever Press (s)')
plt.show()



plt.figure()
for i in range (10):
    mean_trial=[]
    for mouse, session, trial in latency_to_leverpress:
        if trial == i:
            plt.scatter(x=trial,y= latency_to_leverpress[mouse, session, trial], color=colors10[session], alpha=0.3)
            mean_trial.append(latency_to_leverpress[mouse, session,trial])
    plt.scatter(x=i,y=np.mean(mean_trial),color=colors10[i])
    plt.errorbar(x=i,y=np.mean(mean_trial),yerr=sem(mean_trial),ecolor=colors10[i])
plt.xlabel('Trial in Each Session')
plt.ylabel('Latency to Lever Press (s)')
plt.show


plt.figure()
for i in range (10):
    mean_trial=[]
    for mouse, session, trial in avglickbouttrace_dict:
        if session == i:
            plt.scatter(x=session,y= avglickbouttrace_dict[mouse, session, trial][1], color=colors10[session], alpha=0.3)
            mean_trial.append(avglickbouttrace_dict[mouse, session,trial][1])
    plt.scatter(x=i,y=np.mean(mean_trial),color=colors10[i])
    plt.errorbar(x=i,y=np.mean(mean_trial),yerr=sem(mean_trial),ecolor=colors10[i])
plt.xlabel('Trial in Each Session')
plt.ylabel('Lick Bouts')
plt.show

##############################################################
### finding peak heigth
##############################################################

allcuepeakheights={}
from scipy.signal import find_peaks         
peakheight_cue_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
plt.figure(figsize=(10, 6))
for mouse,session,trial in avgcuetrace_dict:
    if mouse in mice:
        if session in range(8):
            if trial in range (10):
                avgtrace = avgcuetrace_dict[str(mouse),session,trial][1]
                param_thresh= 0.0
                param_prom= 0.6
                peaks, properties = find_peaks(avgtrace, threshold = param_thresh, prominence=param_prom, height=.01)
                plt.plot(avgtrace, label=f'Session {session}', alpha=0.2)
                maxpeak=0
                for k in range(len(peaks)):
                    if peaks[k] > len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(0.3-timerange_cue[0]) and peaks[k] < len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(1-timerange_cue[0]):
                        if properties['peak_heights'][k] > maxpeak:
                            maxpeak = properties['peak_heights'][k]
                            time = peaks[k]
                            plt.scatter(x=peaks[k], y= properties['peak_heights'][k], alpha=0.8)
                
                num = len(peakheight_cue_df)
                peakheight_cue_df.at[num,'Mouse']=mouse
                peakheight_cue_df.at[num,'Session']=session
                peakheight_cue_df.at[num,'Trial']=trial
                if maxpeak > 0:
                    peakheight_cue_df.at[num,'X-Axis']= time
                    peakheight_cue_df.at[num,'PeakHeight']= maxpeak
                    allcuepeakheights[mouse,session,trial] = maxpeak
                else:
                    peakheight_cue_df.at[num,'X-Axis']= np.nan
                    peakheight_cue_df.at[num,'PeakHeight']= np.nan
                    allcuepeakheights[mouse,session,trial] = maxpeak
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(avgtrace)+1,len(avgtrace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Cue-aligned Trace with SEM by Session')
plt.show()

plt.figure()
cuepeakheight = []
for i in range(8):
    session_peakheight = []
    for mouse,session,trial in allcuepeakheights:
        if session == i:
            session_peakheight.append(allcuepeakheights[mouse,session,trial])
            plt.scatter(x=session,y= allcuepeakheights[mouse,session,trial], color=colors10[session], alpha=0.05)
    cuepeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylim(0,8)
plt.ylabel('Peak Height At Cue')
plt.show()


cue_df=pd.DataFrame()
for i in range(8):
    num = 0
    for mouse,session,trial in allcuepeakheights:
        if session == i:
            cue_df.at[num,i]=allcuepeakheights[mouse,session,trial]
            num = num + 1





allleverpeakheights={}
from scipy.signal import find_peaks         
peakheight_lever_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
plt.figure(figsize=(10, 6))
for mouse,session,trial in avglevertrace_dict:
    if mouse in mice:
        if session in range(8):
            if trial in range (10):
                avgtrace = avglevertrace_dict[str(mouse),session,trial][1]
                param_thresh= 0.0
                param_prom= 0.6
                peaks, properties = find_peaks(avgtrace, threshold = param_thresh, prominence=param_prom, height=.01)
                plt.plot(avgtrace, label=f'Session {session}', alpha=0.2)
                maxpeak=0
                for k in range(len(peaks)):
                    if peaks[k] > len(avgtrace)/(timerange_lever[1]-timerange_lever[0])*(-0.5-timerange_lever[0]) and peaks[k] < len(avgtrace)/(timerange_lever[1]-timerange_lever[0])*(0.5-timerange_lever[0]):
                        if properties['peak_heights'][k] > maxpeak:
                            maxpeak = properties['peak_heights'][k]
                            time = peaks[k]
                            plt.scatter(x=peaks[k], y= properties['peak_heights'][k], alpha=0.8)
                
                num = len(peakheight_lever_df)
                peakheight_lever_df.at[num,'Mouse']=mouse
                peakheight_lever_df.at[num,'Session']=session
                peakheight_lever_df.at[num,'Trial']=trial
                if maxpeak > 0:
                    peakheight_lever_df.at[num,'X-Axis']= time
                    peakheight_lever_df.at[num,'PeakHeight']= maxpeak
                    allleverpeakheights[mouse,session,trial] = maxpeak
                else:
                    peakheight_lever_df.at[num,'X-Axis']= np.nan
                    peakheight_lever_df.at[num,'PeakHeight']= np.nan
                    allleverpeakheights[mouse,session,trial] = maxpeak
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(avgtrace)+1,len(avgtrace)/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(avgtrace)/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lever-aligned Trace with SEM by Session')
plt.show()

plt.figure()
leverpeakheight = []
for i in range(8):
    session_peakheight = []
    for mouse,session,trial in allleverpeakheights:
        if session == i:
            session_peakheight.append(allcuepeakheights[mouse,session,trial])
            plt.scatter(x=session,y= allcuepeakheights[mouse,session,trial], color=colors10[session], alpha=0.05)
    leverpeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylim(0,8)
plt.ylabel('Peak Height At Lvever')
plt.show()


lever_df=pd.DataFrame()
for i in range(8):
    num = 0
    for mouse,session,trial in allleverpeakheights:
        if session == i:
            lever_df.at[num,i]=allcuepeakheights[mouse,session,trial]
            num = num + 1





alllickpeackheight={}
plt.figure(figsize=(10, 6))
peakheight_lickbout_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
for mouse,session,trial in avglickbouttrace_dict:
    if mouse in mice:
        if session in range(8):
            if trial in range (10):
                avgtrace = avglickbouttrace_dict[str(mouse),session,trial][2]
                param_thresh= 0.0
                param_prom= 0.6
                peaks, properties = find_peaks(avgtrace, threshold = param_thresh, prominence=param_prom, height=.01)
                plt.plot(avgtrace, label=f'Session {session}', alpha=0.2)
                maxpeak=0
                for k in range(len(peaks)):
                    if peaks[k] > len(avgtrace)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]) and peaks[k] < len(avgtrace)/(timerange_lick[1]-timerange_lick[0])*(1.5-timerange_lick[0]):
                        if properties['peak_heights'][k] > maxpeak:
                            maxpeak = properties['peak_heights'][k]
                            time = peaks[k]
                            plt.scatter(x=peaks[k], y= properties['peak_heights'][k], alpha=0.8)
                
                num = len(peakheight_lickbout_df)
                peakheight_lickbout_df.at[num,'Mouse']=mouse
                peakheight_lickbout_df.at[num,'Session']=session
                peakheight_lickbout_df.at[num,'Trial']=trial
                if maxpeak > 0:
                    peakheight_lickbout_df.at[num,'X-Axis']= time
                    peakheight_lickbout_df.at[num,'PeakHeight']= maxpeak
                    alllickpeackheight[mouse,session,trial] = maxpeak
                else:
                    peakheight_lickbout_df.at[num,'X-Axis']= np.nan
                    peakheight_lickbout_df.at[num,'PeakHeight']= np.nan
                    #alllickpeackheight[mouse,session,trial] = maxpeak
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(avgtrace)+1,len(avgtrace)/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(avgtrace)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Cue-aligned Trace with SEM by Session')
plt.show()


plt.figure()
flickpeakheight = []
for i in range(8):
    session_peakheight = []
    for mouse,session,trial in alllickpeackheight:
        if session == i:
            session_peakheight.append(alllickpeackheight[mouse,session,trial])
            plt.scatter(x=session,y= alllickpeackheight[mouse,session,trial], color=colors10[session], alpha=0.05)
    flickpeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylim(0,8)
plt.ylabel('Peak Height At First Lick')
plt.show()

flick_df=pd.DataFrame()
for i in range(8):
    num = 0
    for mouse,session,trial in alllickpeackheight:
        if session == i:
            flick_df.at[num,i]=alllickpeackheight[mouse,session,trial]
            num = num + 1

## does  lick bout show anything about peak height of cue after ####
plt.figure()
x=[]
y=[]
num =0
cue_to_bout = pd.DataFrame(columns=['Mouse','Session','Trial','PeakHeight','NextBout'])
for cue_mouse, cue_session, cue_trial in avglickbouttrace_dict:
    if (cue_mouse, cue_session, cue_trial) in allcuepeakheights.keys():
        x.append(allcuepeakheights[cue_mouse, cue_session, cue_trial])
        y.append(avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1])
        cue_to_bout.at[num,'PeakHeight'] = allcuepeakheights[cue_mouse, cue_session, cue_trial]
        cue_to_bout.at[num,'NextBout'] = avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1]
        cue_to_bout.at[num,'Mouse'] = cue_mouse
        cue_to_bout.at[num, 'Session']=cue_session
        cue_to_bout.at[num, 'Trial'] =cue_trial
        num = num + 1
        plt.scatter(x=allcuepeakheights[cue_mouse, cue_session, cue_trial], y=avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1], color=colors10[cue_session])
if len(x) > 0:
    a,b=np.polyfit(x,y,1)
    plt.plot(np.array(x), a*np.array(x)+b)
plt.xlabel('Peak Height')
plt.ylabel('Lick Bout')
plt.show()


## does  lick peack height show anything about peak height of cue after ####
plt.figure()
x=[]
y=[]
num =0
cue_to_lick = pd.DataFrame(columns=['Mouse','Session','Trial','PeakHeight','NextBout'])
for cue_mouse, cue_session, cue_trial in alllickpeackheight:
    if (cue_mouse, cue_session, cue_trial) in allcuepeakheights.keys():
        x.append(allcuepeakheights[cue_mouse, cue_session, cue_trial])
        y.append(alllickpeackheight[cue_mouse, cue_session, cue_trial])
        cue_to_lick.at[num,'PeakHeight'] = allcuepeakheights[cue_mouse, cue_session, cue_trial]
        cue_to_lick.at[num,'NextBout'] = alllickpeackheight[cue_mouse, cue_session, cue_trial]
        cue_to_lick.at[num,'Mouse'] = cue_mouse
        cue_to_lick.at[num, 'Session']=cue_session
        cue_to_lick.at[num, 'Trial'] =cue_trial
        num = num + 1
        plt.scatter(x=allcuepeakheights[cue_mouse, cue_session, cue_trial], y=alllickpeackheight[cue_mouse, cue_session, cue_trial], color=colors10[cue_session])
if len(x) > 0:
    a,b=np.polyfit(x,y,1)
    plt.plot(np.array(x), a*np.array(x)+b)
plt.xlabel('Cue Peak Height')
plt.ylabel('Lick Bout Peak Height')
plt.show()

    
##### does previous lick bout show anything about peak height of cue after ####
plt.figure()
x=[]
y=[]
num = 0
bout_to_cue = pd.DataFrame(columns=['Mouse','Session','Trial','PeakHeight','PreviousBout'])
for cue_mouse, cue_session, cue_trial in avglickbouttrace_dict:
    next_cue = cue_trial+1
    if (cue_mouse, cue_session, next_cue) in allcuepeakheights.keys():
        x.append(allcuepeakheights[cue_mouse, cue_session, next_cue])
        y.append(avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1])
        bout_to_cue.at[num,'PeakHeight'] = allcuepeakheights[cue_mouse, cue_session, next_cue]
        bout_to_cue.at[num,'PreviousBout'] = avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1]
        bout_to_cue.at[num,'Mouse'] = cue_mouse
        bout_to_cue.at[num, 'Session']=cue_session
        bout_to_cue.at[num, 'Trial'] =cue_trial
        num = num + 1
        plt.scatter(x=allcuepeakheights[cue_mouse, cue_session, next_cue], y=avglickbouttrace_dict[cue_mouse, cue_session, cue_trial][1], color=colors10[cue_session])
if len(x) > 0:
    a,b=np.polyfit(x,y,1)
    plt.plot(np.array(x), a*np.array(x)+b)
plt.xlabel('Peak Height')
plt.ylabel('Previous Lick Bout')
plt.show()


##############################################################
### responding cue height v. session
##############################################################
respondingcue_peakheights = {}
for cue_mouse, cue_session, cue_trial in latency_to_leverpress:
    for mouse, session, trial in allpeakheights:
        if cue_mouse == mouse and cue_session==session and cue_trial==trial:
            respondingcue_peakheights[cue_mouse, cue_session, cue_trial]=allpeakheights[mouse, session, trial]


    
for i in range (8):
    mean_peakheight=[]
    for mouse, session, trial in respondingcue_peakheights:
        if session == i:
            mean_peakheight.append(respondingcue_peakheights[mouse, session,trial])
            plt.scatter(x=session,y= respondingcue_peakheights[mouse, session, trial],color=colors10[i], alpha=0.2)
    plt.scatter(x=i,y=np.mean(mean_peakheight),color=colors10[i])
    plt.errorbar(x=i,y=np.mean(mean_peakheight),yerr=sem(mean_peakheight),ecolor=colors10[i])
plt.xlabel('Session')
plt.ylabel('Peak Height')
plt.show


colors10 = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61','#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
plt.figure()
for i in range (10):
    mean_peakheight=[]
    for mouse, session, trial in respondingcue_peakheights:
        if session >=0 and trial == i:
            plt.scatter(x=trial,y= respondingcue_peakheights[mouse, session, trial], color=colors10[session], alpha=0.3)
            mean_peakheight.append(respondingcue_peakheights[mouse, session,trial])
    plt.scatter(x=i,y=np.mean(mean_peakheight),color=colors10[i])
    plt.errorbar(x=i,y=np.mean(mean_peakheight),yerr=sem(mean_peakheight),ecolor=colors10[i])
plt.xlabel('Trial in Each Session')
plt.ylabel('Peak Height')
plt.show


plt.figure()
for i in range(0,8):
    x=[]
    y=[]
    for mouse, session, trial in respondingcue_peakheights:
        for cue_mouse, cue_session, cue_trial in latency_to_leverpress:
            if mouse == cue_mouse and session==cue_session and trial==cue_trial:
                if session == i:
                    x.append(respondingcue_peakheights[mouse, session, trial])
                    y.append(latency_to_leverpress[mouse, session, trial])
                    plt.scatter(x=respondingcue_peakheights[mouse, session, trial],y= latency_to_leverpress[mouse, session, trial], color=colors10[session], alpha=0.5)
    if len(x) != 0:
        a,b=np.polyfit(x,y,1)
        plt.plot(np.array(x), a*np.array(x)+b,color=colors10[i], label=i)
plt.xlabel('Peak Height at Cue')
plt.ylabel('Latency to Press')
plt.legend()
plt.show()


plt.figure()
for i in range(0,8):
    x=[]
    y=[]
    for mouse, session, trial in respondingcue_peakheights:
        for cue_mouse, cue_session, cue_trial in avglickbouttrace_dict:
            if mouse == cue_mouse and session==cue_session and trial==cue_trial:
                if session == i:
                    x.append(respondingcue_peakheights[mouse, session, trial])
                    y.append(avglickbouttrace_dict[mouse, session, trial][1])
                    plt.scatter(x=respondingcue_peakheights[mouse, session, trial],y= avglickbouttrace_dict[mouse, session, trial][1], color=colors10[session], alpha=0.5)
    if len(x) != 0:
        a,b=np.polyfit(x,y,1)
        plt.plot(np.array(x), a*np.array(x)+b,color=colors10[i], label=i)
plt.xlabel('Peak Height at Cue')
plt.ylabel('Lick Bouts')
plt.legend()
plt.show()


##############################################################
### by all session
##############################################################
#by session average
session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace = value
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    #if session in [0,1]:
    if session in range(0,8):
        plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.title('Average Cue Aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/cuebysessions.pdf', transparent=True)


##############################################################
#by session average
session_data = {}  
for key, value in avglevertrace_dict.items():
    mouse, session, time = key
    trialtrace = value
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    if session in range(0,8):
        plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lever Press Aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/leverbysessions.pdf', transparent=True)

##############################################################
#by session average
session_data = {}  
for key, value in avgflicktrace_dict.items():
    mouse, session, trial = key
    time, trialtrace = value
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    if session in range(3):
        plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average First Lick Aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/flickbysessions.pdf', transparent=True)

##############################################################
######################## BOUT ANALYSIS #######################
##############################################################

#RASTER PLOT
tinybouts = []
smallbouts = []
mediumbouts = []
longbouts = []
colors = sns.color_palette("husl", 8)
lickbouts_rasterplot={}

for key, licks in lick_rasterdata.items():
    mouse, session, licklength = key
    if licklength not in lickbouts_rasterplot:
        lickbouts_rasterplot[licklength] = []
    lickbouts_rasterplot[licklength].append(licks)

for boutlength, lickbouts in lickbouts_rasterplot.items():
    if boutlength > 10 and boutlength <30:
        for i in range(len(lickbouts)):
            tinybouts.append(lickbouts[i])
    if boutlength > 29 and boutlength <50:
        for i in range(len(lickbouts)):
            smallbouts.append(lickbouts[i])
    if boutlength > 49 and boutlength <70:
        for i in range(len(lickbouts)):
            mediumbouts.append(lickbouts[i])
    if boutlength > 70:
        for i in range(len(lickbouts)):
            longbouts.append(lickbouts[i])
fig, axs = plt.subplots(4, sharex=True)
for i in range(0,len(tinybouts)):
    position = i
    axs[0].eventplot(tinybouts[i], lineoffsets=position, color=colors[0])
for i in range(0,len(smallbouts)):
    position = i
    axs[1].eventplot(smallbouts[i], lineoffsets=position, color=colors[1])
for i in range(0,len(mediumbouts)):
    position = i
    axs[2].eventplot(mediumbouts[i], lineoffsets=position, color=colors[3])
for i in range(0,len(longbouts)):
    position = i
    axs[3].eventplot(longbouts[i], lineoffsets=position, color=colors[5])
plt.xlim(timerange_lick)
plt.savefig('/Users/kristineyoon/Documents/lickraster.pdf', transparent=True)

# RASTER PLOT
def find_first_above_zero(x, y, start_time, end_time):
    """
    Finds the first (x, y) point where y > 0 within the specified time range.
    :param x: List of x values (time points).
    :param y: List of y values (trace values).
    :param start_time: Start of the time range.
    :param end_time: End of the time range.
    :return: The first (x, y) point where y > 0 within the time range, or (NaN, NaN) if not found.
    """
    for xi, yi in zip(x, y):
        if start_time <= xi <= end_time and yi > 0:
            return xi, yi
    return np.nan, np.nan

tinybouts = []
smallbouts = []
mediumbouts = []
longbouts = []
longlongbouts = []
peakheight_range = [0,1.5]
peakheight_bouts={}
offset_bouts={}
offset_range=[2,10]
timeline=np.arange(timerange_lick[0], timerange_lick[1],N/fs)
colors = sns.color_palette("husl", 8)

for key, trace in bout_traces.items():
    mouse, date, trial, licklength = key
    offset_t=find_first_above_zero(timeline, trace, 
                          ((offset_range[0]-timerange_lick[0])), 
                          ((offset_range[1]-timerange_lick[0])))
    offset_bouts[mouse,licklength]=offset_t
    
    if len(trace) == len(avgtrace):
        
        param_thresh= 0.0
        param_prom= 0.6
        peaks, properties = find_peaks(trace, threshold = param_thresh, prominence=param_prom, height=.01)
        maxpeak=0
        for k in range(len(peaks)):
            if peaks[k] > len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(0.3-timerange_cue[0]) and peaks[k] < len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(1-timerange_cue[0]):
                if properties['peak_heights'][k] > maxpeak:
                    maxpeak = properties['peak_heights'][k]
                    time = peaks[k]
        peakheight_bouts[mouse, date, trial, licklength]=maxpeak
        
        
        if licklength > 10 and licklength < 30 :
            smallbouts.append(np.array(trace))            
        elif licklength <10:
            tinybouts.append(np.array(trace))
        elif licklength > 29 and licklength < 50:
            mediumbouts.append(np.array(trace))
        elif licklength > 49 and licklength <70:
            longbouts.append(np.array(trace))
        elif licklength > 70:
            longlongbouts.append(np.array(trace))
            
mean_tinybouts = np.mean(tinybouts, axis=0)
sem_tinybouts = sem(tinybouts)
mean_smallbouts = np.mean(smallbouts, axis=0)
sem_smallbouts = sem(np.array(smallbouts))
mean_mediumbouts = np.mean(mediumbouts, axis=0)
sem_mediumbouts= sem(mediumbouts)
mean_longbouts = np.mean(longbouts, axis=0)
sem_longbouts = sem(longbouts)
mean_longlongbouts = np.mean(longlongbouts, axis=0)
sem_longlongbouts = sem(longlongbouts)

plt.figure(figsize=(10,5))
# plt.plot(np.arange(0,len(mean_tinybouts)), mean_tinybouts, label = '<10 Licks', color = colors[0])
# plt.fill_between(range(len(mean_tinybouts)), mean_tinybouts -sem_tinybouts , mean_tinybouts + sem_tinybouts, color = colors[0] , alpha=0.1)
plt.plot(np.arange(0,len(mean_smallbouts)), mean_smallbouts, label = '10-30 Licks', color = sns.color_palette("husl", 8)[0])
plt.fill_between(range(len(mean_smallbouts)), mean_smallbouts -sem_smallbouts , mean_smallbouts + sem_smallbouts, color = colors[0] , alpha=0.1)
plt.plot(np.arange(0,len(mean_mediumbouts)),mean_mediumbouts, label = '30-50 Licks' , color = sns.color_palette("husl", 8)[1])
plt.fill_between(range(len(mean_mediumbouts)), mean_mediumbouts -sem_mediumbouts , mean_mediumbouts + sem_mediumbouts, color = colors[1], alpha=0.1)
plt.plot(np.arange(0,len(mean_longbouts)),mean_longbouts, label = '50-70 Licks', color = colors[3])
plt.fill_between(range(len(mean_longbouts)), mean_longbouts -sem_longbouts , mean_longbouts + sem_longbouts,color = colors[3], alpha=0.1)
plt.plot(np.arange(0,len(mean_longlongbouts)),mean_longlongbouts, label = '70+ Licks', color = colors[5])
plt.fill_between(range(len(mean_longlongbouts)), mean_longlongbouts -sem_longlongbouts , mean_longlongbouts + sem_longlongbouts, color = colors[5], alpha=0.1)

plt.xticks(np.arange(0,len(longbouts[0])+1,len(longbouts[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.show()
plt.xlabel('Lick Bout Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/bylickbouts.pdf', transparent=True)




############################################################
offset_bout_df=pd.DataFrame()
lengths=[]
offset=[]
for key, value in offset_bouts.items():
    mouse, length = key
    x, y = value
    lengths.append(length)
    offset.append(x)
offset_bout_df['Lengths']= lengths
offset_bout_df['Offset']= offset


peakheght_by_bout_df=pd.DataFrame()
cell = 0
for key, value in peakheight_bouts.items():
    mouse, date, trial, licklength =key
    peak = value
    peakheght_by_bout_df.at[cell,'mouse'] = mouse
    peakheght_by_bout_df.at[cell,'date'] = date
    peakheght_by_bout_df.at[cell,'licklength'] = licklength
    peakheght_by_bout_df.at[cell,'peak'] = peak
    cell = cell + 1

##############################################################
######################## CUE ANALYSIS ########################
##############################################################

#by session individual
peakheight_range = [0,1.5]
peakheight={}
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgcuetrace_dict:
        plt.plot(avgcuetrace_dict[str(mouse),session],color=sns.color_palette("husl", 8)[int(session)],label=session)
        peakmax = max(avgcuetrace_dict[str(mouse),session][round(-timerange_cue[0]*fs/N):round((peakheight_range[1]-timerange_cue[0])*fs/N)])
        peakheight[mouse,session]=peakmax
plt.xticks(np.arange(0,len(avgcuetrace_dict[str(mouse),session][0])+1,len(avgcuetrace_dict[str(mouse),session][0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(avgcuetrace_dict[str(mouse),session][0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Cue Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/cue.pdf', transparent=True)

# PEAK HEIGHT
peakheight_session = {}
for key, value in peakheight.items():
    mouse, session = key
    if session not in peakheight_session:
        peakheight_session[session] = []
    peakheight_session[session].append((mouse, value))
plt.figure(figsize=(10, 6))
for session, data in peakheight_session.items():
    mice, peaks = zip(*data)
    plt.scatter([session] * len(mice), peaks, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}', alpha=0.7)
pkht_df=pd.DataFrame()
for key, value in peakheight_session.items():
    for mouse, peaks in value:
        pkht_df.loc[key,mouse]=peaks
plt.legend()
plt.xlabel('Session')
plt.ylabel('Peak Height')
plt.title('Peak Height')
plt.tight_layout()
plt.show()


#by session average
session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session = key
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(value)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    # plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
    # plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.1)
    if session in [1,6]:
        plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('Z-score')
plt.title('Average Cue-aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/cuebysessions.pdf', transparent=True)




#####################################################################
######################## LEVERPRESS ANALYSIS ########################
#####################################################################

#by session individual
peakheight_lp_range = [0,1.5]
peakheight_leverpress={}
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avglevertrace_dict:
        
        plt.plot(avglevertrace_dict[str(mouse),session],color=sns.color_palette("husl", 8)[int(session)],label=i)
        peakmax = max(avglevertrace_dict[str(mouse),session][round(-timerange_lever[0]*fs/N):round((peakheight_lp_range[1]-timerange_lever[0])*fs/N)])
        peakheight_leverpress[mouse,session]=peakmax
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Lever Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/lever.pdf', transparent=True)

peakheight_lp_session = {}
for key, value in peakheight_leverpress.items():
    mouse, session = key
    if session not in peakheight_lp_session:
        peakheight_lp_session[session] = []
    peakheight_lp_session[session].append((mouse, value))
plt.figure(figsize=(10, 6))
for session, data in peakheight_lp_session.items():
    mice, peaks = zip(*data)
    plt.scatter([session] * len(mice), peaks, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}', alpha=0.7)
pkht_lp_df=pd.DataFrame()
for key, value in peakheight_lp_session.items():
    for mouse, peaks in value:
        pkht_lp_df.loc[key,mouse]=peaks
plt.legend()
plt.xlabel('Session')
plt.ylabel('Peak Height')
plt.title('Peak Height')
plt.tight_layout()
plt.show()


#by session average
session_leverdata = {}  
for key, value in avglevertrace_dict.items():
    mouse, session = key
    if session not in session_leverdata:
        session_leverdata[session] = []
    session_leverdata[session].append(value)
mean_levertraces = {}
sem_levertraces = {}
for session, traces in session_leverdata.items():
    mean_levertraces[session] = np.mean(traces, axis=0)
    sem_levertraces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_trace in mean_levertraces.items():
    sem_levertrace = sem_levertraces[session]
    if session in [1,2,3,4,5,6,7]:
        plt.plot(mean_trace, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_levertrace, mean_trace + sem_levertrace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.3)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_levertraces[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.ylabel('Z-score')
plt.title('Average Lever-aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/leverbysessions.pdf', transparent=True)



###########################################################################
######################## AREA UNDER CURVE ANALYSIS ########################
###########################################################################
#area under the curve
after_zero = 2
start_time = round(len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]))
end_time = round(len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(after_zero-timerange_cue[0]))
area_under_curve = {}

for key, value in avgcuetrace_dict.items():
    mouse, session = key
    positivevalues = np.maximum(value[start_time:end_time], 0)
    area = np.sum(value[start_time:end_time])
    if session not in area_under_curve:
        area_under_curve[session] = []
    area_under_curve[session].append((mouse, area))
plt.figure(figsize=(10, 6))
for session, data in area_under_curve.items():
    mice, areas = zip(*data)
    plt.scatter([session] * len(mice), areas, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}', alpha=0.7)
auc_df=pd.DataFrame()
for key, value in area_under_curve.items():
    for mouse, auc in value:
        auc_df.loc[key,mouse]=auc
plt.legend()
plt.xlabel('Session')
plt.ylabel('Area under the curve')
plt.title(f'Area under the curve from 0 to {after_zero} sec')
plt.xticks(list(area_under_curve.keys()))
plt.tight_layout()
plt.show()


#####################################################################
######################## LOOK AT FULL TRACES ########################
#####################################################################

plt.figure(figsize=(12, 6))
totaltrace = np.array(df.iloc[:,2])
totaltracemean = np.mean(totaltrace)
totaltracestddev = np.std(totaltrace)
totaltracezscore = (totaltrace-totaltracemean)/totaltracestddev
multiplied_track_licks = [item * fs for item in track_licks]
multiplied_track_lever = [item * fs for item in track_lever]
multiplied_track_cue = [item * fs for item in track_cue]
multiplied_track_leverend = [item + 20*fs for item in multiplied_track_lever]
plt.plot(totaltracezscore)
plt.eventplot(multiplied_track_licks, colors='red', lineoffsets=0.5, label='lick')
plt.eventplot(multiplied_track_lever, colors='green', lineoffsets=0.5, label='leverpress/ sipper out')
plt.eventplot(multiplied_track_leverend, colors='blue', lineoffsets=0.5, label='end session')
plt.eventplot(multiplied_track_cue, colors='pink', lineoffsets=0.5, label='cue')
plt.legend()



#####################################################################
######################## FIRST LICK ANALYSIS ########################
#####################################################################
#by session individual
peakheight_flick_range = [0,1.5]
peakheight_flick={}
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgflicktrace_dict:
        if session == i:
            plt.plot(avgflicktrace_dict[str(mouse),session],color=sns.color_palette("husl", 8)[i],label=i)
            peakmax = max(avgflicktrace_dict[str(mouse),session][round(-timerange_lick[0]*fs/N):round((peakheight_flick_range[1]-timerange_lick[0])*fs/N)])
            peakheight_flick[mouse,session]=peakmax

plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('First Lick Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/lick.pdf', transparent=True)

peakheight_flick_session = {}
for key, value in peakheight_flick.items():
    mouse, session = key
    if session not in peakheight_flick_range:
        peakheight_flick_session[session] = []
    peakheight_flick_session[session].append((mouse, value))
    print(session, mouse, value)
    
plt.figure(figsize=(10, 6))
for session, data in peakheight_flick_session.items():
    mice, peaks = zip(*data)
    plt.scatter([session] * len(mice), peaks, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}', alpha=0.7)
pkht_flick_df=pd.DataFrame()
for key, value in peakheight_flick_session.items():
    for mouse, peaks in value:
        pkht_flick_df.loc[key,mouse]=peaks
plt.legend()
plt.xlabel('Session')
plt.ylabel('Peak Height')
plt.title('Peak Height')
plt.tight_layout()
plt.show()


#by session average
session_lickdata = {}  
for key, value in avgflicktrace_dict.items():
    mouse, session = key
    if session not in session_lickdata:
        session_lickdata[session] = []
    session_lickdata[session].append(value)
mean_licktraces = {}
sem_licktraces = {}
for session, traces in session_lickdata.items():
    mean_licktraces[session] = np.mean(traces, axis=0)
    sem_licktraces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_licktraces in mean_licktraces.items():
    sem_licktrace = sem_licktraces[session]
    if session in [2,3,4,5,6,7]:
        plt.plot(mean_licktraces, color=sns.color_palette("husl", 8)[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_licktraces)), mean_licktraces - sem_licktrace, mean_licktraces + sem_licktrace, color=sns.color_palette("husl", 8)[int(session)], alpha=0.3)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_licktraces)+1,len(mean_licktraces)/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_licktraces)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.ylabel('Z-score')
plt.title('Average Lick-aligned Trace with SEM by Session')
plt.legend()
plt.show()
plt.savefig('/Users/kristineyoon/Documents/firstlickbysessions.pdf', transparent=True)
