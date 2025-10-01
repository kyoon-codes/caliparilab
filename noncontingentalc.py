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



#NONCONTINGENT
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHNoncontingent'
mice = ['8730','8731','8732']


files = os.listdir(folder)
files.sort()
print(files)
lick_rasterdata = {}
alltrialtrace_dict = {}
avgcuetrace_dict = {}
avglevertrace_dict = {}
avgflicktrace_dict = {}
avgflicktrace1_dict = {}
avglickbouttrace_dict = {}
lickbouttracedict ={}
allcuetraces = {}
bout_traces = {}
bout_peak_heights = {}
bout_peak_indices = {}
timerange_cue = [-2, 5]
timerange_lever = [-2, 5]
timerange_lick = [-2, 10]
avgresptrace_dict = {}
active_time = [-2,40]
cue_time = 20
lick_time= 10
N = 100
cuebaseline = {}
lickspersession = {}
responsepersession = {}
timeoutpersession = {}


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
        df['Sig405'] = data.streams._405B.data
        df['Sig465'] = data.streams._465B.data[0:len(data.streams._405B.data)]
        
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
        track_licks = []
        track_to = []
        latency= []
        leverpermice = []
        for i in range(len(fp_df)):
            if fp_df.iloc[i,0] == 'Cue':
                track_cue.append(fp_df.iloc[i,1])
            if fp_df.iloc[i,0] == 'Licks':
                track_licks.append(fp_df.iloc[i,1])
            if fp_df.iloc[i,0] == 'Timeout Press':
                track_to.append(fp_df.iloc[i,1])
        
        totallicks=[]
        for i in range (len(track_licks)):
            if len(track_cue) == 10:
                totallicks.append(track_licks[i])
            else:
                if track_licks[i] < track_cue[10]:
                    totallicks.append(track_licks[i])
        lickspersession[mouse,Dates.index(date),i]= len(totallicks)
        

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
            
            avgcuetrace_dict[mouse,Dates.index(date),i]= track_cue[i], sampletrial, aligntobase
            cuebaseline[mouse,Dates.index(date),i] = aligntobase, np.std(df.iloc[cue_baseline:cue_zero,2])
        

        ################## FIRST LICK ALIGNMENT ################3
        track_flicks = []
        firstlicks = []
        for cuelight in track_cue:
            lickyes = np.array(track_licks) > cuelight
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
            avgflicktrace_dict[mouse,Dates.index(date),cue_trial]= track_flicks[i], sampletrial, aligntobase
        
        
        ################## FIRST LICK ALIGNMENT WITH CUE BASELINE ################3

        for i in range(len(track_flicks)):
            for trial_num in range(len(track_cue)):
                if track_flicks[i] - track_cue[trial_num] > 0 and track_flicks[i] - track_cue[trial_num] < 30:
                    cue_trial = trial_num
                    
            flick_zero = round(track_flicks[i] * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs
            
            cue_mean = cuebaseline[mouse,Dates.index(date),cue_trial][0]
            cue_std = cuebaseline[mouse,Dates.index(date),cue_trial][1]
            rawtrial = np.array(df.iloc[flick_baseline:flick_end,2])

            trial = []
            for each in rawtrial:
                trial.append((each-cue_mean)/cue_std)
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            
            avgflicktrace1_dict[mouse,Dates.index(date),cue_trial]= track_flicks[i], sampletrial, cue_mean
        
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
            cue_baseline = cue_zero + active_time[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            
        
            for m in range(len(track_flicks)):
                if track_flicks[m] - track_cue[i] > 0 and track_flicks[m] - track_cue[i] < 30:
                    flicktime = track_flicks[m]
        
            else:
                flicktime = np.nan
            aligntobase = np.mean(df.iloc[cue_baseline:cue_zero,2])
            rawtrial = np.array(df.iloc[cue_baseline:cue_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))

            alltrialtrace_dict[mouse,Dates.index(date),i]= track_cue[i], flicktime, sampletrial

########################################################################################################################
############################################################################################################################
### LOOKING AT THE TRACE BY ALL SESSION 
############################################################################################################################
############################################################################################################################

session_ranges = 3

#by session average
#colors10 = ['#8da0cb','#66c2a5','#a6d854','#ffd92f','#e78ac3','#e5c494','#bc80bd','#fc8d62']


colors10 = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen', 'mediumseagreen', 'mediumturquoise', 'deepskyblue', 'dodgerblue', 'slateblue', 'darkorchid','purple']

session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    if trial in range(10):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    #if session in [1,6]:
    if session in range(session_ranges):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.title('Average Cue Aligned Trace with SEM by Session')
plt.legend()

plt.savefig('/Users/kristineyoon/Documents/cuebysessions.pdf', transparent=True)
plt.show()


##############################################################
#by session average
start_time=-2
end_time=10
session_data = {}  
for key, value in avgflicktrace1_dict.items():
    mouse, session, trial = key
    newtrace = value[1]
    if trial in range (10):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(newtrace)
            
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sy.stats.sem(traces)

plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    #if session in [2,7]:
    if session in range (session_ranges):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(newtrace)+1,len(newtrace)/(end_time-start_time)), 
            np.arange(start_time, end_time+1,1, dtype=int),
            rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(newtrace)/(end_time-start_time)*(0-start_time),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average First Lick Aligned Trace with SEM by Session')
plt.legend()
plt.show()


##############################################################
#all sessions togehter
start_time=-2
end_time=10
session_data = {}  
for key, value in avgflicktrace1_dict.items():
    mouse, session, trial = key
    newtrace = value[1]
    if trial in range (10):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(newtrace)
            
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sy.stats.sem(traces)

plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
    plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(newtrace)+1,len(newtrace)/(end_time-start_time)), 
            np.arange(start_time, end_time+1,1, dtype=int),
            rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(newtrace)/(end_time-start_time)*(0-start_time),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average First Lick Aligned Trace with SEM by Session')
plt.legend()
plt.show()


session_data = {}  
for key, value in alltrialtrace_dict.items():
    mouse, session, trial = key
    time, blank1, trialtrace = value
    if trial in range(0,10):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    if session in range(session_ranges):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(active_time[1]-active_time[0])), 
           np.arange(active_time[0], active_time[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.title('Average Cue Aligned Trace with SEM by Session')
plt.legend()
plt.show()


session_data = {}  
for key, value in avglickbouttrace_dict.items():
    mouse, session, trial = key
    time,bout, trialtrace = value
    if trial in range(0,10):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(trialtrace)
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    if session in range(session_ranges):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.title('Average Cue Aligned Trace with SEM by Session')
plt.legend()
plt.show()      
