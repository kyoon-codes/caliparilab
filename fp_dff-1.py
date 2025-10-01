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




# D2 MEDIUM SPINY NEURONS (ALCOHOL)
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHLearning/'
mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321','8729','8730','8731','8732']
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_1WeekWD/'
mice = ['7098','7099','7108','7296', '7311', '7319', '7321','8729','8730','8731','8732']
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHExtinction/'
mice = ['7098','7099','7108','7296', '7311', '7319', '7321','8729','8730','8731','8732']

# D2 MEDIUM SPINY NEURONS (SUCROSE)
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucLearning/'
mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucExtinction/'
mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
#,'8733','8742','8743','8747','8748','8750']

# D2 MEDIUM SPINY NEURONS (SUCROSE TO ETHANOL TO SUCROSE)
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_EtOHLearning/'
mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_AlcExtinction/'
mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_SucExtinction2/'
mice = ['7678', '7680', '7899','8733','8742','8743','8748','8747','8750']



# D1 MEDIUM SPINY NEURONS
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_EtOHLearning/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_1WeekWD/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_EtOHExtinction/'
# mice = ['676', '679', '849', '873', '874', '917']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_SucLearning/'
# mice = ['676', '679', '849', '873', '874', '917']
# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D1_SucExtinction/'
# mice = ['676', '679', '849', '873', '874', '917']


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
        
        totallicks=[]
        for i in range (len(track_licks)):
            if len(track_cue) == 10:
                totallicks.append(track_licks[i])
            else:
                if track_licks[i] < track_cue[10]:
                    totallicks.append(track_licks[i])
        lickspersession[mouse,Dates.index(date),i]= len(totallicks)
        
        totalresponse=[]
        for i in range (len(track_lever)):
            if len(track_cue) == 10:
                totalresponse.append(track_lever[i])
            else:
                if track_lever[i] < track_cue[10]:
                    totalresponse.append(track_lever[i])
        responsepersession[mouse,Dates.index(date),i]= len(totalresponse)
        
        
        totaltimeout = []
        for i in range (len(track_to)):
            if len(track_cue) == 10:
                totaltimeout.append(track_to[i])
            else:
                if track_to[i] < track_cue[10]:
                    totaltimeout.append(track_to[i])
        timeoutpersession[mouse,Dates.index(date),i]= len(totaltimeout)
        

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
            
            avglevertrace_dict[mouse,Dates.index(date),cue_trial]= track_lever[i], sampletrial, aligntobase
            cuebaseline[mouse,Dates.index(date),cue_trial]=aligntobase, np.std(df.iloc[cue_baseline:cue_zero,2])

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
            
            for l in range(len(track_lever)):
                if  track_lever[l] - track_cue[i] > 0 and track_lever[l] - track_cue[i] < 20:
                    levertime = track_lever[l]
            else:
                    levertime = np.nan
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

            alltrialtrace_dict[mouse,Dates.index(date),i]= track_cue[i], levertime, flicktime, sampletrial

        

############################################################################################################################
############################################################################################################################
### FIRST CHECK POINT
############################################################################################################################
############################################################################################################################
#by session individual
# ['7098','7099','7107','7108','7296','7310','7311','7319','7321','7329']
testlist = ['7098'] #1
rangelist = [3,7]

fig, axs = plt.subplots(6, sharex=True, figsize=(10,12))
for i in range(6):
    for mouse,session,trial in avgcuetrace_dict:
        if mouse in testlist:
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

fig, axs = plt.subplots(len(rangelist), sharex=True, figsize=(10,12))
for i in rangelist:
    for mouse,session,trial in avgflicktrace_dict:
        if mouse in testlist:
            if session == i:
                axs[rangelist.index(i)].plot(avgflicktrace_dict[str(mouse),session,trial][1],color=colors10[int(session)],label=session)
                axs[rangelist.index(i)].axvline(x=len(avgflicktrace_dict[str(mouse),session,trial][1])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.xticks(np.arange(0,len(avgflicktrace_dict[str(mouse),session,trial][1])+1,len(avgflicktrace_dict[str(mouse),session,trial][1])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.legend()
plt.xlabel('First Lick Onset (s)')

##############################################################
#by session individual
fig, axs = plt.subplots(len(rangelist), sharex=True, figsize=(10,12))
for i in rangelist:
    for mouse,session,trial in totalactivetrace_dict:
        if mouse in testlist and trial == 8:
            if session == i:
                trace = totalactivetrace_dict[str(mouse),session,trial][3]
                cuetime = totalactivetrace_dict[str(mouse),session,trial][0]
                levertime = totalactivetrace_dict[str(mouse),session,trial][1]
                licktime = totalactivetrace_dict[str(mouse),session,trial][2]
                axs[rangelist.index(i)].plot(trace, color=colors10[int(session)],label=session)
                axs[rangelist.index(i)].axvline(x=len(trace)/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black')
                axs[rangelist.index(i)].axvline(x=len(trace)/(active_time[1]-active_time[0])*(levertime-cuetime+2),linewidth=1, color='blue')
                axs[rangelist.index(i)].axvline(x=len(trace)/(active_time[1]-active_time[0])*(licktime-cuetime+2),linewidth=1, color='green')
plt.xticks(np.arange(0,len(trace)+1,len(trace)/(active_time[1]-active_time[0])), 
           np.arange(active_time[0], active_time[1]+1,1, dtype=int),
           rotation=0)
plt.legend()
plt.xlabel('Full Active Trace (s)')

############################################################################################################################
############################################################################################################################
### LOOKING AT THE TRACE BY ALL SESSION 
############################################################################################################################
############################################################################################################################

session_ranges = len(files)

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
    #if session in range(session_ranges):
    if session in range(1,7):
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
session_data = {}  
for key, value in avglevertrace_dict.items():
    mouse, session, time = key
    trialtrace = value
    if session not in session_data:
        session_data[session] = []
    session_data[session].append(trialtrace[1])
mean_traces = {}
sem_traces = {}
for session, traces in session_data.items():
    mean_traces[session] = np.mean(traces, axis=0)
    sem_traces[session] = sem(traces)
plt.figure(figsize=(10, 8))
for session, mean_trace in mean_traces.items():
    sem_trace = sem_traces[session]
    #if session in range(session_ranges):
    if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
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
plt.savefig('/Users/kristineyoon/Documents/leverbysessions_sucroseextinction.pdf', transparent=True)


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
    if session in range(1,7):
    #if session in range (session_ranges):
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

plt.savefig('/Users/kristineyoon/Documents/flick_sucextinct.pdf', transparent=True)




session_data = {}  
for key, value in alltrialtrace_dict.items():
    mouse, session, trial = key
    time, blank1,blank2, trialtrace = value
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
##############################################################

# ##############################################################
# start_time=-2
# end_time=10
# #by session average
# session_data = {}  
# for key, value in avgflicktrace_dict.items():
#     mouse, session, trial = key
#     newtrace = value[1] #[:122] ## had to add 122 to get only the range [-2,10]
#     if session not in session_data:
#         session_data[session] = []
#     session_data[session].append(newtrace)
        
# mean_traces = {}
# sem_traces = {}
# for session, traces in session_data.items():
#     mean_traces[session] = np.mean(traces, axis=0)
#     sem_traces[session] = sem(traces)

# plt.figure(figsize=(10, 6))
# for session, mean_trace in mean_traces.items():
#     sem_trace = sem_traces[session]
#     if session in range(session_ranges):
#         plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
#         plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
# plt.xlabel('Time (samples)')
# plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(end_time-start_time)), 
#             np.arange(start_time, end_time+1,1, dtype=int),
#             rotation=0)
# plt.axhline(y=0, linestyle=':', color='black')
# plt.axvline(x=len(newtrace)/(end_time-start_time)*(0-start_time),linewidth=1, color='black')
# plt.ylabel('DF/F')
# plt.title('Average First Lick Aligned Trace with SEM by Session')
# plt.legend()
# plt.show()


#plt.savefig('/Users/kristineyoon/Documents/flickbysessions.pdf', transparent=True)
############################################################################################################################
############################################################################################################################
### PEAK HEIGHTS WITH MAX HEIGHT
############################################################################################################################
############################################################################################################################
# plt.figure(figsize=(10, 6))

## answer this
totalsessions = 2

allcuepeakheights ={}
peakheight_cue_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
for mouse,session,trial in avgcuetrace_dict:
    if trial in range (10):
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[mouse,session,trial][1]))  # 500 points from -2 to 10 seconds
        trace = avgcuetrace_dict[mouse,session,trial][1]
        
        # Define the interval of interest
        start_time = 0
        end_time = 1
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        peakheight = max(trace_segment)
        peakheighttime = time_segment[trace_segment.index(peakheight)]
        num = len(peakheight_cue_df)
        peakheight_cue_df.at[num,'Mouse']=mouse
        peakheight_cue_df.at[num,'Session']=session
        peakheight_cue_df.at[num,'Trial']=trial
        peakheight_cue_df.at[num,'X-Axis']= peakheighttime
        peakheight_cue_df.at[num,'PeakHeight']= peakheight
        allcuepeakheights[mouse,session,trial] = peakheight, peakheighttime

 
cuepeakheight = []
plt.figure()
cue_df1=pd.DataFrame()
for i in range(8):
    session_peakheight = []
    for mouse,session,trial in allcuepeakheights:
        if session == i and trial in range(10):
            session_peakheight.append(allcuepeakheights[mouse,session,trial][0])
            plt.scatter(x=session,y= allcuepeakheights[mouse,session,trial][0], color=colors10[session], alpha=0.02)
    cuepeakheight.append(session_peakheight)
    print(i, np.mean(session_peakheight))
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
    cue_df1.at[:,i]=session_peakheight
plt.xlabel('Session')
plt.ylabel('Peak Height At Cue')
plt.show()



allleverpeakheights ={}
peakheight_lever_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
for mouse,session,trial in avglevertrace_dict:
    if trial in range (10):
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(avglevertrace_dict[mouse,session,trial][1]))  # 500 points from -2 to 10 seconds
        trace = avglevertrace_dict[mouse,session,trial][1]
        
        # Define the interval of interest
        start_time = -0.5
        end_time = 0.5
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        peakheight = max(trace_segment)
        peakheighttime = time_segment[trace_segment.index(peakheight)]
        num = len(peakheight_lever_df)
        peakheight_lever_df.at[num,'Mouse']=mouse
        peakheight_lever_df.at[num,'Session']=session
        peakheight_lever_df.at[num,'Trial']=trial
        peakheight_lever_df.at[num,'X-Axis']= peakheighttime
        peakheight_lever_df.at[num,'PeakHeight']= peakheight
        allleverpeakheights[mouse,session,trial] = peakheight, peakheighttime

# leverpeakheight = []
# plt.figure()
# for i in range(totalsessions):
#     session_peakheight = []
#     for mouse,session,trial in allleverpeakheights:
#         if session == i and trial in range(10):
#             session_peakheight.append(allleverpeakheights[mouse,session,trial][0])
#             plt.scatter(x=session,y= allleverpeakheights[mouse,session,trial][0], color=colors10[session], alpha=0.2)
#     leverpeakheight.append(session_peakheight)
#     plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
#     plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
# plt.xlabel('Session')
# plt.ylabel('Peak Height At Lever')
# plt.show()



allflickpeakheights ={}
peakheight_flick_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
for mouse,session,trial in avgflicktrace1_dict:
    if trial in range (10):
        time = np.linspace(timerange_lick[0], timerange_lick[1], len(avgflicktrace1_dict[mouse,session,trial][1]))  # 500 points from -2 to 10 seconds
        trace = avgflicktrace1_dict[mouse,session,trial][1]
        
        # Define the interval of interest
        start_time = 0
        end_time = 1
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        peakheight = max(trace_segment)
        peakheighttime = time_segment[trace_segment.index(peakheight)]
        num = len(peakheight_flick_df)
        peakheight_flick_df.at[num,'Mouse']=mouse
        peakheight_flick_df.at[num,'Session']=session
        peakheight_flick_df.at[num,'Trial']=trial
        peakheight_flick_df.at[num,'X-Axis']= peakheighttime
        peakheight_flick_df.at[num,'PeakHeight']= peakheight
        allflickpeakheights[mouse,session,trial] = peakheight, peakheighttime

flickpeakheight = []
plt.figure()
for i in range(totalsessions):
    session_peakheight = []
    for mouse,session,trial in allflickpeakheights:
        if session == i and trial in range(10):
            session_peakheight.append(allflickpeakheights[mouse,session,trial][0])
            plt.scatter(x=session,y= allflickpeakheights[mouse,session,trial][0], color=colors10[session], alpha=0.2)
    flickpeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Peak Height At First Lick')
plt.show()



############################################################################################################################
############################################################################################################################
### PEAK HEIGHTS
############################################################################################################################
############################################################################################################################

allcuepeakheights={}
from scipy.signal import find_peaks         
peakheight_cue_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
# plt.figure(figsize=(10, 6))
for mouse,session,trial in avgcuetrace_dict:
    if trial in range (10):
        avgtrace = avgcuetrace_dict[str(mouse),session,trial][1]
        param_prom= 1
        peaks, properties = find_peaks(avgtrace, prominence=param_prom, height=avgcuetrace_dict[str(mouse),session,trial][2])
        trialpeakheights=[]
        trialprominence=[]
        time=[]
        for k in range(len(peaks)):
            if peaks[k] > len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]) and peaks[k] < len(avgtrace)/(timerange_cue[1]-timerange_cue[0])*(1-timerange_cue[0]):
                trialpeakheights.append(properties['peak_heights'][k])
                time.append(peaks[k])
                trialprominence.append(properties['prominences'][k])
        num = len(peakheight_cue_df)
        peakheight_cue_df.at[num,'Mouse']=mouse
        peakheight_cue_df.at[num,'Session']=session
        peakheight_cue_df.at[num,'Trial']=trial
        if len(trialpeakheights) > 0:
            bestpeak=max(trialprominence)
            bestpeak_time = time[trialprominence.index(bestpeak)]
            bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
            peakheight_cue_df.at[num,'X-Axis']= time
            peakheight_cue_df.at[num,'PeakHeight']= bestpeak_height
            allcuepeakheights[mouse,session,trial] = bestpeak_height, bestpeak_time

        else:
            peakheight_cue_df.at[num,'X-Axis']= 0
            peakheight_cue_df.at[num,'PeakHeight']= avgcuetrace_dict[str(mouse),session,trial][2]
            allcuepeakheights[mouse,session,trial] = avgcuetrace_dict[str(mouse),session,trial][2], 0


cuepeakheight = []
cue_df1=pd.DataFrame()
for i in range(12):
    session_peakheight = []
    for mouse,session,trial in allcuepeakheights:
        if session == i and trial in range(10):
            session_peakheight.append(allcuepeakheights[mouse,session,trial][0])
            plt.scatter(x=session,y= allcuepeakheights[mouse,session,trial][0], color=colors10[session], alpha=0.2)
    cuepeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
    cue_df1.at[:,i]=session_peakheight
plt.xlabel('Session')
plt.ylabel('Peak Height At Cue')
plt.show()

fig, axs = plt.subplots(8, sharey=True, sharex=True, figsize=(10,12))
for i in range(8):
    for mouse,session,trial in avgcuetrace_dict:
        if session == i and trial in range (10):
            if allcuepeakheights[mouse,session,trial][1] != 0:
                axs[i].scatter(x= allcuepeakheights[mouse,session,trial][1], y=allcuepeakheights[mouse,session,trial][0], c=colors10[int(session)])
            axs[i].plot(avgcuetrace_dict[str(mouse),session,trial][1],color=colors10[int(session)],label=session, alpha=0.2)
            axs[i].axvline(x=len(avgcuetrace_dict[str(mouse),session,trial][1])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=0.5, color='black')
plt.xticks(np.arange(0,len(avgcuetrace_dict[str(mouse),session,trial][1])+1,len(avgcuetrace_dict[str(mouse),session,trial][1])/(timerange_cue[1]-timerange_cue[0])), 
       np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
       rotation=0)
plt.xlabel('Cue Onset (s)')
plt.ylim(-5,12)
fig.tight_layout()


trialcolor = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
plt.figure()
cuepeakheight = []
for i in range(10):
    session_peakheight = []
    for mouse,session,trial in allcuepeakheights:
        if trial == i and session in [1,3,7]:
            session_peakheight.append(allcuepeakheights[mouse,session,trial][0])
            plt.scatter(x=trial,y= allcuepeakheights[mouse,session,trial][0], color=trialcolor[session], alpha=0.3, label=session)
    cuepeakheight.append(session_peakheight)
plt.legend()
plt.xlabel('Trial')
plt.ylabel('Peak Height At Cue')
plt.show()


cue_df=pd.DataFrame()
num = 0
for i in range(8):
    for mouse,session,trial in allcuepeakheights:
        if session == i:
            cue_df.at[num,'Cue PkHt']=allcuepeakheights[mouse,session,trial][0]
            cue_df.at[num,'Session']=session
            cue_df.at[num,'mouse']=mouse
            num = num + 1


##############################################################
# LEVER PRESS

allleverpeakheights={}  
peakheight_lever_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
plt.figure(figsize=(10, 6))
for mouse,session,trial in avglevertrace_dict:
    if mouse in mice:
        if session in range(10):
            if trial in range (10):
                avgtrace = avglevertrace_dict[str(mouse),session,trial][1]
                param_prom= 1
                peaks, properties = find_peaks(avgtrace, prominence=param_prom, height=avglevertrace_dict[str(mouse),session,trial][2])
                plt.plot(avgtrace, label=f'Session {session}', alpha=0.5, color=colors10[trial])
                trialpeakheights=[]
                trialprominence=[]
                time=[]
                for k in range(len(peaks)):
                    if peaks[k] > len(avgtrace)/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]) and peaks[k] < len(avgtrace)/(timerange_lever[1]-timerange_lever[0])*(1-timerange_lever[0]):
                        trialpeakheights.append(properties['peak_heights'][k])
                        time.append(peaks[k])
                        trialprominence.append(properties['prominences'][k])
                        # if properties['peak_heights'][k] > maxpeak:
                        #     maxpeak = properties['peak_heights'][k]
                        #     time = peaks[k]
                        #     plt.scatter(x=peaks[k], y= properties['peak_heights'][k], alpha=0.8)
                
                num = len(peakheight_lever_df)
                peakheight_lever_df.at[num,'Mouse']=mouse
                peakheight_lever_df.at[num,'Session']=session
                peakheight_lever_df.at[num,'Trial']=trial
                if len(trialpeakheights) > 0:
                    bestpeak=max(trialprominence)
                    bestpeak_time = time[trialprominence.index(bestpeak)]
                    bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
                    peakheight_lever_df.at[num,'X-Axis']= time
                    peakheight_lever_df.at[num,'PeakHeight']= bestpeak
                    allleverpeakheights[mouse,session,trial] = bestpeak
                    plt.scatter(x=bestpeak_time, 
                                y=bestpeak, 
                                alpha=0.8,
                                color=colors10[trial])
                else:
                    peakheight_lever_df.at[num,'X-Axis']= np.nan
                    peakheight_lever_df.at[num,'PeakHeight']= avglevertrace_dict[str(mouse),session,trial][2]
                    allleverpeakheights[mouse,session,trial] = avglevertrace_dict[str(mouse),session,trial][2]
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
for i in range(10):
    session_peakheight = []
    for mouse,session,trial in allleverpeakheights:
        if session == i:
            session_peakheight.append(allleverpeakheights[mouse,session,trial])
            plt.scatter(x=session,y= allleverpeakheights[mouse,session,trial], color=colors10[session], alpha=0.05)
    leverpeakheight.append(session_peakheight)
    plt.scatter(x=i, y=np.mean(session_peakheight), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(session_peakheight), yerr=sem(session_peakheight), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylim(0,8)
plt.ylabel('Peak Height At Lvever')
plt.show()


lever_df=pd.DataFrame()
for i in range(10):
    num = 0
    for mouse,session,trial in allleverpeakheights:
        if session == i:
            lever_df.at[num,i]=allleverpeakheights[mouse,session,trial]
            num = num + 1

lever_df1=pd.DataFrame()
for i in range(10):
    num = 0
    for mouse,session,trial in allleverpeakheights:
        if session == i:
            lever_df1.at[num,i]=allleverpeakheights[mouse,session,trial]
            num = num + 1

##############################################################
# FIRST LICK
start_time=-2
end_time=10

alllickpeackheight={}
plt.figure(figsize=(10, 6))
peakheight_lickbout_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
for mouse,session,trial in avgflicktrace_dict:
    if mouse in mice:
        if session in range(12):
            if trial in range (10):
                avgtrace = avgflicktrace_dict[str(mouse),session,trial][1][:122] #snipping out the last 5 minutes after lick
                param_prom= 1
                peaks, properties = find_peaks(avgtrace, prominence=param_prom, height=avgflicktrace_dict[str(mouse),session,trial][2])
                plt.plot(avgtrace, label=f'Session {session}', alpha=0.5, color=colors10[trial])
                trialpeakheights=[]
                trialprominence=[]
                time=[]
                for k in range(len(peaks)):
                    if peaks[k] > len(avgtrace)/(end_time-start_time)*(0-start_time) and peaks[k] < len(avgtrace)/(end_time-start_time)*(1-start_time):
                        trialpeakheights.append(properties['peak_heights'][k])
                        time.append(peaks[k])
                        trialprominence.append(properties['prominences'][k])
                        # if properties['peak_heights'][k] > maxpeak:
                        #     maxpeak = properties['peak_heights'][k]
                        #     time = peaks[k]
                        #     plt.scatter(x=peaks[k], y= properties['peak_heights'][k], alpha=0.8)
                
                num = len(peakheight_lickbout_df)
                peakheight_lickbout_df.at[num,'Mouse']=mouse
                peakheight_lickbout_df.at[num,'Session']=session
                peakheight_lickbout_df.at[num,'Trial']=trial
                if len(trialpeakheights) > 0:
                    bestpeak=max(trialprominence)
                    bestpeak_time = time[trialprominence.index(bestpeak)]
                    bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
                    peakheight_lickbout_df.at[num,'X-Axis']= time
                    peakheight_lickbout_df.at[num,'PeakHeight']= bestpeak
                    alllickpeackheight[mouse,session,trial] = bestpeak
                    plt.scatter(x=bestpeak_time, 
                                y=bestpeak, 
                                alpha=0.8,
                                color=colors10[trial])
                else:
                    peakheight_lickbout_df.at[num,'X-Axis']= np.nan
                    peakheight_lickbout_df.at[num,'PeakHeight']= np.nan #avgflicktrace_dict[str(mouse),session,trial][2]
                    alllickpeackheight[mouse,session,trial] = np.nan #avgflicktrace_dict[str(mouse),session,trial][2]
plt.axhline(y=0, linestyle=':', color='black')
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(avgtrace)+1,len(avgtrace)/(end_time-start_time)), 
           np.arange(start_time, end_time+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(avgtrace)/(end_time-start_time)*(0-start_time),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Cue-aligned Trace with SEM by Session')
plt.show()


plt.figure()
flickpeakheight = []
for i in range(12):
    session_peakheight = []
    for mouse,session,trial in alllickpeackheight:
        if session == i:
            session_peakheight.append(alllickpeackheight[mouse,session,trial])
            plt.scatter(x=session,y= alllickpeackheight[mouse,session,trial], color=colors10[session], alpha=0.5)
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


flick_df1=pd.DataFrame()
for i in range(10):
    num = 0
    for mouse,session,trial in alllickpeackheight:
        if session == i:
            flick_df1.at[num,i]=alllickpeackheight[mouse,session,trial]
            num = num + 1

############################################################################################################################
############################################################################################################################
# CALCULATING TIME TO BASELINE FOR THE FIRST LICK D2 MSN SUPPRESION 
############################################################################################################################
############################################################################################################################
# Initialize a dictionary to store the times to reach baseline grouped by session
#### see if time to reach baseline is different across sessions for first lick ####
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal


session_times = {}
time_to_baseline_dict = {}

# Iterate through the dictionary

for (mouse, session, trial), data in avgflicktrace_dict.items():
    start_time = 0
    end_time = 15
    time_index=[]
    time = np.linspace(timerange_lick[0], timerange_lick[1], 316)
    # Find the indices for the interval 0 to 1 second
    start_index = np.searchsorted(time, start_time)
    end_index = np.searchsorted(time, end_time)

    # Select the relevant portion of the trace
    time_segment = time[start_index:end_index]
    ogtrace = data[1]
    trace = ogtrace[start_index:end_index]
    baseline = data[2]  # Extract the baseline value
    for i,value in enumerate(trace):
        if value > baseline:
            time_index.append(i)
    #print(time_index)
    
    if len(time_index) > 0:
        initial = 3
        for i in range(len(time_index)):
            if time_index[i]-initial > 2:
                baselinetime = time_segment[time_index[i]]
                break
                
            else:
                initial = time_index[i]
        if session == 7:
            print(time_index[i])
            print(baselinetime)
        
        time_to_baseline_dict[mouse,session,trial]=baselinetime
        # Store in the session-based dictionary
        if session not in session_times:
            session_times[session] = []
    
        session_times[session].append(baselinetime)

# Statistical Test: Kruskal-Wallis H-test (non-parametric)
sessions = list(session_times.keys())
session_values = [session_times[session] for session in sessions]
stat, p_value = kruskal(*session_values)

print(f"Kruskal-Wallis H-test: H-statistic = {stat:.2f}, p-value = {p_value:.4f}")

# Visualization: Boxplot or Violin Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=session_values, palette='pastel')
plt.xticks(ticks=range(len(sessions)), labels=sessions, rotation=45)
plt.xlabel('Session')
plt.ylabel('Time to Reach Baseline (s)')
plt.title('Comparison of Time to Reach Baseline Across Sessions')
plt.tight_layout()
plt.show()

session_value_df=pd.DataFrame.from_dict(session_times, orient='index')

time_to_baseline_df = pd.DataFrame()
for i in range(10):
    num = 0
    for (mouse,session,trial),times in time_to_baseline_dict.items():
        if session == i:
            time_to_baseline_df.at[num,i] = times
            num = num + 1
############################################################################################################################
############################################################################################################################
### HEAT MAPS
############################################################################################################################
############################################################################################################################
# #### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by latency to lick ####

# colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

# sess_of_interest = [1,7]

# alltraces=[]
# for i in sess_of_interest:
#     bysession = {}
#     for subj,session,trial in totalactivetrace_dict:
#         if session == i and trial in range (10):
#             if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
#                 bysession[20 + totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
#             else:
#                 bysession[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
#     sorted_bysession=[]
#     for k in sorted(bysession.keys()):
#         #print(i)
#         sorted_bysession.append(bysession[k])
#     alltraces.append(sorted_bysession)

# fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
# for i in sess_of_interest:
#     bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
#     sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
#     axs[sess_of_interest.index(i)].axvline(x=(bysessiondf.shape[1])/(active_time[1]-active_time[0])*(0-active_time[0]),linewidth=1, color='black', label='Cue Onset')
# plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2), 
#             np.arange(active_time[0], active_time[1],2, dtype=int),
#             rotation=0)

# plt.ylabel('Trials')
# plt.xlabel('Time (sec)')
# plt.show()


####################################################################################
####################################################################################


#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by trial ####
colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

sess_of_interest = [1,3,7]
end_time = 5.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    bytrial =  {}
    for subj,session,trial in avgcuetrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[subj,i,trial][1]))
            trace = avgcuetrace_dict[subj,i,trial][1]
            start_index = np.searchsorted(timespace, timerange_cue[0])
            end_index = np.searchsorted(timespace, timerange_cue[1])  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            bysession[trial, session, subj]= trace_segment
            
    #sort by mean trace
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=False):
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in range(len(sess_of_interest)):
    bysessiondf= pd.DataFrame(alltraces[i])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[i])
    difference_array=np.absolute(time_segment-0)
    axs[i].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(timerange_cue[1]-timerange_cue[0])), np.arange(timerange_cue[0],timerange_cue[1],1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Cue (sec)')
plt.show()    

#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by mean trace ####
colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

sess_of_interest = [1,7]
end_time = 5.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    bytrial =  {}
    for subj,session,trial in avgcuetrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[subj,i,trial][1]))
            trace = avgcuetrace_dict[subj,i,trial][1]
            start_index = np.searchsorted(timespace, timerange_cue[0])
            end_index = np.searchsorted(timespace, timerange_cue[1])  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            bysession[np.average(trace_segment)]= trace_segment

    #sort by mean trace
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=True):
        #print(i)
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in range(len(sess_of_interest)):
    bysessiondf= pd.DataFrame(alltraces[i])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[i])
    difference_array=np.absolute(time_segment-0)
    axs[i].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(timerange_cue[1]-timerange_cue[0])), np.arange(timerange_cue[0],timerange_cue[1],1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Cue (sec)')
plt.show()    



#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by prior licks ####
colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

sess_of_interest = [1,7]
end_time = 5.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    bytrial =  {}
    for subj,session,trial in avgcuetrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[subj,i,trial][1]))
            trace = avgcuetrace_dict[subj,i,trial][1]
            start_index = np.searchsorted(timespace, timerange_cue[0])
            end_index = np.searchsorted(timespace, timerange_cue[1])  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            key = subj,session, trial-1
            if key in avglickbouttrace_dict:
                bysession[avglickbouttrace_dict[subj,session, trial-1][1],trial, session,subj]= trace_segment
            else:
                bysession[0,trial, session,subj]= trace_segment
            
    #sort by mean trace
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=False):
        print (k)
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in range(len(sess_of_interest)):
    bysessiondf= pd.DataFrame(alltraces[i])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[i])
    difference_array=np.absolute(time_segment-0)
    axs[i].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(timerange_cue[1]-timerange_cue[0])), np.arange(timerange_cue[0],timerange_cue[1],1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Cue (sec)')
plt.show()    

#### TRIALS ON HEATMAP ALIGNED TO FIRST LICK by session sorted by mean trace####
sess_of_interest = [1,6]
start_time = -2
end_time = 10.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    for subj,session,trial in avgflicktrace1_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(timerange_lick[0], timerange_lick[1], len(avgflicktrace1_dict[subj,i,trial][1]))
            trace = avgflicktrace1_dict[subj,i,trial][1]
            start_index = np.searchsorted(timespace, start_time)
            end_index = np.searchsorted(timespace, end_time)  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            bysession[np.average(trace_segment)]= trace_segment
           
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=True):
        #print(i)
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in sess_of_interest:
    bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
    difference_array=np.absolute(time_segment-0)
    axs[sess_of_interest.index(i)].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(end_time-start_time)), np.arange(start_time,end_time,1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Lick (sec)')
plt.show()


#### TRIALS ON HEATMAP ALIGNED TO FIRST LICK BOUT by session sorted by lick bout####
sess_of_interest = [1,3,5,6,7]
start_time = -2
end_time = 10.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    for subj,session,trial in avglickbouttrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(timerange_lick[0], timerange_lick[1], len(avglickbouttrace_dict[subj,i,trial][2]))
            trace = avglickbouttrace_dict[subj,i,trial][2]
            start_index = np.searchsorted(timespace, start_time)
            end_index = np.searchsorted(timespace, end_time)  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            bysession[avglickbouttrace_dict[subj,i,trial][1], trial, session, subj]= trace_segment
           
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=False):
        print(k)
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in sess_of_interest:
    bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
    difference_array=np.absolute(time_segment-0)
    axs[sess_of_interest.index(i)].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(end_time-start_time)), np.arange(start_time,end_time,1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Lick (sec)')
plt.show()


### RUN PCA ###

import pandas as pd
import numpy as np

# Assuming you have the dictionaries filled from the previous processing steps
# For simplicity, let's assume avgcuetrace_dict, avglevertrace_dict, avgflicktrace1_dict contain the necessary data.

def create_pca_matrix(allcuepeakheights, allleverpeakheights, allflickpeakheights):
    pca_data = []

    # Loop through each session and trial
    for (mouse, session, trial) in allcuepeakheights.keys():
        if trial < 10 and session in range (1,8):  
            # Retrieve data
            # cue_data = avgcuetrace_dict[(mouse, session, trial)]
            # lever_data = avglevertrace_dict.get((mouse, session, trial), (None, None))  # Handle missing data
            # flick_data = avgflicktrace1_dict.get((mouse, session, trial), (None, None))
            lever_peak_data = allleverpeakheights.get((mouse, session, trial), (None, None)) 
            flick_peak_data = allflickpeakheights.get((mouse, session, trial), (None, None))
            
            peak_height_cue = allcuepeakheights[mouse, session, trial][0] 
            
            if lever_peak_data == (None, None):
                peak_height_lever = 0
            else:
                peak_height_lever = lever_peak_data[0]
                
            if flick_peak_data == (None, None):
                peak_height_flick = 0
            else:
                peak_height_flick = flick_peak_data[0]
                
            # Licks prior (counting from trials)
            licks_prior = 0
            for t in range(trial-1):
                if (mouse, session, t) in avglickbouttrace_dict:
                    licks_prior = avglickbouttrace_dict[mouse, session, t][1]

            
            # # Latency (assuming the second value of lever_data is the time of lever press)
            # if lever_data == (None, None):
            #     latency_press = 20
            # else:
            #     latency_press = lever_data[0] -  cue_data[0] 
            
            # if lever_data == (None, None):
            #     latency_lick = 30
            # elif flick_data == (None, None):
            #     latency_lick = 10
            # else:
            #     latency_lick = flick_data[0] -  lever_data[0]
            
            # Compile data into a row of the matrix
            pca_data.append([
                session,
                trial,
                licks_prior,  
                peak_height_cue,
                peak_height_lever,
                peak_height_flick
            ])

    # Create DataFrame for PCA analysis
    pca_matrix = pd.DataFrame(pca_data, columns=['session', 'trial', 'licks_prior', 'peak_height_cue', 
                                                 'peak_height_lever', 
                                                 'peak_height_flick'])
    
    
    return pca_matrix

# Construct PCA matrix
import scipy.stats
pca_matrix = create_pca_matrix(allcuepeakheights, allleverpeakheights, allflickpeakheights)
print(pca_matrix)

pca_matrix['norm_licks'] = scipy.stats.zscore(pca_matrix['licks_prior'], axis=0)
pca_numeric = pca_matrix[['peak_height_cue','peak_height_lever','peak_height_flick']]

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats
import numpy as np

# Standardize across columns (features), not rows (samples)
org_zscore = scipy.stats.zscore(pca_numeric, axis=0)
matrix = np.array(org_zscore)

# Ensure shape matches your feature list
feature_names = ['peak_height_cue',
                 'peak_height_lever', 
                 'peak_height_flick']

df = pd.DataFrame(matrix, columns=feature_names)

# PCA
pca = PCA(n_components=3)
pca.fit(df)
data_pc = pca.transform(df)

# Loadings and Variance
loadings = pca.components_
loadings_pc1 = loadings[0]
loadings_pc2 = loadings[1]

explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio for PCA1: {explained_variance_ratio[0]:.4f}")
print(f"Explained Variance Ratio for PCA2: {explained_variance_ratio[1]:.4f}")

# Contributions to PC1
contributions_pc1 = pd.DataFrame({
    'Feature': df.columns,
    'Contribution to PC1': loadings_pc1
})
contributions_pc1 = contributions_pc1.reindex(
    contributions_pc1['Contribution to PC1'].abs().sort_values(ascending=False).index
)
print(contributions_pc1)

# Contributions to PC2
contributions_pc2 = pd.DataFrame({
    'Feature': df.columns,
    'Contribution to PC2': loadings_pc2
})
contributions_pc2 = contributions_pc2.reindex(
    contributions_pc2['Contribution to PC2'].abs().sort_values(ascending=False).index
)
print(contributions_pc2)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Standardize the data (important for k-means)
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)


# Fit PCA on the standardized data
pca_result = pca.fit_transform(df_standardized)

# Get the transformed data (with reduced dimensions)
plt.figure()
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2','PC3'])
plt.scatter(df_pca['PC1'], df_pca['PC2'], cmap='viridis')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()



# PC1 V. PC2 WITH LOADINGS WITH THE MALE V FEMALE
plt.figure()
plt.grid(True, alpha=0.2)
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.4f} Explained Variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.4f} Explained Variance)')
# Add arrows for feature loadings
text = list(df.columns)
for i, feature in enumerate(df.columns):
    if i in range(8):
        plt.arrow(0, 0, loadings_pc1[i], loadings_pc2[i], alpha=0.3, width=0.01, head_width=0.015)
        plt.annotate(text[i], (loadings_pc1[i], loadings_pc2[i] + 0.005)) 
plt.show()


import matplotlib.pyplot as plt
pastel2 = plt.get_cmap('cividis')
colors = [pastel2(i) for i in range(pastel2.N)]  # pastel2.N = 8
color_range = np.linspace(0,8,len(colors))


plt.figure(figsize=(8, 6))
for i in range(len(df_pca)):
    colorindex = np.searchsorted(color_range, pca_matrix.loc[i,'session'])
    plt.scatter(
        x=df_pca.loc[i, 'PC1'],
        y=df_pca.loc[i, 'PC2'],
        color=colors[colorindex],  # replace with 'trial' to color by trial instead
        label=pca_matrix.loc[i,'session'],
        alpha=0.5
    )
plt.title('PCA: PC1 vs PC2 colored by Session')
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel(f'PC1 ({round(0.4136 * 100, 1)}% Variance)')
plt.ylabel(f'PC2 ({round(0.1971 * 100, 1)}% Variance)')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Dictionary removes duplicates
plt.legend(by_label.values(), by_label.keys(), title='Session')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
pastel2 = plt.get_cmap('cividis')
colors = [pastel2(i) for i in range(pastel2.N)]  # pastel2.N = 8
color_range = np.linspace(0,10,len(colors))

plt.figure(figsize=(8, 6))
for i in range(len(df_pca)):
    colorindex = np.searchsorted(color_range, pca_matrix.loc[i,'trial'])
    plt.scatter(
        x=df_pca.loc[i, 'PC1'],
        y=df_pca.loc[i, 'PC2'],
        color=colors[colorindex],  # replace with 'trial' to color by trial instead
        label=pca_matrix.loc[i,'trial'],
        alpha=0.5
    )
plt.title('PCA: PC1 vs PC2 colored by Trial')
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel(f'PC1 ({round(0.4136 * 100, 1)}% Variance)')
plt.ylabel(f'PC2 ({round(0.1971 * 100, 1)}% Variance)')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Dictionary removes duplicates
plt.legend(by_label.values(), by_label.keys(), title='Trial')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
cmap =plt.get_cmap('cividis') 
colors = [cmap(i) for i in range(cmap.N)]  # pastel2.N = 8
color_range = np.linspace(-3,3, len(colors))


plt.figure(figsize=(8, 6))
for i in range(len(df_pca)):
    
    colorindex = np.searchsorted(color_range, pca_matrix.loc[i,'norm_licks'])
    plt.scatter(
        x=df_pca.loc[i, 'PC1'],
        y=df_pca.loc[i, 'PC2'],
        color=colors[colorindex],  # replace with 'lickprior' to color by trial instead
        s=60,
        label=pca_matrix.loc[i,'norm_licks'],
        alpha=0.7
    )
plt.title('PCA: PC1 vs PC2 colored by Normalized Licks')
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel(f'PC1 ({round(0.4136 * 100, 1)}% Variance)')
plt.ylabel(f'PC2 ({round(0.1971 * 100, 1)}% Variance)')

plt.tight_layout()
plt.show()

#################### FOR EXTINCTION ####################

if folder == '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucExtinction/':
    available = plt.get_cmap('GnBu')
    av_colors = [available(i) for i in range(available.N)]  # pastel2.N = 8
    extinction = plt.get_cmap('PuRd')
    ext_colors = [extinction(i) for i in range(extinction.N)]  # pastel2.N = 8
    plt.figure(figsize=(8, 6))
    for i in range(len(df_pca)):
        if pca_matrix.loc[i,'session'] in [0,10]:
            plt.scatter(x=df_pca.loc[i, 'PC1'], y=df_pca.loc[i, 'PC2'],
                color='g',  
                s=60, label='Available')
        elif pca_matrix.loc[i,'session'] in [1,2,3,4,5,6]:
            plt.scatter(x=df_pca.loc[i, 'PC1'], y=df_pca.loc[i, 'PC2'],
                color=ext_colors[pca_matrix.loc[i,'session']*20+10],  
                s=60, label='Extinction')
    plt.title('PCA: PC1 vs PC2 colored by Extinction')
    plt.axhline(0, linestyle='--', color='gray')
    plt.axvline(0, linestyle='--', color='gray')
    plt.xlabel(f'PC1 ({round(0.4136 * 100, 1)}% Variance)')
    plt.ylabel(f'PC2 ({round(0.1971 * 100, 1)}% Variance)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Dictionary removes duplicates
    plt.legend(by_label.values(), by_label.keys(), title='Available')
    plt.tight_layout()
    plt.show()


if folder == '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHExtinction/':
    available = plt.get_cmap('GnBu')
    av_colors = [available(i) for i in range(available.N)]  # pastel2.N = 8
    extinction = plt.get_cmap('PuRd')
    ext_colors = [extinction(i) for i in range(extinction.N)]  # pastel2.N = 8
    plt.figure(figsize=(8, 6))
    for i in range(len(df_pca)):
        if pca_matrix.loc[i,'session'] in [0,7]:
            plt.scatter(x=df_pca.loc[i, 'PC1'], y=df_pca.loc[i, 'PC2'],
                color='g',  
                s=60, label='Available')
        elif pca_matrix.loc[i,'session'] in [1,2,3,4,5,6]:
            plt.scatter(x=df_pca.loc[i, 'PC1'], y=df_pca.loc[i, 'PC2'],
                color=ext_colors[pca_matrix.loc[i,'session']*20+10],  
                s=60, label='Extinction')
    plt.title('PCA: PC1 vs PC2 colored by Extinction')
    plt.axhline(0, linestyle='--', color='gray')
    plt.axvline(0, linestyle='--', color='gray')
    plt.xlabel(f'PC1 ({round(0.4136 * 100, 1)}% Variance)')
    plt.ylabel(f'PC2 ({round(0.1971 * 100, 1)}% Variance)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Dictionary removes duplicates
    plt.legend(by_label.values(), by_label.keys(), title='Available')
    plt.tight_layout()
    plt.show()

#################### FOR EXTINCTION ####################


import statsmodels.formula.api as smf

model = smf.ols('peak_height_cue ~ licks_prior + latency_press + latency_lick + session + trial', data=df).fit()
print(model.summary())

from sklearn.cluster import KMeans

X = df_pca[['PC1', 'PC2']]
kmeans = KMeans(n_clusters=3).fit(X)
df_pca['cluster'] = kmeans.labels_


import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

import seaborn as sns
sns.lmplot(x='licks_prior', y='peak_height_cue', hue='trial', data=pca_matrix)

import matplotlib.pyplot as plt

avg = pca_matrix.groupby('session')['peak_height_flick'].mean()
plt.plot(avg.index, avg.values)
plt.xlabel('Session')
plt.ylabel('Average Peak Height (Flick)')
plt.title('Peak Height Over Sessions')

####################################################################################
####################################################################################
####################################################################################
############################################################################################################################
#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by latency to lick ####
        

colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

sess_of_interest = [0,1]
end_time = 5.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    for subj,session,trial in totalactivetrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(active_time[0], active_time[1], len(totalactivetrace_dict[subj,i,trial][3]))
            trace = totalactivetrace_dict[subj,i,trial][3]
            start_index = np.searchsorted(timespace, start_time)
            end_index = np.searchsorted(timespace, end_time)  
            time_segment = timespace[start_index:end_index]
            trace_segment = trace[start_index:end_index]
            bysession[np.average(trace_segment)]= trace_segment
           
            # if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
            #     bysession[20 + totalactivetrace_dict[subj,i,trial][0]]= trace_segment
            # else:
            #     bysession[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= trace_segment
    sorted_bysession=[]
    for k in sorted(bysession.keys(),reverse=True):
        #print(i)
        sorted_bysession.append(bysession[k])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(len(sess_of_interest), sharex=True)
for i in sess_of_interest:
    bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
    sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
    difference_array=np.absolute(time_segment-0)
    axs[sess_of_interest.index(i)].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(end_time-start_time)), np.arange(start_time,end_time,1, dtype=int),rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time Aligned to Cue (sec)')
plt.show()

#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by latency to lick ####
######### for the ommission session separately #########


alltraces=[]
for i in range(8):
    bysession = {}
    for subj,session,trial in totalactivetrace_dict:
        if session == i and trial in range (10):
            if np.isnan(totalactivetrace_dict[subj,i,trial][2]):
                bysession[totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
            else:
                bysession[totalactivetrace_dict[subj,i,trial][2]-totalactivetrace_dict[subj,i,trial][0]]= totalactivetrace_dict[subj,i,trial][3]
    sorted_bysession=[]
    for i in reversed(sorted(bysession.keys())):
        #print(i)
        sorted_bysession.append(bysession[i])
    alltraces.append(sorted_bysession)

fig, axs = plt.subplots(3, sharex=True, figsize=(10,12))
for i in range (8):
    bysessiondf= pd.DataFrame(alltraces[i])    
    sns.heatmap(bysessiondf, cmap=sns.diverging_palette(255, 28, l=68, as_cmap=True), vmin=-5, vmax=5, ax=axs[i])
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
    if trial in range(10) and session in range (4,5):
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
plt.savefig('/Users/kristineyoon/Documents/heatmaptracesallmice.pdf', transparent=True)



#### TRIALS ON HEATMAP ALIGNED TO FIRST LICK by session sorted by SIGNAL MEAN ####
plt.figure(figsize=(10, 12))
bysession = {}
for subj,session,trial in avgflicktrace_dict:
    if trial < 10:
        bysession[np.mean(avgflicktrace_dict[subj,session,trial][1])]=avgflicktrace_dict[subj,session,trial][1]
sorted_bysession = []
for i in reversed(sorted(bysession.keys())):
    sorted_bysession.append(bysession[i])
bysessiondf= pd.DataFrame(sorted_bysession)           
sns.heatmap(bysessiondf, cmap=sns.diverging_palette(265, 28, l=68, as_cmap=True), vmin=-5, vmax=5)
plt.axvline(x=(bysessiondf.shape[1])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black', label='First Lick Onset')
plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(timerange_lick[1]-timerange_lick[0])*2), 
            np.arange(timerange_lick[0], timerange_lick[1],2, dtype=int),
            rotation=0)
plt.ylabel('Trials')
plt.xlabel('Time (sec)')
plt.show()
plt.savefig('/Users/kristineyoon/Documents/heatmaptracesallmice_licks.pdf', transparent=True)

############################################################################################################################
############################################################################################################################
### BEHAVIORAL PARADIGMS
############################################################################################################################
############################################################################################################################
licksinsession_df=pd.DataFrame(columns=mice)
for i in range(12):
    licks=[]
    for  mouse, session, trial in lickspersession:
        if session == i:
            licksinsession_df.at[i,mouse] = lickspersession[mouse,session,trial]

session_ranges = len(files)

licksinsession1_df=pd.DataFrame(columns=mice)
for i in range(session_ranges):
    for subj in mice:
        lickcount=0
        for mouse, session, trial in avglickbouttrace_dict:
            if session == i and mouse == subj and trial in range(10):
                lickcount=lickcount + avglickbouttrace_dict[mouse, session, trial][1]
        licksinsession1_df.at[i,subj] = lickcount
            
responseinsession_df=pd.DataFrame(columns=mice)
for i in range(12):
    licks=[]
    for  mouse, session, trial in responsepersession:
        if session == i:
            responseinsession_df.at[i,mouse] = responsepersession[mouse,session,trial]

timeout_df=pd.DataFrame(columns=mice)
for i in range(12):
    licks=[]
    for  mouse, session, trial in timeoutpersession:
        if session == i:
            timeout_df.at[i,mouse] = timeoutpersession[mouse,session,trial]
            



# Latency to press
latency_to_leverpress = {}
for  cue_mouse, cue_session, cue_trial in avgcuetrace_dict:
    cue_time = avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0]
    for lever_mouse, lever_session, lever_trial in avglevertrace_dict:
        lever_time = avglevertrace_dict[lever_mouse, lever_session, lever_trial][0]
        if cue_mouse == lever_mouse and cue_session == lever_session:
            if lever_time - cue_time > 0 and lever_time - cue_time < 20:
                latency_to_leverpress[cue_mouse, cue_session, cue_trial] = lever_time - cue_time

plt.figure()
for i in range (12):
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

latency_to_leverpress_df=pd.DataFrame()
count = 0
for mouse, session, trial in latency_to_leverpress:
    latency_to_leverpress_df.at[count,'Session'] = session
    latency_to_leverpress_df.at[count,'Latency'] = latency_to_leverpress[mouse, session, trial]
    count = count + 1

plt.figure()
for i in range (12):
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
for i in range (12):
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

lickbout_df=pd.DataFrame()
count = 0
for mouse, session, trial in avglickbouttrace_dict:
    lickbout_df.at[count,'Session'] = session
    lickbout_df.at[count,'Bouts'] = avglickbouttrace_dict[mouse, session, trial][1]
    count = count + 1

# Latency to lick

latency_to_lick = {}
for  cue_mouse, cue_session, cue_trial in avgcuetrace_dict:
    cue_time = avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0]
    for lick_mouse, lick_session, lick_trial in avgflicktrace_dict:
        if cue_mouse == lick_mouse and cue_session == lick_session:
            if avgflicktrace_dict[lick_mouse, lick_session, lick_trial][0] - avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0] > 0 and avgflicktrace_dict[lick_mouse, lick_session, lick_trial][0] - avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0] <30:
                latency_to_lick[cue_mouse, cue_session, cue_trial] = avgflicktrace_dict[lick_mouse, lick_session, lick_trial][0] - avgcuetrace_dict[cue_mouse, cue_session, cue_trial][0]
latency_to_lick_df=pd.DataFrame()
count = 0
for mouse, session, trial in latency_to_lick:
    latency_to_lick_df.at[count,'Session'] = session
    latency_to_lick_df.at[count,'Latency'] = latency_to_lick[mouse, session, trial]
    count = count + 1


############################################################################################################################
############################################################################################################################
### AREA UNDER THE CURVE
############################################################################################################################
############################################################################################################################
# Calculates the area under the curve in that interval using Simpson's rule, which is generally accurate for oscillating data like neural traces.
##############################################################
import numpy as np
from scipy.integrate import simps

auc_cue_dict = {}
for cue_mouse, cue_session, cue_trial in avgcuetrace_dict:
    if cue_trial in range(10):
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[cue_mouse, cue_session, cue_trial][1]))  # 500 points from -2 to 10 seconds
        trace = avgcuetrace_dict[cue_mouse, cue_session, cue_trial][1]

        
        # Define the interval of interest
        start_time = 0
        end_time = 5.01
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_cue_dict[cue_mouse, cue_session, cue_trial] = area

auc_cue_df=pd.DataFrame()
count = 0
for mouse, session, trial in auc_cue_dict:
    auc_cue_df.at[count,'Session'] = session
    auc_cue_df.at[count,'Mouse'] = mouse
    auc_cue_df.at[count,'AUC'] = auc_cue_dict[mouse, session, trial]
    count = count + 1


plt.figure()
for i in range (10):
    mean_session=[]
    for mouse, session, trial in auc_cue_dict:
        if session == i:
            mean_session.append(auc_cue_dict[mouse, session,trial])
            plt.scatter(x=session,y= auc_cue_dict[mouse, session, trial],color=colors10[i], alpha=0.01)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Area Under Curve')
plt.show()



### for first lick
auc_flick_dict = {}
for cue_mouse, cue_session, cue_trial in avgflicktrace_dict:
    if cue_trial in range(10):
        time = np.linspace(timerange_lick[0], timerange_lick[1], 122)  #for timerange -2,10
        trace = avgflicktrace_dict[cue_mouse, cue_session, cue_trial][1][:122]
        
        # Define the interval of interest
        start_time = 0
        end_time = 10
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_flick_dict[cue_mouse, cue_session, cue_trial] = area

auc_flick_df=pd.DataFrame()
count = 0
for mouse, session, trial in auc_flick_dict:
    auc_flick_df.at[count,'Session'] = session
    auc_flick_df.at[count,'AUC'] = auc_flick_dict[mouse, session, trial]
    count = count + 1


plt.figure()
for i in range (8):
    mean_session=[]
    for mouse, session, trial in auc_flick_dict:
        if session == i:
            mean_session.append(auc_flick_dict[mouse, session,trial])
            plt.scatter(x=session,y= auc_flick_dict[mouse, session, trial],color=colors10[i], alpha=0.05)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Area Under Curve')
plt.show()


############################################################################################################################
############################################################################################################################
### LOOKING MORE INDEPTH AT TRACES AT RESPONDED LEVER PRESSES
############################################################################################################################
############################################################################################################################

respondingcue_trace = {}
for cue_mouse, cue_session, cue_trial in latency_to_leverpress:
    for mouse, session, trial in allcuepeakheights:
        if cue_mouse == mouse and cue_session==session and cue_trial==trial:
            respondingcue_trace[cue_mouse, cue_session, cue_trial]=avgcuetrace_dict[mouse, session, trial]


auc_respond_cue_dict = {}
for cue_mouse, cue_session, cue_trial in respondingcue_trace:
    if cue_trial in range(10):
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(respondingcue_trace[cue_mouse, cue_session, cue_trial][1]))  # 500 points from -2 to 10 seconds
        trace = respondingcue_trace[cue_mouse, cue_session, cue_trial][1]
        
        # Define the interval of interest
        start_time = 0
        end_time = 1
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_respond_cue_dict[cue_mouse, cue_session, cue_trial] = area

auc_respondingcue_df=pd.DataFrame()
count = 0
for mouse, session, trial in auc_respond_cue_dict:
    auc_respondingcue_df.at[count,'Session'] = session
    auc_respondingcue_df.at[count,'Mouse'] = mouse
    auc_respondingcue_df.at[count,'AUC'] = auc_respond_cue_dict[mouse, session, trial]
    count = count + 1


plt.figure()
for i in range (8):
    mean_session=[]
    for mouse, session, trial in auc_respond_cue_dict:
        if session == i:
            mean_session.append(auc_respond_cue_dict[mouse, session,trial])
            plt.scatter(x=session,y= auc_respond_cue_dict[mouse, session, trial],color=colors10[i], alpha=0.01)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Area Under Curve')
plt.show()


### for lever press

auc_leverpress_dict = {}
for cue_mouse, cue_session, cue_trial in avglevertrace_dict:
    if cue_trial in range(10):
        time = np.linspace(timerange_lever[0], timerange_lever[1], len(avglevertrace_dict[cue_mouse, cue_session, cue_trial][1]))  # 500 points from -2 to 10 seconds
        trace = avglevertrace_dict[cue_mouse, cue_session, cue_trial][1]
        
        # Define the interval of interest
        start_time = -0.5
        end_time = 0.5
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_leverpress_dict[cue_mouse, cue_session, cue_trial] = area

auc_lever_df=pd.DataFrame()
count = 0
for mouse, session, trial in auc_leverpress_dict:
    auc_lever_df.at[count,'Session'] = session
    auc_lever_df.at[count,'AUC'] = auc_leverpress_dict[mouse, session, trial]
    count = count + 1


plt.figure()
for i in range (10):
    mean_session=[]
    for mouse, session, trial in auc_leverpress_dict:
        if session == i:
            mean_session.append(auc_leverpress_dict[mouse, session,trial])
            plt.scatter(x=session,y= auc_leverpress_dict[mouse, session, trial],color=colors10[i], alpha=0.01)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Area Under Curve')
plt.show()




########################## DOES THE PEAK HEIGHT OF Cues CORRELATE WITH LATENCY TO LICK ##########################
##################################################################################################################
i=0
peakheightcuetolatency=pd.DataFrame(columns=['Latency', 'Peak Height'])
for mouse1,session1,trial1 in allcuepeakheights:
    for mouse2,session2,trial2 in latency_to_lick:
        if mouse1==mouse2 and session1==session2 and trial1==trial2:
            peakheightcuetolatency.at[i,'Latency'] = latency_to_lick[mouse2,session2,trial2]
            peakheightcuetolatency.at[i,'Peak Height'] = allcuepeakheights[mouse1,session1,trial1]
            peakheightcuetolatency.at[i,'Session'] = session2
            i = i+1
plt.scatter(x=peakheightcuetolatency['Latency'],y= peakheightcuetolatency['Peak Height'], color=colors10[session], alpha=0.5)

########################## DOES THE PEAK HEIGHT OF Cues CORRELATE WITH LATENCY TO LEVER PRESS ##########################
##################################################################################################################
i=0
peakheightcuetoleverlatency=pd.DataFrame(columns=['Latency', 'Peak Height'])
for mouse1,session1,trial1 in allcuepeakheights:
    for mouse2,session2,trial2 in latency_to_leverpress:
        if mouse1==mouse2 and session1==session2 and trial1==trial2:
            peakheightcuetoleverlatency.at[i,'Latency'] = latency_to_leverpress[mouse2,session2,trial2]
            peakheightcuetoleverlatency.at[i,'Peak Height'] = allcuepeakheights[mouse1,session1,trial1]
            peakheightcuetoleverlatency.at[i,'Session'] = session2
            i = i+1
plt.scatter(x=peakheightcuetolatency['Latency'],y= peakheightcuetoleverlatency['Peak Height'], color=colors10[session], alpha=0.5)




########################## DOES THE PEAK HEIGHT OF LICKS CORRELATE WITH LATENCY TO LICK ##########################
##################################################################################################################
i=0
peakheighttolatency=pd.DataFrame(columns=['Latency', 'Peak Height'])
for mouse1,session1,trial1 in alllickpeackheight:
    for mouse2,session2,trial2 in latency_to_lick:
        if mouse1==mouse2 and session1==session2 and trial1==trial2:
            peakheighttolatency.at[i,'Latency'] = latency_to_lick[mouse2,session2,trial2]
            peakheighttolatency.at[i,'Peak Height'] = alllickpeackheight[mouse1,session1,trial1]
            peakheighttolatency.at[i,'Session'] = session2
            i = i+1
plt.scatter(x=peakheighttolatency['Latency'],y= peakheighttolatency['Peak Height'], color=colors10[session], alpha=0.5)


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
#LOOKING AT THE LICK BOUTS THROUGHOUT THE SESSIONS
lickbout_data = {}
for i in range(8):
    session_data = {}
    for key, value in avglickbouttrace_dict.items():
        mouse, session, trial = key
        time, bout, newtrace = value
        if bout in range (10):
            lickbout = 0
        elif bout in range (10,30):
            lickbout =1
        elif bout in range (30,50):
            lickbout =2
        elif bout in range (50,70):
            lickbout = 3
        elif bout >= 70:
            lickbout =4
        else:
            print('lick was not assigned')

        if lickbout not in lickbout_data:
            lickbout_data[lickbout] ={}
        
        if session == i:
            if session not in lickbout_data[lickbout]:
                lickbout_data[lickbout][session] = []
            lickbout_data[lickbout][session].append(newtrace)


from scipy.stats import sem
fig, axs = plt.subplots(4, sharex=True, figsize=(10, 12))
for i in range(1,5):  # Loop through lick types (0-4)
    for licktype, session_data in lickbout_data.items():
        if licktype == i:
            for session in range(8):  # Loop through sessions (0-7)
                mean_trace = None
                sem_trace = None
                
                if session in session_data:  # Check if session exists in data
                    traces = session_data[session]
                    
                    if traces:  # Ensure traces list is not empty
                        mean_trace = np.mean(traces, axis=0)
                        sem_trace = sem(traces, axis=0)

                if mean_trace is not None and sem_trace is not None:
                    axs[i-1].plot(mean_trace, color=colors10[session], label=f'Session {session}')
                    axs[i-1].fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace,
                                        color=colors10[session], alpha=0.1)

            axs[i-1].axhline(y=0, linestyle=':', color='black')
            axs[i-1].axvline(x=len(newtrace) / (timerange_lick[1] - timerange_lick[0]) * (0 - timerange_lick[0]),
                           linewidth=1, color='black')
            axs[i-1].legend()

# Set X-axis ticks and labels
plt.xticks(np.arange(0, len(newtrace) + 1, len(newtrace) / (timerange_lick[1] - timerange_lick[0])),
           np.arange(timerange_lick[0], timerange_lick[1] + 1, 1, dtype=int),
           rotation=0)
plt.xlabel('Bout Onset (s)')
fig.tight_layout()
plt.show()


##################################################################################################################
##################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import sem

# Dictionary to store peak heights categorized by lick bout
allboutpeakheights = {}
peakheight_lick_df = pd.DataFrame(columns=['LickBout', 'Session', 'X-Axis', 'PeakHeight'])

# Loop through lick bout categories
for lickbout, session_dicts in lickbout_data.items():
    for session, traces in session_dicts.items():  # Each session dictionary
        for trace in traces:
            param_prom = 1  # Peak prominence threshold
            peaks, properties = find_peaks(trace, prominence=param_prom, height=-.4)
            peak_heights, peak_times, prominences = [], [], []
            for k in range(len(peaks)):
                peak_time = peaks[k]
                if (len(trace) / (timerange_lick[1] - timerange_lick[0]) * (0 - timerange_lick[0]) < peak_time < 
                    len(trace) / (timerange_lick[1] - timerange_lick[0]) * (1 - timerange_lick[0])):
                    peak_heights.append(properties['peak_heights'][k])
                    peak_times.append(peak_time)
                    prominences.append(properties['prominences'][k])

            # Assign peak height data
            if peak_heights:
                best_peak_idx = prominences.index(max(prominences))
                best_peak_time = peak_times[best_peak_idx]
                best_peak_height = peak_heights[best_peak_idx]

                # Save peak height categorized by lick bout
                peakheight_lick_df.loc[len(peakheight_lick_df)] = [lickbout, session, best_peak_time, best_peak_height]
                allboutpeakheights[(lickbout, session, len(allboutpeakheights))] = (best_peak_height, best_peak_time)
            else:
                peakheight_lick_df.loc[len(peakheight_lick_df)] = [lickbout, session, 0, np.nan]
                allboutpeakheights[(lickbout, session, len(allboutpeakheights))] = (np.nan, np.nan)

# --- PLOTTING PEAK HEIGHTS ACROSS SESSIONS FOR EACH LICK BOUT ---
plt.figure(figsize=(10, 6))

legend_handles = []
legend_labels = []

for i in range(5):
    scatter = None
    for k in range(8):
        session_peakheight = []
        for (lickbout, session, length), (peak_height, _) in allboutpeakheights.items():
            if lickbout == i and session == k and not np.isnan(peak_height):
                session_peakheight.append(peak_height)
        if session_peakheight:
            mean_height = np.mean(session_peakheight)
            plt.errorbar(x=k, y=mean_height, yerr=sem(session_peakheight), ecolor=colors10[i], capsize=3)
            scatter = plt.scatter(x=k, y=mean_height, color=colors10[i], alpha=0.8)
    # Add a single legend entry per lick bout category
    if scatter:
        legend_handles.append(scatter)
        legend_labels.append(f'LickBout {i}')
plt.xlabel('Session')
plt.ylabel('Peak Height At Lickbout')
plt.legend(legend_handles, legend_labels, title="Lick Bout Categories", loc="upper right")
plt.show()

#run stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Convert categorical variables (LickBout and Session)
peakheight_lick_df['LickBout'] = peakheight_lick_df['LickBout'].astype(int)
peakheight_lick_df['Session'] = peakheight_lick_df['Session'].astype(int)

# Drop NaN values to ensure valid analysis
peakheight_lick_df = peakheight_lick_df.dropna(subset=['PeakHeight'])

# Fit a two-way ANOVA model
model = ols('PeakHeight ~ C(LickBout) + C(Session) + C(LickBout):C(Session)', data=peakheight_lick_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA (recommended for balanced designs)

# Print results
print(anova_table)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(peakheight_lick_df['PeakHeight'], peakheight_lick_df['LickBout'])
print(tukey)
tukey1 = pairwise_tukeyhsd(peakheight_lick_df['PeakHeight'], peakheight_lick_df['Session'])
print(tukey1)


# # --- PLOTTING INDIVIDUAL PEAK HEIGHTS & TRACES PER SESSION ---
# fig, axs = plt.subplots(5, sharey=True, sharex=True, figsize=(10, 12))

# for i in range(5):  # Loop through lick bout categories
#     for session_dict in lickbout_data.get(i, []):  # Get session data for each lick bout
#         for session, traces in session_dict.items():
#             for trace in traces:
#                 peak_height, peak_time = allboutpeakheights.get((i, session), (np.nan, np.nan))
#                 if peak_time != 0:
#                     axs[i].scatter(x=peak_time, y=peak_height, c=colors10[int(session)])
#                 axs[i].plot(trace, color=colors10[int(session)], label=f'Session {session}', alpha=0.2)
#                 axs[i].axvline(x=len(trace) / (timerange_lick[1] - timerange_lick[0]) * (0 - timerange_lick[0]), linewidth=0.5, color='black')

# plt.xticks(
#     np.arange(0, len(trace) + 1, len(trace) / (timerange_lick[1] - timerange_lick[0])),
#     np.arange(timerange_lick[0], timerange_lick[1] + 1, 1, dtype=int),
#     rotation=0
# )
# plt.xlabel('Bout Onset (s)')
# fig.tight_layout()
# plt.show()

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


########################### SEPARATE BUT SAME VARIABLES for lick bouts ########################
allpeackheight={}
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
                peakheight_lickbout_df.at[num,'Bout']=avglickbouttrace_dict[str(mouse),session,trial][1]
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
boutpeakheight = []
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


##############################################################
##############################################################
##############################################################
### FIGURE 2 ANALYSIS
### PAST PRESENT FUTURE


### sum of all lick bouts before cue

totallicktocue_dict = {}
for mouse, session, trial in avglickbouttrace_dict:
    next_cue = trial+1
    if (mouse, session, next_cue) in allcuepeakheights.keys():
        total_lick = 0
        for k in range (0,next_cue):
            if (mouse, session, k) in avglickbouttrace_dict.keys():
                total_lick = total_lick + avglickbouttrace_dict[mouse, session, k][1]
        totallicktocue_dict[mouse, session, next_cue] = allcuepeakheights[mouse, session, next_cue][0], total_lick

plt.figure()
#for i in [3,7]:
for i in range(1, session_ranges):
    x=[]
    y=[]
    for mouse,session,trial in totallicktocue_dict:

        if session == i:
            x.append(totallicktocue_dict[mouse,session,trial][0])
            y.append(totallicktocue_dict[mouse,session,trial][1])
            plt.scatter(x=totallicktocue_dict[mouse,session,trial][0],y= totallicktocue_dict[mouse,session,trial][1], color=colors10[session], alpha=0.5)
    if len(x) > 0:
        a, b = np.polyfit(x, y, 1)
        plt.plot(np.array(x), a*np.array(x)+b, color=colors10[i], label= str(i))
        plt.legend()

plt.xlabel('Cue peak Height')
plt.ylabel('all previous lickbouts')
plt.show()



#### how about after?

triallicktocue_dict = {}
bout_to_cue = pd.DataFrame(
    columns=['Mouse', 'Session', 'Trial', 'PeakHeight', 'PreviousBout'])
for mouse, session, trial in avglickbouttrace_dict:
    if (mouse, session, trial) in allcuepeakheights.keys():
        triallicktocue_dict[mouse, session, trial] = allcuepeakheights[mouse, session, trial][0], avglickbouttrace_dict[mouse, session, trial][1]

plt.figure()
for i in range(1, session_ranges):
#for i in [3,7]:
    x=[]
    y=[]
    for mouse,session,trial in triallicktocue_dict:

        if session == i:
            x.append(triallicktocue_dict[mouse,session,trial][0])
            y.append(triallicktocue_dict[mouse,session,trial][1])
            plt.scatter(x=triallicktocue_dict[mouse,session,trial][0],y= triallicktocue_dict[mouse,session,trial][1], color=colors10[session], alpha=0.5)
    if len(x) > 0:
        a, b = np.polyfit(x, y, 1)
        
        plt.plot(np.array(x), a*np.array(x)+b, color=colors10[i], label= str(i))
        plt.legend()

plt.xlabel('trial Cue peak Height')
plt.ylabel('trial lickbout')
plt.show()


##### does previous lick bout show anything about peak height of cue after ####

previouslicktocue_dict = {}
bout_to_cue = pd.DataFrame(
    columns=['Mouse', 'Session', 'Trial', 'PeakHeight', 'PreviousBout'])
for mouse, session, trial in avglickbouttrace_dict:
    next_cue = trial+1
    if (mouse, session, next_cue) in allcuepeakheights.keys():
        previouslicktocue_dict[mouse, session, next_cue] = allcuepeakheights[mouse, session, next_cue][0], avglickbouttrace_dict[mouse, session, trial][1]

plt.figure()
for i in range(1, session_ranges):
#for i in [3,7]:
    x=[]
    y=[]
    for mouse,session,trial in previouslicktocue_dict:

        if session == i:
            x.append(previouslicktocue_dict[mouse,session,trial][0])
            y.append(previouslicktocue_dict[mouse,session,trial][1])
            plt.scatter(x=previouslicktocue_dict[mouse,session,trial][0],y= previouslicktocue_dict[mouse,session,trial][1], color=colors10[session], alpha=0.5)
    if len(x) > 0:
        a, b = np.polyfit(x, y, 1)
        plt.plot(np.array(x), a*np.array(x)+b, color=colors10[i], label= str(i))
        plt.legend()

plt.xlabel('Cue peak Height')
plt.ylabel('previous lickbout')
plt.show()


##############################################################
##############################################################
##############################################################

##############################################################
### responding cue height v. session
##############################################################
respondingcue_peakheights = {}
for cue_mouse, cue_session, cue_trial in latency_to_leverpress:
    for mouse, session, trial in allcuepeakheights:
        if cue_mouse == mouse and cue_session==session and cue_trial==trial:
            respondingcue_peakheights[cue_mouse, cue_session, cue_trial]=allcuepeakheights[mouse, session, trial]

ressponding_df=pd.DataFrame()
count = 0

for subj in mice:
    for i in range(8):
        response_count = 0
        for mouse, session, trial in respondingcue_peakheights:
            if subj == mouse and session == i and trial in range(10):
                response_count = response_count+1
        ressponding_df.at[count,'Session'] = i
        ressponding_df.at[count,'Latency'] = response_count
        count = count + 1


    
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
