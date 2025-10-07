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
    if session in range(session_ranges):
    #if session in range(1,7):
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
plt.ylim(-1,4)

plt.savefig('/Users/kristineyoon/Documents/cuebysessions.pdf', transparent=True)
plt.show()

##############################################################
#by session average --- lever press
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
    if session in range(session_ranges):
    #if session in range(1,7):
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
plt.savefig('/Users/kristineyoon/Documents/leverbysessions.pdf', transparent=True)


##############################################################
#by session average --- first lick
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
    #if session in range(1,8):
    if session in range (1,session_ranges):
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

plt.savefig('/Users/kristineyoon/Documents/firstlickbysession.pdf', transparent=True)



# --------------------------------------------
#looking at cues that animals licked v. animals didnt lick


session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    if key in avgflicktrace1_dict:
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Responding Cue Aligned Trace with SEM by Session')
plt.legend()
plt.savefig('/Users/kristineyoon/Documents/respondedcuebysessions.pdf', transparent=True)
plt.show()

session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    if key in avgflicktrace1_dict:
        pass
    else:
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Non-Responding Cue Aligned Trace with SEM by Session')
plt.legend()
plt.savefig('/Users/kristineyoon/Documents/nonrespondedcuebysessions.pdf', transparent=True)
plt.show()



# --------------------------------------------
#looking at cues that animals licked  v. animals did not lick in previous trial
# licks -- responding cue v. no licks -- responding cue

session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if previoustrial in avgflicktrace1_dict:
        if trial in range (10):
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
    if session in range(1,session_ranges):
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(trialtrace)/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Previous Lick -- Responding Cue Aligned Trace with SEM by Session')
plt.legend()


session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if previoustrial in avgflicktrace1_dict:
        continue
    else:
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
    if session in range(1,session_ranges):
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(trialtrace)/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average No Lick -- Responding Cue Aligned Trace with SEM by Session')
plt.legend()


#--- looking at perfect streak trials or all miss trials


session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if previoustrial in avgflicktrace1_dict and key in avgflicktrace1_dict:
    #if previoustrial not in avgflicktrace1_dict and key not in avgflicktrace1_dict:
        if trial in range (10):
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
    if session in range(1,session_ranges):
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(trialtrace)/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Previous Lick -- Responding Cue Aligned Trace with SEM by Session')
plt.legend()


# --------------------------------------------
#looking at non responding cues that animals licked  v. animals did not lick in previous trial
# licks -- not responding cue v. no licks -- not responding cue

session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if key in avgflicktrace1_dict:
        pass
    else:
        if previoustrial in avgflicktrace1_dict and trial in range(10):
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Lick -- Non-Responding Cue Aligned Trace with SEM by Session')
plt.legend()


session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if key in avgflicktrace1_dict:
        pass
    else:
        if previoustrial not in avgflicktrace1_dict and trial in range(10):
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average No Lick -- Non-Responding Cue Aligned Trace with SEM by Session')
plt.legend()

# --------------------------------------------
### no lick -- cue v. lick -- cue


session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if previoustrial in avgflicktrace1_dict:
        pass
    else:
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average No Lick -- Cue Aligned Trace with SEM by Session')
plt.legend()



session_data = {}  
for key, value in avgcuetrace_dict.items():
    mouse, session, trial = key
    time, trialtrace, baseline = value
    previoustrial = mouse, session, trial-1
    if previoustrial in avgflicktrace1_dict:
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
    #if session in range(1,7):
        plt.plot(mean_trace, color=colors10[int(session)], label=f'Session {session}')
        plt.fill_between(range(len(mean_trace)), mean_trace - sem_trace, mean_trace + sem_trace, color=colors10[int(session)], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(mean_trace)+1,len(mean_trace)/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.ylabel('z-score')
plt.ylim(-3,6)
plt.title('Average Lick -- Cue Aligned Trace with SEM by Session')
plt.legend()


############################################################################################################################
############################################################################################################################
### PEAK HEIGHTS WITH MAX HEIGHT
############################################################################################################################
############################################################################################################################
# plt.figure(figsize=(10, 6))

totalsessions = 8
results = []

for (mouse, session, trial), (_, trace, _) in avgcuetrace_dict.items():

    if trial < 10:
        # Build time array
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(trace))
        
        # Interval of interest
        start_time, end_time = 0, 2
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)

        # Slice trace
        trace_segment = trace[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        # Peak
        peakheight = np.max(trace_segment)
        peak_idx = np.argmax(trace_segment)
        peakheighttime = time_segment[peak_idx]

        # Store as dict row
        results.append({
            "Mouse": mouse,
            "Session": session,
            "Trial": trial,
            "X-Axis": peakheighttime,
            "PeakHeight": peakheight
        })

# Create DataFrame in one go
peak_cue_df = pd.DataFrame(results)

# ---- Aggregate by session and plot ----
plt.figure()
cuepeakheight = []

for session in range(totalsessions):
    session_data = peak_cue_df.loc[peak_cue_df["Session"] == session, "PeakHeight"]

    if not session_data.empty:
        cuepeakheight.append(session_data.values)
        mean_val = session_data.mean()
        err_val = sem(session_data)

        # scatter individual points
        plt.scatter([session]*len(session_data), session_data,
                    color=colors10[session], alpha=0.02)

        # mean + errorbar
        plt.scatter(session, mean_val, color=colors10[session])
        plt.errorbar(session, mean_val, yerr=err_val,
                     ecolor=colors10[session], capsize=3)

plt.xlabel('Session')
plt.ylabel('Peak Height At Cue')
plt.show()

# --------- statistics ---------------------------
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import pingouin as pg

# Ensure data types are categorical where appropriate
peak_cue_df["Mouse"] = peak_cue_df["Mouse"].astype("category")
peak_cue_df["Session"] = peak_cue_df["Session"].astype("category")
peak_cue_df["Trial"] = peak_cue_df["Trial"].astype("category")

# --- Linear Mixed-Effects Model ---
# Random intercept for Mouse (accounts for individual baseline differences)
# Session as a fixed effect
# Trial nested within Session is handled implicitly via the repeated structure
model = mixedlm("PeakHeight ~ Session", data=peak_cue_df, groups=peak_cue_df["Mouse"])
result = model.fit(reml=True)
print(result.summary())

# --- Post hoc tests (pairwise session comparisons) ---
# Using estimated marginal means from pingouin
posthoc = pg.pairwise_tests(
    dv="PeakHeight",
    within="Session",
    subject="Mouse",
    data=peak_cue_df,
    parametric=True,
    padjust="bonf"
)
print("\nPost hoc pairwise comparisons (Session effects):")
print(posthoc)

# --- Optionally: compute within-session trial effects per mouse ---
trial_effects = (
    peak_cue_df.groupby(["Mouse", "Session", "Trial"])["PeakHeight"]
    .mean()
    .reset_index()
)
trial_effects_summary = (
    trial_effects.groupby(["Session", "Trial"])["PeakHeight"]
    .agg(["mean", "sem"])
    .reset_index()
)
print("\nWithin-session trial-level summary:")
print(trial_effects_summary)

##### --------- looking at trials 0 and 9------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# --- Create summary dataframe comparing trial 0 and trial 9 for each session ---
summary_data = []

for session in sorted(peak_cue_df["Session"].unique()):
    for trial in [0, 9]:
        data = peak_cue_df.loc[
            (peak_cue_df["Session"] == session) & (peak_cue_df["Trial"] == trial),
            "PeakHeight"
        ]
        if not data.empty:
            summary_data.append({
                "Session": session,
                "Trial": trial,
                "MeanPeakHeight": data.mean(),
                "SEMPeakHeight": sem(data)
            })

cue_trial_summary_df = pd.DataFrame(summary_data)
print(cue_trial_summary_df)

# --- Plot comparison (Trial 0 vs. Trial 9 per session) ---
plt.figure(figsize=(8,5))

sessions = cue_trial_summary_df["Session"].unique()
width = 0.35
x = range(len(sessions))

trial0_means = cue_trial_summary_df.loc[cue_trial_summary_df["Trial"] == 0, "MeanPeakHeight"]
trial9_means = cue_trial_summary_df.loc[cue_trial_summary_df["Trial"] == 9, "MeanPeakHeight"]
trial0_sems = cue_trial_summary_df.loc[cue_trial_summary_df["Trial"] == 0, "SEMPeakHeight"]
trial9_sems = cue_trial_summary_df.loc[cue_trial_summary_df["Trial"] == 9, "SEMPeakHeight"]

plt.bar(
    [s - width/2 for s in x], trial0_means, width, yerr=trial0_sems, label="Trial 0",
    capsize=4, color='skyblue', edgecolor='black'
)
plt.bar(
    [s + width/2 for s in x], trial9_means, width, yerr=trial9_sems, label="Trial 9",
    capsize=4, color='salmon', edgecolor='black'
)

plt.xticks(x, [f"Session {s}" for s in sessions])
plt.xlabel("Session")
plt.ylabel("Average Peak Height at Cue")
plt.title("Cue-evoked Peak Height: Trial 0 vs Trial 9 by Session")
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------------

results = []
for (mouse, session, trial), (_, trace, _) in avglevertrace_dict.items():
    if trial < 10:
        # Build time array
        time = np.linspace(timerange_cue[0], timerange_cue[1], len(trace))
        
        # Interval of interest
        start_time, end_time = -0.5, 0.5
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)

        # Slice trace
        trace_segment = trace[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        # Peak
        peakheight = np.max(trace_segment)
        peak_idx = np.argmax(trace_segment)
        peakheighttime = time_segment[peak_idx]

        # Store as dict row
        results.append({
            "Mouse": mouse,
            "Session": session,
            "Trial": trial,
            "X-Axis": peakheighttime,
            "PeakHeight": peakheight
        })

# Create DataFrame in one go
peak_lever_df = pd.DataFrame(results)

# ---- Aggregate by session and plot ----
plt.figure()
leverpeakheight = []

for session in range(totalsessions):
    session_data = peak_lever_df.loc[peak_lever_df["Session"] == session, "PeakHeight"]

    if not session_data.empty:
        leverpeakheight.append(session_data.values)
        mean_val = session_data.mean()
        err_val = sem(session_data)

        # scatter individual points
        plt.scatter([session]*len(session_data), session_data,
                    color=colors10[session], alpha=0.2)

        # mean + errorbar
        plt.scatter(session, mean_val, color=colors10[session])
        plt.errorbar(session, mean_val, yerr=err_val,
                     ecolor=colors10[session], capsize=3)

plt.xlabel('Session')
plt.ylabel('Peak Height At Lever')
plt.show()


#---------- peak height for first lick ----------
results = []

for (mouse, session, trial), (_, trace,_) in avgflicktrace1_dict.items():
    if trial < 10:
        # Build time array
        time = np.linspace(timerange_lick[0], timerange_lick[1], len(trace))
        
        # Interval of interest
        start_time, end_time = 0, 2
        start_idx = np.searchsorted(time, start_time)
        end_idx = np.searchsorted(time, end_time)

        # Slice trace
        trace_segment = trace[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]

        # Peak
        peakheight = np.max(trace_segment)
        peak_idx = np.argmax(trace_segment)
        peakheighttime = time_segment[peak_idx]

        # Store row
        results.append({
            "Mouse": mouse,
            "Session": session,
            "Trial": trial,
            "X-Axis": peakheighttime,
            "PeakHeight": peakheight
        })

# Build DataFrame once
peak_flick_df = pd.DataFrame(results)

# ---- Aggregate and plot ----
plt.figure()
flickpeakheight = []

for session in range(totalsessions):
    session_data = peak_flick_df.loc[peak_flick_df["Session"] == session, "PeakHeight"]

    if not session_data.empty:
        flickpeakheight.append(session_data.values)
        mean_val = session_data.mean()
        err_val = sem(session_data)

        # scatter individual trials
        plt.scatter([session]*len(session_data), session_data,
                    color=colors10[session], alpha=0.2)

        # mean + errorbar
        plt.scatter(session, mean_val, color=colors10[session])
        plt.errorbar(session, mean_val, yerr=err_val,
                     ecolor=colors10[session], capsize=3)

plt.xlabel('Session')
plt.ylabel('Peak Height At First Lick')
plt.show()


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


#-------------------------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------------------------
#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by latency to lick ####
        

colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)

sess_of_interest = [0,1]
end_time = 5.1
alltraces=[]
for i in sess_of_interest:
    bysession = {}
    for subj,session,trial in alltrialtrace_dict:
        if session == i and trial in range (10):
            timespace = np.linspace(active_time[0], active_time[1], len(alltrialtrace_dict[subj,i,trial][3]))
            trace = alltrialtrace_dict[subj,i,trial][3]
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
        end_time = 2
        
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
    auc_cue_df.at[count,'Trial'] = trial
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



### for lever press
auc_lever_dict = {}
for lever_mouse, lever_session, lever_trial in avglevertrace_dict:
    if cue_trial in range(10):
        time = np.linspace(timerange_lever[0], timerange_lever[1], 122)  #for timerange -2,10
        trace = avglevertrace_dict[lever_mouse, lever_session, lever_trial][1][:122]
        
        # Define the interval of interest
        start_time = -1
        end_time = 1
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_lever_dict[lever_mouse, lever_session, lever_trial] = area

auc_lever_df=pd.DataFrame()
count = 0
for mouse, session, trial in auc_lever_dict:
    auc_lever_df.at[count,'Session'] = session
    auc_lever_df.at[count,'Mouse'] = mouse
    auc_lever_df.at[count,'Trial'] = trial
    auc_lever_df.at[count,'AUC'] = auc_lever_dict[mouse, session, trial]
    count = count + 1


plt.figure()
for i in range (8):
    mean_session=[]
    for mouse, session, trial in auc_lever_dict:
        if session == i:
            mean_session.append(auc_lever_dict[mouse, session,trial])
            plt.scatter(x=session,y= auc_lever_dict[mouse, session, trial],color=colors10[i], alpha=0.05)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Area Under Curve')
plt.show()



### for first lick
auc_flick_dict = {}
for cue_mouse, cue_session, cue_trial in avgflicktrace1_dict:
    if cue_trial in range(10):
        time = np.linspace(timerange_lick[0], timerange_lick[1], 122)  #for timerange -2,10
        trace = avgflicktrace_dict[cue_mouse, cue_session, cue_trial][1][:122]
        
        # Define the interval of interest
        start_time = 0
        end_time = 2
        
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
    auc_flick_df.at[count,'Mouse'] = mouse
    auc_flick_df.at[count,'Trial'] = trial
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


##############################################################
# BOUT ANALYSIS 
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

