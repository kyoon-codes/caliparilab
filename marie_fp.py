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
max_interval = 1.0
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

#folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/New/NewRig'
#mice=['6364','6365','6605','6361']

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_FirstBinge'
mice = ['A204']
mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212']

files = os.listdir(folder)
files.sort()
print(files)

avgcuetrace_dict = {}
avgresptrace_dict = {}
avglevertrace_dict = {}
avgflicktrace_dict = {}
avglickbouttrace_dict = {}
lickbouttracedict ={}      
bout_traces = {}
bout_peak_heights = {}
bout_peak_indices = {}
timerange = [-5, 10]
cue_time = 20
lick_time= 10
N = 100

auc_lickbout = {}

for mouse in mice:
    mouse_dir = os.path.join(folder, mouse)
    
    # Dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
    # Dates.sort()
    # for date in Dates:
    #     date_dir = os.path.join(mouse_dir, date)
    data = tdt.read_block(mouse_dir)
    print(mouse_dir)
    df = pd.DataFrame()
    df['Sig470'] = data.streams._470A.data
    df['Dff'] = ((df['Sig470']-np.mean(df['Sig470']))/np.mean(df['Sig470']))
    fs = round(data.streams._470A.fs)

    split1 = str(data.epocs).split('\t')
    y = []
    for elements in split1:
        x = elements.split('\n')
        if '[struct]' in x:
            x.remove('[struct]')
        y.append(x)
    z= [item for sublist in y for item in sublist]

    fp_df = pd.DataFrame(columns=['Event','Timestamp'])
    events = ['Lick', 'LeftLever', 'RightLever', 'SipperExt']
    epocs = ['Lik_','LLr_','RLr_','Sir_']
    
    for a, b in zip(events, epocs):
        if b in z:
            event_df = pd.DataFrame(columns=['Event','Timestamp'])
            event_df['Timestamp'] = data.epocs[b].onset
            event_df['Event'] = a
            fp_df = pd.concat([fp_df, event_df])
    
    track_lick = []
    track_leftlever = []
    track_rightlever = []
    track_sipperext = []
    latency= []
    leverpermice = []
    for i in range(len(fp_df)):
        if fp_df.iloc[i,0] == 'Lick':
            track_lick.append(fp_df.iloc[i,1])
        if fp_df.iloc[i,0] == 'LeftLever':
            track_leftlever.append(fp_df.iloc[i,1])
        if fp_df.iloc[i,0] == 'RightLever':
            track_rightlever.append(fp_df.iloc[i,1])
        if fp_df.iloc[i,0] == 'SipperExt':
            track_sipperext.append(fp_df.iloc[i,1])

                
    # # LOOKING AT THE FULL TRACE
    # plt.figure(figsize=(12, 6))
    # totaltrace = np.array(df.iloc[:,1])
    # multiplied_track_licks = [item * fs for item in track_lick]
    # multiplied_track_lever = [item * fs for item in track_leftlever]
    # multiplied_track_cue = [item * fs for item in track_rightlever]
    # multiplied_track_leverend = [item *fs for item in track_sipperext]
    
    
    # plt.plot(totaltrace)
    # plt.eventplot(multiplied_track_licks, colors='red', lineoffsets=0, label='track_lick')
    # plt.eventplot(multiplied_track_lever, colors='green', lineoffsets=0, label='track_leftlever')
    # plt.eventplot(multiplied_track_cue, colors='blue', lineoffsets=0, label='end track_rightlever')
    # plt.eventplot(multiplied_track_leverend, colors='pink', lineoffsets=0, label='track_sipperext')
    # plt.suptitle(f'MOUSE: {mouse}')
    # plt.legend()
     ################## FIRST LICK ALIGNMENT ################3
    track_flicks = []
    firstlicks = []
    baselinedict={}
    for extension in track_sipperext:
        lickyes = np.array(track_lick) > extension
        firstlicktime = np.where(lickyes == True)
        if len(firstlicktime[0]) > 0:
            firstlicks.append(firstlicktime[0][0])
    firstlicks = list(set(firstlicks))
    for index in firstlicks:
        track_flicks.append(track_lick[index])
                
    align_flick = []
    zscore_flick = []
    
    for i in range(len(track_flicks)):
        flick_zero = round(track_flicks[i] * fs)
        baseline_neg10 = round(track_flicks[i] + -10 * fs)
        baseline_neg5 = round(track_flicks[i] + -5 * fs)
        lick_beg = flick_zero + timerange[0] * fs
        lick_end = flick_zero + timerange[1] * fs
        
        zb = np.mean(df.iloc[baseline_neg10:baseline_neg5,1])
        zsd = np.std(df.iloc[baseline_neg10:baseline_neg5,1])
        baselinedict[track_flicks[i]]=zb,zsd
        trial = np.array(df.iloc[lick_beg:lick_end,1])
        align_flick.append(trial)
       
        newtrial = [(x-zb)/zsd for x in trial]
        zscore_flick.append(newtrial)
    
    ### TRACES
    fptrace_df = pd.DataFrame()
    if len(zscore_flick) > 0:
        for i in range(len(zscore_flick)):
            fptrace_df[i]=zscore_flick[i]
    
        
    #### LICK ANALYSIS ####
    plt.figure(figsize=(8,4))
    for i in range(len(zscore_flick)):
        plt.plot(zscore_flick[i],label=i)
        plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange[1]-timerange[0])), 
               np.arange(timerange[0], timerange[1]+1,1, dtype=int),
               rotation=0)
    plt.axvline(x=len(zscore_flick[0])/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=1, color='black')
    plt.axhline(y=0, linestyle=':', color='black')
    plt.legend()
    plt.xlabel('First Lick Onset (s)')
    plt.title(mouse)



########## 여기까지 다 했습니다 #############










        working_lst = []
        for i in range(len(fptrace_df)):
            avg_at_time = np.mean(fptrace_df.loc[i,:])
            working_lst.append(avg_at_time)
        avgflicktrace_dict[mouse,Dates.index(date)]=working_lst
 
########################## CUE ALIGNMENT ##########################    
        zscore_lick = []
        baselinedict={}
        for i in range(len(track_lick)):
            lick_zero = round(track_lick[i] * fs)
            baseline_neg10 = lick_zero + -10 * fs
            baseline_neg5 = lick_zero + -5 * fs
            lick_beg = lick_zero + timerange[0] * fs
            lick_end = lick_zero + timerange[1] * fs
            
            zb = np.mean(df.iloc[baseline_neg10:baseline_neg5,2])
            zsd = np.std(df.iloc[baseline_neg10:baseline_neg5,2])
            baselinedict[track_lick[i]]=zb,zsd
            
            rawtrial = np.array(df.iloc[lick_beg:lick_beg,2])
            
            # No moving average for Marie's data
            # sampletrial=[]
            # for i in range(0, len(rawtrial), N):
            #     sampletrial.append(np.mean(rawtrial[i:i+N-1]))
                
            zscoretrial = [(x-zb)/zsd for x in rawtrial]
            zscore_lick.append(zscoretrial)

        ### TRACES
        fptrace_df = pd.DataFrame()
        for i in range(0,len(track_sipperext)):
            fptrace_df[i]=zscore_lick[i]

        working_lst = []
        for i in range(len(fptrace_df)):
            avg_at_time = np.mean(fptrace_df.loc[i,:])
            working_lst.append(avg_at_time)
        avgcuetrace_dict[mouse,Dates.index(date)]=working_lst

        
        ########################## LEVER ALIGNMENT ##########################
        zscore_lever = []
        
        for i in range(len(track_lever)):
            starttime= track_lever[i]
            lever_zero = round(track_lever[i] * fs)
            lever_baseline = lever_zero + timerange_lever[0] * fs
            lever_end = lever_zero + timerange_lever[1] * fs
            
            rawtrial = np.array(df.iloc[lever_baseline:lever_end,2])
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
            
            
            for key, value in baselinedict.items():
                if starttime > int(key) and starttime < int(key+cue_time):
                    value = zb,zsd
                    zscoretrial = [(x-zb)/zsd for x in sampletrial]
                    
                    zscore_lever.append(zscoretrial)
            
            ### TRACES
            fptracelever_df = pd.DataFrame()
            for i in range(len(zscore_lever)):
                fptracelever_df[i]=zscore_lever[i]

            working_lst = []
            for i in range(len(fptracelever_df)):
                avg_at_time = np.mean(fptracelever_df.loc[i,:])
                working_lst.append(avg_at_time)
            avglevertrace_dict[mouse,Dates.index(date)]=working_lst

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

        cuestolicks={}
        for i in range(len(track_cue)):
            for k in range(len(track_flicks)):
                if track_flicks[k]- track_cue[i] <= (cue_time + lick_time) and track_flicks[k]- track_cue[i]>0:
                    cuestolicks[track_cue[i]]=track_flicks[k]
                    
        align_flick = []
        
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
        for trace in sample_flick:
            newtrial = [(x-zb)/zsd for x in trace]
            zscore_flick.append(newtrial)
        
        ### TRACES
        fptrace_df = pd.DataFrame()
        if len(zscore_flick) > 0:
            for i in range(len(zscore_flick)):
                fptrace_df[i]=zscore_flick[i]
    
            working_lst = []
            for i in range(len(fptrace_df)):
                avg_at_time = np.mean(fptrace_df.loc[i,:])
                working_lst.append(avg_at_time)
            avgflicktrace_dict[mouse,Dates.index(date)]=working_lst
        
        
        ############ LICKBOUTS #################
        # Identify and sort lick bouts by length
        sorted_lick_bouts = identify_and_sort_lick_bouts(track_licks, max_interval)
        sorted2_lick_bouts=[]
        for order, value in sorted_lick_bouts:
            if order > 5:
                sorted2_lick_bouts.append(value)
        track_lickbouts=[]
        for lickbouts in sorted2_lick_bouts:
            track_lickbouts.append(lickbouts[0])
            
        # Align traces to the start of each lick bout
        zscore_lickbout=[]
        for length, bout in sorted_lick_bouts:
            start_time = bout[0]
            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero - round(timerange_lick[0] * fs)
            lickb_end = lickb_zero + round(10 * fs)
            rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,2])
            
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
            
            for key, value in baselinedict.items():
                if start_time > int(key) and start_time < int(key+cue_time+lick_time):
                    value = zb,zsd
                    zscoretrial = [(x-zb)/zsd for x in sampletrial]
                    zscore_lickbout.append(zscoretrial)
                    bout_traces[mouse, length]=(zscoretrial)
                        
            #area under the curve
            after_zero = 1
            start_time = round(-timerange_lick[0] * fs/N)
            end_time = round((-timerange_lick[0] + after_zero) * fs/N)
            area = np.sum(zscoretrial[start_time:end_time])
            if length not in auc_lickbout:
                auc_lickbout[length] = []
            auc_lickbout[length].append(area)

        ### TRACES
        fptracelickbout_df = pd.DataFrame()
        for i in range(len(zscore_lickbout)):
            fptracelickbout_df[i]=zscore_lickbout[i]
            

############################################################
############################################################
############################################################
############################################################
############################################################
smallbouts = []
mediumbouts = []
longbouts = []
for key, trace in bout_traces.items():
    mouse, licklength = key
    if licklength > 9 and licklength < 30 :
        smallbouts.append(trace)
    elif licklength > 29 and licklength < 50:
        mediumbouts.append(trace)
    elif licklength > 49:
        longbouts.append(trace)
mean_smallbouts = np.mean(smallbouts, axis=0)
sem_smallbouts = sem(smallbouts)
mean_mediumbouts = np.mean(mediumbouts, axis=0)
sem_mediumbouts= sem(mediumbouts)
mean_longbouts = np.mean(longbouts, axis=0)
sem_longbouts = sem(longbouts)
plt.plot(mean_smallbouts, label = '10-29 Licks')
plt.fill_between(range(len(mean_smallbouts)), mean_smallbouts -sem_smallbouts , mean_smallbouts + sem_smallbouts, alpha=0.1)
plt.plot(mean_mediumbouts, label = '30-49 Licks')
plt.fill_between(range(len(mean_mediumbouts)), mean_mediumbouts -sem_mediumbouts , mean_mediumbouts + sem_mediumbouts, alpha=0.1)

plt.plot(mean_longbouts, label = '50+ Licks')
plt.fill_between(range(len(mean_longbouts)), mean_longbouts -sem_longbouts , mean_longbouts + sem_longbouts, alpha=0.1)

plt.xticks(np.arange(0,len(mean_smallbouts)+1,len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Lick Bout Onset (s)')
#plt.savefig('/Users/kristineyoon/Documents/cue.pdf', transparent=True)

#### CUE ANALYSIS ####
#by session individual
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgcuetrace_dict:
        if session == i:
            plt.plot(avgcuetrace_dict[str(mouse),session],color=sns.color_palette("light:teal")[i],label=i)
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Cue Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/cue.pdf', transparent=True)

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
    if session in [0,4,5,6,7]:
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
    if session in [0,3,6]:
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


#### LEVER ANALYSIS ####
#by session individual
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avglevertrace_dict:
        if session in [0,4,5,6,7]:
            plt.plot(avglevertrace_dict[str(mouse),session],color=sns.color_palette("husl", 8)[int(session)],label=i)
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Lever Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/cue.pdf', transparent=True)


#area under the curve
after_zero = 1
start_time = round(len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]))
end_time = round(len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(after_zero-timerange_cue[0]))
area_under_curve = {}

for key, value in avgcuetrace_dict.items():
    mouse, session = key
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

############LICKS
#by session individual
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgflicktrace_dict:
        if session == i:
            plt.plot(avgflicktrace_dict[str(mouse),session],color=sns.color_palette("light:deeppink")[i],label=i)
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('First Lick Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/lick.pdf', transparent=True)

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
    if session in [0,3,6]:
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


#area under the curve
after_zero = 1
start_time = round(len(mean_traces[0])/(timerange_lick[1]-timerange_cue[0])*(0-timerange_cue[0]))
end_time = round(len(mean_traces[0])/(timerange_cue[1]-timerange_cue[0])*(after_zero-timerange_cue[0]))
area_under_curve = {}

for key, value in avgcuetrace_dict.items():
    mouse, session = key
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



for key, value in lickbouttracedict.items():
    mouse, session, trial = key
    if trial == 0:
        plt.plot(lickbouttracedict[mouse, session, trial])

#by session average lick bouts
session_lickdata = {}  
for key, value in bout_traces.items():
    if key not in session_lickdata:
        session_lickdata[key] = []
    session_lickdata[key].append(value)
mean_licktraces = {}
sem_licktraces = {}
for session, traces in session_lickdata.items():
    mean_licktraces[session] = np.mean(traces, axis=0)
    sem_licktraces[session] = sem(traces)
plt.figure(figsize=(10, 6))
for session, mean_licktraces in mean_licktraces.items():
    sem_licktrace = sem_licktraces[session]
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



#### LEVER ANALYSIS ####
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avglevertrace_dict:
        if session == i:
            plt.plot(avglevertrace_dict[str(mouse),session],color=sns.color_palette("light:orange")[i],label=i)
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
           np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Lever Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/lever.pdf', transparent=True)

#### LICK ANALYSIS ####
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgflicktrace_dict:
        if session == i:
            plt.plot(avgflicktrace_dict[str(mouse),session],color=sns.color_palette("light:deeppink")[i],label=i)
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('First Lick Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/lick.pdf', transparent=True)

#### RESPONSIVE ANALYSIS ####
#by session
plt.figure(figsize=(8,4))
for i in range(10):
    for mouse,session in avgresptrace_dict:
        if session == i:
            plt.plot(avgresptrace_dict[str(mouse),session],color=sns.color_palette("light:limegreen")[i],label=i)
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Response Cue Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/response.pdf', transparent=True)



# ######################## TO ALIGNMENT ##########################
# align_to = []
# timerange_to = [-2, 10]
# for i in range(len(track_to)):
#     lever_zero = round(track_to[i] * fs)
#     lever_baseline = lever_zero + timerange_to[0] * fs
#     lever_end = lever_zero + timerange_to[1] * fs
#     trial = np.array(df.iloc[lever_baseline:lever_end,2])
#     align_to.append(trial)

# ### DOWNSAMPLING
# sample_to=[]
# for lst in align_to: 
#     small_lst = []
#     for i in range(0, len(trial), N):
#         small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
#     sample_to.append(small_lst)
    
# zscore_to = []
# baselinedict = {}
# for i in range(len(sample_to)):
#     trial = sample_to[i]
#     zb = np.mean(trial[0:round((-timerange_to[0]*fs/N))])
#     zsd = np.std(trial[0:round((-timerange_to[0]*fs/N))])
#     baselinedict[track_to[i]] = zb, zsd
#     trial = (trial - zb)/zsd
#     zscore_to.append(trial)

# #### ALIGN TO LEVER PRESS: AVERAGE LINE PLOT OF TRIALS ####
# for i in range(len(zscore_to)):
# plt.plot((zscore_to[0]), color=sns.color_palette("light:navy")[3])
# plt.xticks(np.arange(0,len(zscore_to[0])+1,len(zscore_to[0])/(timerange_to[1]-timerange_to[0])), 
#            np.arange(timerange_to[0], timerange_to[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(zscore_to[0])/(timerange_to[1]-timerange_to[0])*(0-timerange_to[0]),linewidth=1, color='black')
# plt.axhline(y=0, linestyle=':', color='black')
# plt.xlabel('Lever Press Onset (s)')

# #by mouse
# for i in mice:
#     for mouse,session in avgcuetrace_dict:
# plt.figure(figsize=(8,4))
# for 
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


# #PEAK HEIGHT IN THE FIRST SECOND
# calc_range = 1
# peakheight_df = pd.DataFrame()
# for i in range(0,10):
#     peakheight = max(fptrace_df[i][round((-timerange_cue[0])*fs/N):round((calc_range-timerange_cue[0])*fs/N)])
#     peakheight_df.loc[0,i]=peakheight




# #### GRAPHING ####
# plt.figure(figsize=(8,4))
# plt.plot(np.mean(respondingcue, axis=0), color='limegreen')
# plt.fill_between(np.arange(0,len(respondingcue[0]),1),
#                 np.mean(respondingcue, axis=0)+np.std(respondingcue, axis=0), 
#                 np.mean(respondingcue, axis=0)-np.std(respondingcue, axis=0),
#                 facecolor='limegreen',
#                 alpha=0.2)
# plt.xticks(np.arange(0,len(respondingcue[0])+1,len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])), 
#            np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(respondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
# plt.axhline(y=0, linestyle=':', color='black')
# plt.xlabel('Cue Onset (s)')
# plt.ylim(-2,5)
# plt.tight_layout()
# plt.savefig('/Users/kristineyoon/Documents/response.pdf', transparent=True)



# #### GRAPHING ####
# plt.figure(figsize=(8,3))
# plt.plot(np.mean(nonrespondingcue, axis=0), color='limegreen')
# plt.fill_between(np.arange(0,len(nonrespondingcue[0]),1),
#                 np.mean(nonrespondingcue, axis=0)+np.std(nonrespondingcue, axis=0), 
#                 np.mean(nonrespondingcue, axis=0)-np.std(nonrespondingcue, axis=0),
#                 facecolor='limegreen',
#                 alpha=0.2)
# plt.xticks(np.arange(0,len(nonrespondingcue[0])+1,len(nonrespondingcue[0])/(timerange_cue[1]-timerange_cue[0])), 
#            np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(nonrespondingcue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black', label='Cue Onset')
# plt.axhline(y=0, linestyle=':', color='black')
# plt.xlabel('Cue Onset (s)')
# plt.ylim(-2,5)
# plt.tight_layout()
# plt.savefig('/Users/kristineyoon/Documents/noresponse.pdf', transparent=True)

# ########################## LEVER PRESS ALIGNMENT ##########################
# align_lever = []
# timerange_lever = [-2, 5]
# for i in range(len(track_lever)):
#     lever_zero = round(track_lever[i] * fs)
#     lever_baseline = lever_zero + timerange_lever[0] * fs
#     lever_end = lever_zero + timerange_lever[1] * fs
#     trial = np.array(df.iloc[lever_baseline:lever_end,2])
#     align_lever.append(trial)


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

for i in range(len(sample_cue)):
    trial = sample_cue[i]
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
sns.heatmap(zscore_lever, cmap='Blues', vmin=-5, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
plt.xticks(np.arange(0,len(zscore_lever[0])+1,len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])), 
            np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
            rotation=0)
plt.axvline(x=len(zscore_lever[0])/(timerange_lever[1]-timerange_lever[0])*(0-timerange_lever[0]),linewidth=1, color='black', label='Lever Onset')
plt.ylabel('Trials')
plt.xlabel('Lever Press Onset')

#### TRIALS ON HEATMAP ALIGNED TO LEVERPRESS ####

def sort_traces_by_average_desc(traces):
    # Calculate the average of each trace
    averages = [(trace, sum(trace)/len(trace)) for trace in traces]
    
    # Sort the list of pairs (trace, average) by average value in descending order
    sorted_averages = sorted(averages, key=lambda x: x[1], reverse=True)
    
    # Extract the sorted traces
    sorted_traces = [trace for trace, avg in sorted_averages]
    
    return sorted_traces

                  
plt.figure(figsize=(5,3))
allflicktraces=[]

for key, value in avgflicktrace_dict.items():
    mouse, session = key
    if session in [4,5,6,7]:
        allflicktraces.append(value)
allflicktraces= sort_traces_by_average_desc(allflicktraces)
sns.heatmap(allflicktraces, cmap=sns.color_palette("light:firebrick", as_cmap=True), vmin=-2, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
#sns.heatmap(allflicktraces2, cmap='RdBu', vmin=-5, vmax=5, cbar_kws={'label': 'Delta F/F From Baseline'})
plt.xticks(np.arange(0,len(zscore_flick[0])+1,len(zscore_flick[0])/(timerange_lever[1]-timerange_lever[0])), 
            np.arange(timerange_lever[0], timerange_lever[1]+1,1, dtype=int),
            rotation=0)
plt.axvline(x=len(zscore_flick[0])/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black', label='Lever Onset')
plt.ylabel('Trials')
plt.xlabel('Lick Onset')

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
