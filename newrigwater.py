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

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/New/Water'
#mice=['6364','6365','6605','6361']
#mice = ['7098','7099','7107','7108']
mice = ['6361', '6364', '6605']
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
timerange_cue = [-2, 5]
timerange_lever = [-2, 10]
timerange_lick = [-2, 10]
cue_time = 20
lick_time= 10
N = 100

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

        # plt.figure(figsize=(12, 6))
        # totaltrace = np.array(df.iloc[:,2])
        # totaltracemean = np.mean(totaltrace)
        # totaltracestddev = np.std(totaltrace)
        # totaltracezscore = (totaltrace-totaltracemean)/totaltracestddev
        # multiplied_track_licks = [item * fs for item in track_licks]
        # multiplied_track_lever = [item * fs for item in track_lever]
        # multiplied_track_cue = [item * fs for item in track_cue]
        # multiplied_track_leverend = [item + 20*fs for item in multiplied_track_lever]
        
        
        # plt.plot(totaltracezscore)
        # plt.eventplot(multiplied_track_licks, colors='red', lineoffsets=0.5, label='lick')
        # plt.eventplot(multiplied_track_lever, colors='green', lineoffsets=0.5, label='leverpress/ sipper out')
        # plt.eventplot(multiplied_track_leverend, colors='blue', lineoffsets=0.5, label='end session')
        # plt.eventplot(multiplied_track_cue, colors='pink', lineoffsets=0.5, label='cue')
        # plt.suptitle(f'{mouse} and {Dates.index(date)}')
        # plt.legend()

########################## CUE ALIGNMENT ##########################    
        zscore_cue = []
        baselinedict={}
        for i in range(len(track_cue)):
            cue_zero = round(track_cue[i] * fs)
            cue_baseline = cue_zero + timerange_cue[0] * fs
            cue_end = cue_zero + timerange_cue[1] * fs
            
            zb = np.mean(df.iloc[cue_baseline:cue_zero,2])
            zsd = np.std(df.iloc[cue_baseline:cue_zero,2])
            baselinedict[track_cue[i]]=zb,zsd
            
            rawtrial = np.array(df.iloc[cue_baseline:cue_end,2])
            
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
                
            zscoretrial = [(x-zb)/zsd for x in sampletrial]
            zscore_cue.append(zscoretrial)

        ### TRACES
        fptrace_df = pd.DataFrame()
        for i in range(len(track_cue)):
            fptrace_df[i]=zscore_cue[i]

        working_lst = []
        for i in range(len(fptrace_df)):
            avg_at_time = np.mean(fptrace_df.loc[i,:])
            working_lst.append(avg_at_time)
        avgcuetrace_dict[mouse,Dates.index(date)]=working_lst

        
        ############ LICKBOUTS #################
        # Identify and sort lick bouts by length
        sorted_lick_bouts = identify_and_sort_lick_bouts(track_licks, max_interval)
       
        # Align traces to the start of each lick bout
        zscore_lickbout=[]
        for length, bout in sorted_lick_bouts:
            start_time = bout[0]
            print(start_time)
            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero + timerange_lick[0] * fs
            lickb_end = lickb_zero + timerange_lick[1] * fs
            rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,2])
            #plt.plot(np.arange(0,len(rawtrial)), rawtrial)
            
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
            
            for key, value in baselinedict.items():
                if start_time > int(key) and start_time < int(key+cue_time+lick_time):
                    value = zb,zsd
                    zscoretrial = [(x-zb)/zsd for x in sampletrial]
                    zscore_lickbout.append(zscoretrial)
                    plt.plot(np.arange(0,len(zscoretrial)), zscoretrial)
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
longlongbouts = []
for key, trace in bout_traces.items():
    mouse, licklength = key
    if licklength > 9 and licklength < 30 :
        smallbouts.append(trace)
    elif licklength > 29 and licklength < 50:
        mediumbouts.append(trace)
    elif licklength > 49 and licklength <70:
        longbouts.append(trace)
    elif licklength > 69:
        longlongbouts.append(trace)
mean_smallbouts = np.mean(smallbouts, axis=0)
sem_smallbouts = sem(smallbouts)
mean_mediumbouts = np.mean(mediumbouts, axis=0)
sem_mediumbouts= sem(mediumbouts)
mean_longbouts = np.mean(longbouts, axis=0)
sem_longbouts = sem(longbouts)
mean_longlongbouts = np.mean(longlongbouts, axis=0)
sem_longlongbouts = sem(longlongbouts)
plt.figure(figsize=(8,4))
plt.plot(np.arange(0,len(mean_smallbouts)), mean_smallbouts, label = '1WaterBouts', color='slateblue')
plt.fill_between(range(len(mean_smallbouts)), mean_smallbouts -sem_smallbouts , mean_smallbouts + sem_smallbouts, color='slateblue', alpha=0.1)
# plt.plot(np.arange(0,len(mean_mediumbouts)),mean_mediumbouts, label = '30-49 Licks')
# plt.fill_between(range(len(mean_mediumbouts)), mean_mediumbouts -sem_mediumbouts , mean_mediumbouts + sem_mediumbouts, alpha=0.1)
# plt.plot(np.arange(0,len(mean_longbouts)),mean_longbouts, label = '50-69 Licks')
# plt.fill_between(range(len(mean_longbouts)), mean_longbouts -sem_longbouts , mean_longbouts + sem_longbouts, alpha=0.1)
# plt.plot(np.arange(0,len(mean_longlongbouts)),mean_longlongbouts, label = '70+ Licks')
# plt.fill_between(range(len(mean_longlongbouts)), mean_longlongbouts -sem_longlongbouts , mean_longlongbouts + sem_longlongbouts, alpha=0.1)
plt.xticks(np.arange(0,len(smallbouts[0])+1,len(smallbouts[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.legend()
plt.xlabel('Lick Bout Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/waterbouts.pdf', transparent=True)


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='lower right')

plt.figure(figsize=(8,4))
for i in range(len(smallbouts)):
    plt.plot(np.arange(0,len(smallbouts[i])), smallbouts[i], color='tab:blue', label = '10-29 Licks',  alpha=0.6)
# for i in range(len(mediumbouts)):
#     plt.plot(np.arange(0,len(mediumbouts[i])), mediumbouts[i], color='tab:orange', label = '30-49 Licks',  alpha=0.6)
# for i in range(len(longbouts)):
#     plt.plot(np.arange(0,len(longbouts[i])), longbouts[i], color='tab:green', label = '50-69 Licks', alpha=0.6)    
# for i in range(len(longlongbouts)):
#     plt.plot(np.arange(0,len(longlongbouts[i])), longlongbouts[i], color='tab:red', label = '70+ Licks', alpha=0.6)    
plt.xticks(np.arange(0,len(smallbouts[0])+1,len(smallbouts[0])/(timerange_lick[1]-timerange_lick[0])), 
           np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
plt.axhline(y=0, linestyle=':', color='black')
legend_without_duplicate_labels(plt)
plt.xlabel('Lick Bout Onset (s)')