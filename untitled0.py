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

#folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/New/NewRig'
#mice=['6364','6365','6605','6361']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'
# mice = ['7098','7099','7107','7108']


folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/All/'
mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321', '7329']


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
        for i in range(0,10):
            fptrace_df[i]=zscore_cue[i]

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
            
            zb = np.mean(df.iloc[lever_baseline:lever_zero,2])
            zsd = np.std(df.iloc[lever_baseline:lever_zero,2])
            
            rawtrial = np.array(df.iloc[lever_baseline:lever_end,2])
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
            
            zscoretrial = [(x-zb)/zsd for x in sampletrial]
            zscore_lever.append(zscoretrial)
            
            # for key, value in baselinedict.items():
            #     if starttime > int(key) and starttime < int(key+cue_time):
            #         value = zb,zsd
            #         zscoretrial = [(x-zb)/zsd for x in sampletrial]
                    
            #         zscore_lever.append(zscoretrial)
            
            ### TRACES
            fptracelever_df = pd.DataFrame()
            for i in range(len(zscore_lever)):
                fptracelever_df[i]=zscore_lever[i]

            working_lst = []
            for i in range(len(fptracelever_df)):
                eachtrace = fptracelever_df.loc[i,:]
                working_lst.append(eachtrace)
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
                    
        zscore_flick = []
        
        for i in range(len(track_flicks)):
            flick_zero = round(track_flicks[i] * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs
            trial = np.array(df.iloc[flick_baseline:flick_end,2])
            
    
            zb = np.mean(df.iloc[flick_baseline:flick_zero,2])
            zsd = np.std(df.iloc[flick_baseline:flick_zero,2])
        
            ### DOWNSAMPLING
            sample_flick=[]
            for i in range(0, len(trial), N):
                sample_flick.append(np.mean(trial[i:i+N-1])) # This is the moving window mean
            
            newtrial = [(x-zb)/zsd for x in sample_flick]
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
       
        # Align traces to the start of each lick bout
        zscore_lickbout=[]
        
        for length, bout in sorted_lick_bouts:
            start_time = bout[0]
            lickbout_each = []
            for items in bout:
                licklick = items-start_time
                lickbout_each.append(licklick)
            lick_rasterdata[mouse,length]=lickbout_each
            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero + timerange_lick[0] * fs
            lickb_end = lickb_zero + timerange_lick[1] * fs
            rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,2])
            #plt.plot(np.arange(0,len(rawtrial)), rawtrial)
            
            zb = np.mean(df.iloc[lickb_baseline:lickb_zero,2])
            zsd = np.std(df.iloc[lickb_baseline:lickb_zero,2])
            
            sampletrial=[]
            for i in range(0, len(rawtrial), N):
                sampletrial.append(np.mean(rawtrial[i:i+N-1]))
            
            
            # newtrial = [(x-zb)/zsd for x in sampletrial]
            # zscore_lickbout.append(newtrial)
            # bout_traces[mouse, length]=(zscoretrial)
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



##############################################################
######################## BOUT ANALYSIS #######################
##############################################################

#RASTER PLOT
lickbouts_rasterplot={}
for key, licks in lick_rasterdata.items():
    mouse, licklength = key
    if licklength not in lickbouts_rasterplot:
        lickbouts_rasterplot[licklength] = []
    lickbouts_rasterplot[licklength].append(licks)


tinybouts = []
smallbouts = []
mediumbouts = []
longbouts = []
colors = sns.color_palette("husl", 8)

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

print(len(tinybouts), position)
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
    mouse, licklength = key
    offset_t=find_first_above_zero(timeline, trace, 
                          ((offset_range[0]-timerange_lick[0])), 
                          ((offset_range[1]-timerange_lick[0])))
    offset_bouts[mouse,licklength]=offset_t
    
    if licklength > 10 and licklength < 30 :
        smallbouts.append(trace)
        peakmax = max(trace[round(-timerange_lick[0]*fs/N):round((peakheight_range[1]-timerange_lick[0])*fs/N)])
        peakheight_bouts[mouse, licklength]=peakmax
    elif licklength <10:
        tinybouts.append(trace)
    elif licklength > 19 and licklength < 50:
        mediumbouts.append(trace)
        peakmax = max(trace[round(-timerange_lick[0]*fs/N):round((peakheight_range[1]-timerange_lick[0])*fs/N)])
        peakheight_bouts[mouse,licklength]=peakmax
    elif licklength > 49 and licklength <70:
        longbouts.append(trace)
        peakmax = max(trace[round(-timerange_lick[0]*fs/N):round((peakheight_range[1]-timerange_lick[0])*fs/N)])
        peakheight_bouts[mouse,licklength]=peakmax
    elif licklength > 70:
        longlongbouts.append(trace)
        peakmax = max(trace[round(-timerange_lick[0]*fs/N):round((peakheight_range[1]-timerange_lick[0])*fs/N)])
        peakheight_bouts[mouse,licklength]=peakmax
    
mean_tinybouts = np.mean(tinybouts, axis=0)
sem_tinybouts = sem(tinybouts)
mean_smallbouts = np.mean(smallbouts, axis=0)
sem_smallbouts = sem(smallbouts)
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
plt.xlabel('Lick Bout Onset (s)')
plt.savefig('/Users/kristineyoon/Documents/bylickbouts.pdf', transparent=True)


############################################################
pkht_bout_df=pd.DataFrame()
lengths=[]
peakheights=[]
for key, value in peakheight_bouts.items():
    mouse, length = key
    lengths.append(length)
    peakheights.append(value)
pkht_bout_df['Lengths']= lengths
pkht_bout_df['Heights']= peakheights


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


# ############################################################
# def legend_without_duplicate_labels(figure):
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     figure.legend(by_label.values(), by_label.keys(), loc='lower right')

# plt.figure(figsize=(8,4))
# for i in range(len(smallbouts)):
#     plt.plot(np.arange(0,len(smallbouts[i])), smallbouts[i], color='tab:blue', label = '10-29 Licks',  alpha=0.6)
# for i in range(len(mediumbouts)):
#     plt.plot(np.arange(0,len(mediumbouts[i])), mediumbouts[i], color='tab:orange', label = '30-49 Licks',  alpha=0.6)
# for i in range(len(longbouts)):
#     plt.plot(np.arange(0,len(longbouts[i])), longbouts[i], color='tab:green', label = '50-69 Licks', alpha=0.6)    
# for i in range(len(longlongbouts)):
#     plt.plot(np.arange(0,len(longlongbouts[i])), longlongbouts[i], color='tab:red', label = '70+ Licks', alpha=0.6)    
# plt.xticks(np.arange(0,len(longbouts[0])+1,len(longbouts[0])/(timerange_lick[1]-timerange_lick[0])), 
#            np.arange(timerange_lick[0], timerange_lick[1]+1,1, dtype=int),
#            rotation=0)
# plt.axvline(x=len(mean_smallbouts)/(timerange_lick[1]-timerange_lick[0])*(0-timerange_lick[0]),linewidth=1, color='black')
# plt.axhline(y=0, linestyle=':', color='black')
# legend_without_duplicate_labels(plt)
# plt.xlabel('Lick Bout Onset (s)')
# #plt.savefig('/Users/kristineyoon/Documents/cue.pdf', transparent=True)



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
plt.xticks(np.arange(0,len(zscore_cue[0])+1,len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])), 
           np.arange(timerange_cue[0], timerange_cue[1]+1,1, dtype=int),
           rotation=0)
plt.axvline(x=len(zscore_cue[0])/(timerange_cue[1]-timerange_cue[0])*(0-timerange_cue[0]),linewidth=1, color='black')
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
