import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy


# Define function to find lick bouts

def identify_and_sort_lick_bouts(lick_timestamps, min_licks=3, start_interval=1.0, end_interval=3.0):
    bouts = []              # To store the detected bouts
    current_bout = []       # Temporary list for building each bout

    for i in range(len(lick_timestamps)):
        # Add current timestamp to the current bout
        current_bout.append(lick_timestamps[i])

        # Check if we have enough licks within the start interval to start a bout
        if len(current_bout) >= min_licks:
            if current_bout[-1] - current_bout[-min_licks] <= start_interval:
                # Bout is confirmed; check for continuation or end
                if i + 1 < len(lick_timestamps):
                    # If next lick is more than end_interval away, end the bout
                    if lick_timestamps[i + 1] - lick_timestamps[i] > end_interval:
                        bouts.append(current_bout)
                        current_bout = []
            else:
                # Clear current_bout if we haven't met the start condition
                current_bout = [lick_timestamps[i]]

    # Append the last bout if it exists and is valid
    if current_bout and len(current_bout) >= min_licks:
        bouts.append(current_bout)

    # Calculate length of each bout and sort by descending length
    bout_lengths = [(len(bout), bout) for bout in bouts]
    sorted_bouts = sorted(bout_lengths, key=lambda x: x[0], reverse=True)

    return sorted_bouts

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/First_EtOH_Binge/'
# mice = [ 'A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A409', 'A410', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Last_EtOH_Binge/'
# mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A410']

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Sacchrin/'
mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_0.25Q/'
# mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_1Q/'
# mice = ['A201', 'A204', 'A205', 'A207', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_EtOH_Only/'
# mice = ['A201', 'A204', 'A205', 'A207', 'A212', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Withdrawal_D1'
# mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A414']

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Withdrawal_D28'
# mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

files = os.listdir(folder)
files.sort()
print(files)

avglevertrace_dict = {}
avglickbouttrace_dict = {}
lickbouttracedict ={}      
lick_rasterdata = {}
sipperext_to_lick_dict = {}
bout_traces = {}
timerange = [-10, 10]
N = 100


############################################################################################################################
mice_high = ['A201','A204','A209','A212','A414']
mice_low = ['A205','A216','A402','A409','A410']
mice_comp =  ['A207','A214','A215','A403']

activelever_left = ['A201','A204','A207','A209','A212','A214','A215','A403','A410']
activelever_right =  ['A205','A216','A402','A404','A409','A414']
############################################################################################################################

for mouse in mice:
    mouse_dir = os.path.join(folder, mouse)

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

    #LOOKING AT THE FULL TRACE
    plt.figure(figsize=(12, 6))
    totaltrace = np.array(df.iloc[:,1])
    multiplied_track_licks = [item * fs for item in track_lick]
    multiplied_track_lever = [item * fs for item in track_leftlever]
    multiplied_track_cue = [item * fs for item in track_rightlever]
    multiplied_track_leverend = [item *fs for item in track_sipperext]
    
    
    plt.plot(totaltrace)
    plt.eventplot(multiplied_track_licks, colors='red', lineoffsets=0, label='track_lick', alpha=0.5)
    plt.eventplot(multiplied_track_lever, colors='green', lineoffsets=0, label='track_leftlever', alpha=0.5)
    plt.eventplot(multiplied_track_cue, colors='blue', lineoffsets=0, label='end track_rightlever', alpha=0.5)
    plt.eventplot(multiplied_track_leverend, colors='pink', lineoffsets=0, label='track_sipperext', alpha=0.5)
    plt.suptitle(f'MOUSE: {mouse}')
    plt.legend()
    
    ########################## LEVER ALIGNMENT ##########################
    for i in range(len(track_rightlever)):
        if track_rightlever[i] > 0:
            baseline_neg10 = round((track_rightlever[i] + -10)* fs)
            baseline_neg5 = round((track_rightlever[i] -5)* fs)
            lever_zero = round(track_rightlever[i] * fs)
            lever_baseline = lever_zero + timerange[0] * fs
            lever_end = lever_zero + timerange[1] * fs
            
            zb = np.mean(df.iloc[baseline_neg10:baseline_neg5,1])
            zsd = np.std(df.iloc[baseline_neg10:baseline_neg5,1])
        
            rawtrial = np.array(df.iloc[lever_baseline:lever_end,1])
            
            trial = []
            for each in rawtrial:
                trial.append((each-zb)/zsd)
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            avglevertrace_dict[mouse,'Right',i]= track_rightlever[i], sampletrial, zb

    ########################## LEVER ALIGNMENT ##########################
    for i in range(len(track_leftlever)):
        if track_leftlever[i] > 0:
            baseline_neg10 = round((track_leftlever[i] + -10)* fs)
            baseline_neg5 = round((track_leftlever[i] -5)* fs)
            lever_zero = round(track_leftlever[i] * fs)
            lever_baseline = lever_zero + timerange[0] * fs
            lever_end = lever_zero + timerange[1] * fs
            
            zb = np.mean(df.iloc[baseline_neg10:baseline_neg5,1])
            zsd = np.std(df.iloc[baseline_neg10:baseline_neg5,1])
        
            rawtrial = np.array(df.iloc[lever_baseline:lever_end,1])
            
            trial = []
            for each in rawtrial:
                trial.append((each-zb)/zsd)
                
            
            sampletrial=[]
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            avglevertrace_dict[mouse,'Left',i]= track_leftlever[i], sampletrial, zb
        

    ############ LICKBOUT ALIGNMENT #################
    # Identify and sort lick bouts by length
    sorted_lick_bouts = identify_and_sort_lick_bouts(track_lick, min_licks=3, start_interval=1.0, end_interval=3.0)

    for length, bout in sorted_lick_bouts:
        start_time = bout[0]
        lickbout_each = []
        for items in bout:
            licklick = items-start_time
            lickbout_each.append(licklick)
        lick_rasterdata[mouse, length]=lickbout_each
        lickb_zero = round(start_time * fs)
        baseline_neg10 = round((start_time + -10)* fs)
        baseline_neg5 = round((start_time -5)* fs)
        lickb_baseline = lickb_zero + timerange[0] * fs
        lickb_end = lickb_zero + timerange[1] * fs
        
        zb = np.mean(df.iloc[baseline_neg10:baseline_neg5,1])
        zsd = np.std(df.iloc[baseline_neg10:baseline_neg5,1])
        
        
        rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,1])
        
        trial = []
        for each in rawtrial:
            trial.append((each-zb)/zsd)
        
        sampletrial=[]
        for k in range(0, len(trial), N):
            sampletrial.append(np.mean(trial[k:k+N-1]))
        
       
        bout_traces[mouse,i, length]=(sampletrial)
        avglickbouttrace_dict[mouse,bout[0]]= start_time, length, sampletrial, zb
    
    ####
    for sipperexts in track_sipperext:
        licks_per_trial = []
        for licks in track_lick:
            if licks - sipperexts > 0 and licks - sipperexts < 10:
                licks_per_trial.append(licks)
        sipperext_to_lick_dict[mouse,sipperexts] = licks_per_trial

    
############################################################################################################################
mice_high = ['A201','A204','A209','A212','A414']
mice_low = ['A205','A216','A402','A409','A410']
mice_comp =  ['A207','A214','A215','A403']

activelever_left = ['A201','A204','A207','A209','A212','A214','A215','A403','A410']
activelever_right =  ['A205','A216','A402','A404','A409','A414']

activelever_dict={}
for mouse, lever, num in avglevertrace_dict:
    if mouse in activelever_left and lever == 'Left':
        activelever_dict[mouse,num]=avglevertrace_dict[mouse, lever, num]
    elif mouse in activelever_right and lever == 'Right':
        activelever_dict[mouse,num]=avglevertrace_dict[mouse, lever, num]


############################################################################################################################
#latency to lick from sipper extension
############################################################################################################################

data = []
for mouse, sipperext in sipperext_to_lick_dict:
    if len(sipperext_to_lick_dict[mouse, sipperext]) > 0:
        latency = sipperext_to_lick_dict[(mouse, sipperext)][0] - sipperext
        
        # Determine group based on membership in predefined lists
        if mouse in mice_high:
            group = 'High'
        elif mouse in mice_low:
            group = 'Low'
        elif mouse in mice_comp:
            group = 'Comp'
        else:
            group = 'Unknown'

        # Append the collected data
        data.append({'Mouse': mouse, 'Group': group, 'Latency': latency})

# Create DataFrame from the collected data
latencytolick_df = pd.DataFrame(data)

############################################################################################################################
#by session average of all lever presses
############################################################################################################################
trace_high = {}
trace_low = {}
trace_comp = {}

session_high = []
session_low = []
session_compulsive = []
for key, value in activelever_dict.items():
    mouse, num = key
    trialtrace = value
    if mouse in mice_high and len(trialtrace[1]) == len(sampletrial):
        session_high.append(trialtrace[1])
        trace_high[mouse, num] = trialtrace[1]
    elif mouse in mice_low and len(trialtrace[1]) == len(sampletrial):
        session_low.append(trialtrace[1])
        trace_low[mouse, num] = trialtrace[1]
    elif mouse in mice_comp and len(trialtrace[1]) == len(sampletrial):
        session_compulsive.append(trialtrace[1])
        trace_comp[mouse, num] = trialtrace[1]

trace_high_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_high.items()]))
trace_low_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_low.items()]))
trace_comp_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_comp.items()]))

mean_high = np.mean(session_high, axis=0)
sem_high = np.std(session_high, axis=0) / np.sqrt(len(session_high))
mean_low = np.mean(session_low, axis=0)
sem_low = np.std(session_low, axis=0) / np.sqrt(len(session_low))
mean_comp = np.mean(session_compulsive, axis=0)
sem_comp = np.std(session_compulsive, axis=0) / np.sqrt(len(session_compulsive))
    
plt.figure(figsize=(10, 6))
if isinstance(mean_high, np.ndarray):
    plt.plot(mean_high, color=sns.color_palette("husl", 8)[0], label='High')
    plt.fill_between(range(len(mean_high)), mean_high - sem_high, mean_high + sem_high, color=sns.color_palette("husl", 8)[0], alpha=0.1)
if isinstance(mean_low, np.ndarray):
    plt.plot(mean_low, color=sns.color_palette("husl", 8)[2], label='Low')
    plt.fill_between(range(len(mean_low)), mean_low - sem_low, mean_low + sem_low, color=sns.color_palette("husl", 8)[2], alpha=0.1)
if isinstance(mean_comp, np.ndarray):
    plt.plot(mean_comp, color=sns.color_palette("husl", 8)[4], label='Compulsive')
    plt.fill_between(range(len(mean_comp)), mean_comp - sem_comp, mean_comp + sem_comp, color=sns.color_palette("husl", 8)[4], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(sampletrial)+1,len(sampletrial)/(timerange[1]-timerange[0])), 
           np.arange(timerange[0], timerange[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(sampletrial)/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lever Press Aligned Trace with SEM')
plt.legend()
plt.show()


############################################################################################################################
#by session average of at first lever press
############################################################################################################################

trace_high1 = {}
trace_low1 = {}
trace_comp1 = {}

session_high = []
session_low = []
session_compulsive = []
for key, value in activelever_dict.items():
    mouse, num = key
    trialtrace = value
    if mouse in mice_high and len(trialtrace[1]) == len(sampletrial) and num % 5 == 1:
        session_high.append(trialtrace[1])
        trace_high1[mouse, num] = trialtrace[1]
    elif mouse in mice_low and len(trialtrace[1]) == len(sampletrial) and num % 5 == 1:
        session_low.append(trialtrace[1])
        trace_low1[mouse, num] = trialtrace[1]
    elif mouse in mice_comp and len(trialtrace[1]) == len(sampletrial) and num % 5 == 1:
        session_compulsive.append(trialtrace[1])
        trace_comp1[mouse, num] = trialtrace[1]
            
trace_high1_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_high1.items()]))
trace_low1_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_low1.items()]))
trace_comp1_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_comp1.items()]))

mean_high = np.mean(session_high, axis=0)
sem_high = np.std(session_high, axis=0) / np.sqrt(len(session_high))
mean_low = np.mean(session_low, axis=0)
sem_low = np.std(session_low, axis=0) / np.sqrt(len(session_low))
mean_comp = np.mean(session_compulsive, axis=0)
sem_comp = np.std(session_compulsive, axis=0) / np.sqrt(len(session_compulsive))
    
plt.figure(figsize=(10, 6))
if isinstance(mean_high, np.ndarray):
    plt.plot(mean_high, color=sns.color_palette("husl", 8)[0], label='High')
    plt.fill_between(range(len(mean_high)), mean_high - sem_high, mean_high + sem_high, color=sns.color_palette("husl", 8)[0], alpha=0.1)
if isinstance(mean_low, np.ndarray):
    plt.plot(mean_low, color=sns.color_palette("husl", 8)[2], label='Low')
    plt.fill_between(range(len(mean_low)), mean_low - sem_low, mean_low + sem_low, color=sns.color_palette("husl", 8)[2], alpha=0.1)
if isinstance(mean_comp, np.ndarray):
    plt.plot(mean_comp, color=sns.color_palette("husl", 8)[4], label='Compulsive')
    plt.fill_between(range(len(mean_comp)), mean_comp - sem_comp, mean_comp + sem_comp, color=sns.color_palette("husl", 8)[4], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(sampletrial)+1,len(sampletrial)/(timerange[1]-timerange[0])), 
           np.arange(timerange[0], timerange[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(sampletrial)/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lever Press Aligned Trace with SEM')
plt.legend()
plt.show()


############################################################################################################################
#by session average of at fifth lever press
############################################################################################################################
trace_high5 = {}
trace_low5 = {}
trace_comp5 = {}

session_high = []
session_low = []
session_compulsive = []
for key, value in activelever_dict.items():
    mouse, num = key
    trialtrace = value
    if mouse in mice_high and len(trialtrace[1]) == len(sampletrial) and num % 5 == 0:
        session_high.append(trialtrace[1])
        trace_high5[mouse, num] = trialtrace[1]
    elif mouse in mice_low and len(trialtrace[1]) == len(sampletrial) and num % 5 == 0:
        session_low.append(trialtrace[1])
        trace_low5[mouse, num] = trialtrace[1]
    elif mouse in mice_comp and len(trialtrace[1]) == len(sampletrial) and num % 5 == 0:
        session_compulsive.append(trialtrace[1])
        trace_comp5[mouse, num] = trialtrace[1]
            
trace_high5_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_high5.items()]))
trace_low5_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_low5.items()]))
trace_comp5_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_comp5.items()]))

mean_high = np.mean(session_high, axis=0)
sem_high = np.std(session_high, axis=0) / np.sqrt(len(session_high))
mean_low = np.mean(session_low, axis=0)
sem_low = np.std(session_low, axis=0) / np.sqrt(len(session_low))
mean_comp = np.mean(session_compulsive, axis=0)
sem_comp = np.std(session_compulsive, axis=0) / np.sqrt(len(session_compulsive))
    
plt.figure(figsize=(10, 6))
if isinstance(mean_high, np.ndarray):
    plt.plot(mean_high, color=sns.color_palette("husl", 8)[0], label='High')
    plt.fill_between(range(len(mean_high)), mean_high - sem_high, mean_high + sem_high, color=sns.color_palette("husl", 8)[0], alpha=0.1)
if isinstance(mean_low, np.ndarray):
    plt.plot(mean_low, color=sns.color_palette("husl", 8)[2], label='Low')
    plt.fill_between(range(len(mean_low)), mean_low - sem_low, mean_low + sem_low, color=sns.color_palette("husl", 8)[2], alpha=0.1)
if isinstance(mean_comp, np.ndarray):
    plt.plot(mean_comp, color=sns.color_palette("husl", 8)[4], label='Compulsive')
    plt.fill_between(range(len(mean_comp)), mean_comp - sem_comp, mean_comp + sem_comp, color=sns.color_palette("husl", 8)[4], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(sampletrial)+1,len(sampletrial)/(timerange[1]-timerange[0])), 
           np.arange(timerange[0], timerange[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(sampletrial)/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lever Press Aligned Trace with SEM')
plt.legend()
plt.show()


############################################################################################################################
#by session lickbout
############################################################################################################################
trace_high_lick = {}
trace_low_lick = {}
trace_comp_lick = {}

session_high = []
session_low = []
session_compulsive = []
for key, value in avglickbouttrace_dict.items():
    mouse, num = key
    trialtrace = value
    if mouse in mice_high and len(trialtrace[2]) == len(sampletrial):
        session_high.append(trialtrace[2])
        trace_high_lick[mouse, trialtrace[1]] = trialtrace[2]
    elif mouse in mice_low and len(trialtrace[2]) == len(sampletrial):
        session_low.append(trialtrace[2])
        trace_low_lick[mouse, trialtrace[1]] = trialtrace[2]
    elif mouse in mice_comp and len(trialtrace[2]) == len(sampletrial):
        session_compulsive.append(trialtrace[2])
        trace_comp_lick[mouse, trialtrace[1]] = trialtrace[2]
            
trace_high_lick_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_high_lick.items()]))
trace_low_lick_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_low_lick.items()]))
trace_comp_lick_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in trace_comp_lick.items()]))

        
mean_high = np.mean(session_high, axis=0)
sem_high = np.std(session_high, axis=0) / np.sqrt(len(session_high))
mean_low = np.mean(session_low, axis=0)
sem_low = np.std(session_low, axis=0) / np.sqrt(len(session_low))
mean_comp = np.mean(session_compulsive, axis=0)
sem_comp = np.std(session_compulsive, axis=0) / np.sqrt(len(session_compulsive))
    
plt.figure(figsize=(10, 6))
if isinstance(mean_high, np.ndarray):
    plt.plot(mean_high, color=sns.color_palette("husl", 8)[0], label='High')
    plt.fill_between(range(len(mean_high)), mean_high - sem_high, mean_high + sem_high, color=sns.color_palette("husl", 8)[0], alpha=0.1)
if isinstance(mean_low, np.ndarray):
    plt.plot(mean_low, color=sns.color_palette("husl", 8)[2], label='Low')
    plt.fill_between(range(len(mean_low)), mean_low - sem_low, mean_low + sem_low, color=sns.color_palette("husl", 8)[2], alpha=0.1)
if isinstance(mean_comp, np.ndarray):
    plt.plot(mean_comp, color=sns.color_palette("husl", 8)[4], label='Compulsive')
    plt.fill_between(range(len(mean_comp)), mean_comp - sem_comp, mean_comp + sem_comp, color=sns.color_palette("husl", 8)[4], alpha=0.1)
plt.xlabel('Time (samples)')
plt.xticks(np.arange(0,len(sampletrial)+1,len(sampletrial)/(timerange[1]-timerange[0])), 
           np.arange(timerange[0], timerange[1]+1,1, dtype=int),
           rotation=0)
plt.axhline(y=0, linestyle=':', color='black')
plt.axvline(x=len(sampletrial)/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=1, color='black')
plt.ylabel('DF/F')
plt.title('Average Lick Bouts Aligned Trace ')
plt.legend()
plt.show()


############################################################################################################################
############################################################################################################################
############################################################################################################################


## insert peak height finding time range here ##
pktimerange = [0,10]
# pktimerange = [-2,2]

############################################################################################################################
### finding peak heights
from scipy.signal import find_peaks        
############################################################################################################################

peakheight_active={}
peakheight_active_df = pd.DataFrame(columns=['Mouse','Group','Press', 'X-Axis','PeakHeight'])
for mouse,press in activelever_dict:
    
    trace = activelever_dict[mouse,press][1]
    param_prom= 1
    peaks, properties = find_peaks(trace, prominence=param_prom, height=activelever_dict[mouse,press][2])
    trialpeakheights=[]
    trialprominence=[]
    time=[]
    for k in range(len(peaks)):
        if peaks[k] > len(trace)/(timerange[1]-timerange[0])*(pktimerange[0]-timerange[0]) and peaks[k] < len(trace)/(timerange[1]-timerange[0])*(pktimerange[1]-timerange[0]):
            trialpeakheights.append(properties['peak_heights'][k])
            time.append(peaks[k])
            trialprominence.append(properties['prominences'][k])

    num = len(peakheight_active_df)
    peakheight_active_df.at[num,'Mouse']=mouse
    peakheight_active_df.at[num,'Press']=press

    if len(trialpeakheights) > 0:
        bestpeak=max(trialprominence)
        bestpeak_time = time[trialprominence.index(bestpeak)]
        bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
        peakheight_active_df.at[num,'X-Axis']= bestpeak_time
        peakheight_active_df.at[num,'PeakHeight']= bestpeak_height
        peakheight_active[mouse,press] = bestpeak_height, bestpeak_time
        
    else:
        peakheight_active_df.at[num,'X-Axis']= np.nan
        peakheight_active_df.at[num,'PeakHeight']= activelever_dict[mouse,press][2]
        peakheight_active[mouse,press] = np.nan

    if mouse in mice_high:
        peakheight_active_df.at[num,'Group']='High'
    elif mouse in mice_low:
        peakheight_active_df.at[num,'Group']='Low'
    elif mouse in mice_comp:
        peakheight_active_df.at[num,'Group']='Compulsive'
        

peakheight_1LP={}
peakheight_1LP_df = pd.DataFrame(columns=['Mouse','Group','Press', 'X-Axis','PeakHeight'])
for mouse,press in activelever_dict:
    if press % 5 == 1:
        trace = activelever_dict[mouse,press][1]
        param_prom= 1
        peaks, properties = find_peaks(trace, prominence=param_prom, height=activelever_dict[mouse,press][2])
        trialpeakheights=[]
        trialprominence=[]
        time=[]
        for k in range(len(peaks)):
            if peaks[k] > len(trace)/(timerange[1]-timerange[0])*(pktimerange[0]-timerange[0]) and peaks[k] < len(trace)/(timerange[1]-timerange[0])*(pktimerange[1]-timerange[0]):
                trialpeakheights.append(properties['peak_heights'][k])
                time.append(peaks[k])
                trialprominence.append(properties['prominences'][k])

        num = len(peakheight_1LP_df)
        peakheight_1LP_df.at[num,'Mouse']=mouse
        peakheight_1LP_df.at[num,'Press']=press

        if len(trialpeakheights) > 0:
            bestpeak=max(trialprominence)
            bestpeak_time = time[trialprominence.index(bestpeak)]
            bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
            peakheight_1LP_df.at[num,'X-Axis']= bestpeak_time
            peakheight_1LP_df.at[num,'PeakHeight']= bestpeak_height
            peakheight_1LP[mouse,press] = bestpeak_height, bestpeak_time
            
        else:
            peakheight_1LP_df.at[num,'X-Axis']= np.nan
            peakheight_1LP_df.at[num,'PeakHeight']= activelever_dict[mouse,press][2]
            peakheight_1LP[mouse,press] = np.nan

        if mouse in mice_high:
            peakheight_1LP_df.at[num,'Group']='High'
        elif mouse in mice_low:
            peakheight_1LP_df.at[num,'Group']='Low'
        elif mouse in mice_comp:
            peakheight_1LP_df.at[num,'Group']='Compulsive'


peakheight_5LP={}       
peakheight_5LP_df = pd.DataFrame(columns=['Mouse','Group','Press', 'X-Axis','PeakHeight'])
for mouse,press in activelever_dict:
    if press % 5 == 0:
        trace = activelever_dict[mouse,press][1]
        param_prom= 1
        peaks, properties = find_peaks(trace, prominence=param_prom, height=activelever_dict[mouse,press][2])
        trialpeakheights=[]
        trialprominence=[]
        time=[]
        for k in range(len(peaks)):
            if peaks[k] > len(trace)/(timerange[1]-timerange[0])*(pktimerange[0]-timerange[0]) and peaks[k] < len(trace)/(timerange[1]-timerange[0])*(pktimerange[1]-timerange[0]):
                trialpeakheights.append(properties['peak_heights'][k])
                time.append(peaks[k])
                trialprominence.append(properties['prominences'][k])

        num = len(peakheight_5LP_df)
        peakheight_5LP_df.at[num,'Mouse']=mouse
        peakheight_5LP_df.at[num,'Press']=press

        if len(trialpeakheights) > 0:
            bestpeak=max(trialprominence)
            bestpeak_time = time[trialprominence.index(bestpeak)]
            bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
            peakheight_5LP_df.at[num,'X-Axis']= bestpeak_time
            peakheight_5LP_df.at[num,'PeakHeight']= bestpeak_height
            peakheight_5LP[mouse,press] = bestpeak_height, bestpeak_time
            
        else:
            peakheight_5LP_df.at[num,'X-Axis']= np.nan
            peakheight_5LP_df.at[num,'PeakHeight']= activelever_dict[mouse,press][2]
            peakheight_5LP[mouse,press] = np.nan

        if mouse in mice_high:
            peakheight_5LP_df.at[num,'Group']='High'
        elif mouse in mice_low:
            peakheight_5LP_df.at[num,'Group']='Low'
        elif mouse in mice_comp:
            peakheight_5LP_df.at[num,'Group']='Compulsive'


peakheight_lickbout={}
peakheight_lickbout_df = pd.DataFrame(columns=['Mouse','Group','Bout', 'X-Axis','PeakHeight'])
for mouse,bouttime in avglickbouttrace_dict:
    trace = avglickbouttrace_dict[mouse,bouttime][2]
    param_prom= 1
    peaks, properties = find_peaks(trace, prominence=param_prom, height=avglickbouttrace_dict[mouse,bouttime][3])
    trialpeakheights=[]
    trialprominence=[]
    time=[]
    for k in range(len(peaks)):
        if peaks[k] > len(trace)/(timerange[1]-timerange[0])*(pktimerange[0]-timerange[0]) and peaks[k] < len(trace)/(timerange[1]-timerange[0])*(pktimerange[1]-timerange[0]):
            trialpeakheights.append(properties['peak_heights'][k])
            time.append(peaks[k])
            trialprominence.append(properties['prominences'][k])

    num = len(peakheight_lickbout_df)
    peakheight_lickbout_df.at[num,'Mouse']=mouse
    peakheight_lickbout_df.at[num,'Bout']=avglickbouttrace_dict[mouse,bouttime][1]

    if len(trialpeakheights) > 0:
        bestpeak=max(trialprominence)
        bestpeak_time = time[trialprominence.index(bestpeak)]
        bestpeak_height= trialpeakheights[trialprominence.index(bestpeak)]
        peakheight_lickbout_df.at[num,'X-Axis']= bestpeak_time
        peakheight_lickbout_df.at[num,'PeakHeight']= bestpeak_height
        peakheight_lickbout[mouse,bouttime] = bestpeak_height, bestpeak_time
        
    else:
        peakheight_lickbout_df.at[num,'X-Axis']= np.nan
        peakheight_lickbout_df.at[num,'PeakHeight']= avglickbouttrace_dict[mouse,bouttime][3]
        peakheight_lickbout[mouse,bouttime] = np.nan, np.nan

    if mouse in mice_high:
        peakheight_lickbout_df.at[num,'Group']='High'
    elif mouse in mice_low:
        peakheight_lickbout_df.at[num,'Group']='Low'
    elif mouse in mice_comp:
        peakheight_lickbout_df.at[num,'Group']='Compulsive'

fig, axs = plt.subplots(3, sharey=True, sharex=True, figsize=(10,12))
for mouse, bouttime in avglickbouttrace_dict:
    if mouse in mice_high:
        if peakheight_lickbout[mouse,bouttime][1] != 0:
            axs[0].scatter(x= peakheight_lickbout[mouse,bouttime][1], y=peakheight_lickbout[mouse,bouttime][0], color=sns.color_palette("husl", 8)[0])
        axs[0].plot(avglickbouttrace_dict[mouse,bouttime][2],color=sns.color_palette("husl", 8)[0],label='High', alpha=0.2)
        axs[0].axvline(x=len(avglickbouttrace_dict[mouse,bouttime][2])/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=0.5, color='black')
    if mouse in mice_low:
        if peakheight_lickbout[mouse,bouttime][1] != 0:
            axs[1].scatter(x= peakheight_lickbout[mouse,bouttime][1], y=peakheight_lickbout[mouse,bouttime][0], color=sns.color_palette("husl", 8)[2])
        axs[1].plot(avglickbouttrace_dict[mouse,bouttime][2],color=sns.color_palette("husl", 8)[2],label='Low', alpha=0.2)
        axs[1].axvline(x=len(avglickbouttrace_dict[mouse,bouttime][2])/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=0.5, color='black')
    if mouse in mice_comp:
        if peakheight_lickbout[mouse,bouttime][1] != 0:
            axs[2].scatter(x= peakheight_lickbout[mouse,bouttime][1], y=peakheight_lickbout[mouse,bouttime][0], color=sns.color_palette("husl", 8)[4])
        axs[2].plot(avglickbouttrace_dict[mouse,bouttime][2],color=sns.color_palette("husl", 8)[4],label='Compulsive', alpha=0.2)
        axs[2].axvline(x=len(avglickbouttrace_dict[mouse,bouttime][2])/(timerange[1]-timerange[0])*(0-timerange[0]),linewidth=0.5, color='black')

plt.xticks(np.arange(0,len(avglickbouttrace_dict[mouse,bouttime][2])+1,len(avglickbouttrace_dict[mouse,bouttime][2])/(timerange[1]-timerange[0])), 
       np.arange(timerange[0], timerange[1]+1,1, dtype=int),
       rotation=0)
plt.xlabel('Lick Bout Onset (s)')
fig.tight_layout()

############################################################################################################################
############################################################################################################################
############################################################################################################################


############################################################################################################################
### finding area under the curve
# Calculates the area under the curve in that interval using Simpson's rule, which is generally accurate for oscillating data like neural traces.
import numpy as np
from scipy.integrate import simps
############################################################################################################################
# Define the interval of interest here
start_time = -2
end_time = 2


auc_active_dict = {}
auc_active_df = pd.DataFrame(columns=['Mouse','Group','Press', 'AUC'])
for mouse, press in activelever_dict:
    time = np.linspace(timerange[0], timerange[1], len(activelever_dict[mouse, press][1]))
    trace = activelever_dict[mouse, press][1]
    
    if len(trace) > 0:
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_active_dict[mouse,press] = area
        count = len(auc_active_df)
        auc_active_df.at[count,'Mouse'] = mouse
        auc_active_df.at[count,'Press'] = press
        auc_active_df.at[count,'AUC'] = area
    
        
        if mouse in mice_high:
            auc_active_df.at[count,'Group']='High'
        elif mouse in mice_low:
            auc_active_df.at[count,'Group']='Low'
        elif mouse in mice_comp:
            auc_active_df.at[count,'Group']='Compulsive'

plt.figure(figsize=(10,6))
sns.pointplot(x='Group', y='AUC', data=auc_active_df)
sns.boxplot(x='Group', y='AUC', data=auc_active_df)
sns.swarmplot(x='Group', y='AUC', data=auc_active_df, color=".25")
plt.title('Area Under the Curve by Animal Group')
plt.xlabel('Group')
plt.ylabel('AUC Value')
plt.show()

from scipy import stats

# One-way ANOVA
f_statistic, p_value = stats.f_oneway(
    auc_active_df[auc_active_df['Group']=='High']['AUC'],
    auc_active_df[auc_active_df['Group']=='Low']['AUC'],
    auc_active_df[auc_active_df['Group']=='Compulsive']['AUC']
)
print(f"ANOVA p-value: {p_value}")

auc_1LP_dict = {}
auc_1LP_df = pd.DataFrame(columns=['Mouse','Group','Press', 'AUC'])
for mouse, press in activelever_dict:
    if press % 5 == 1:
        time = np.linspace(timerange[0], timerange[1], len(activelever_dict[mouse, press][1]))
        trace = activelever_dict[mouse, press][1]
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_1LP_dict[mouse,press] = area
        count = len(auc_1LP_df)
        auc_1LP_df.at[count,'Mouse'] = mouse
        auc_1LP_df.at[count,'Press'] = press
        auc_1LP_df.at[count,'AUC'] = area
        
        if mouse in mice_high:
            auc_1LP_df.at[count,'Group']='High'
        elif mouse in mice_low:
            auc_1LP_df.at[count,'Group']='Low'
        elif mouse in mice_comp:
            auc_1LP_df.at[count,'Group']='Compulsive'
        
  
auc_5LP_dict = {}
auc_5LP_df = pd.DataFrame(columns=['Mouse','Group','Press', 'AUC'])
for mouse, press in activelever_dict:
    if press % 5 == 0:
        time = np.linspace(timerange[0], timerange[1], len(activelever_dict[mouse, press][1]))
        trace = activelever_dict[mouse, press][1]
        
        # Find the indices for the interval 0 to 1 second
        start_index = np.searchsorted(time, start_time)
        end_index = np.searchsorted(time, end_time)
        
        # Select the relevant portion of the trace
        time_segment = time[start_index:end_index]
        trace_segment = trace[start_index:end_index]
        
        # Calculate the area under the curve for the interval 0 to 1 second
        area = simps(trace_segment, time_segment)
        
        auc_5LP_dict[mouse,press] = area
        count = len(auc_5LP_df)
        auc_5LP_df.at[count,'Mouse'] = mouse
        auc_5LP_df.at[count,'Press'] = press
        auc_5LP_df.at[count,'AUC'] = area
        
        if mouse in mice_high:
            auc_5LP_df.at[count,'Group']='High'
        elif mouse in mice_low:
            auc_5LP_df.at[count,'Group']='Low'
        elif mouse in mice_comp:
            auc_5LP_df.at[count,'Group']='Compulsive'


auc_lick_dict = {}
auc_lick_df = pd.DataFrame(columns=['Mouse','Group','Bout', 'AUC'])
for mouse, press in avglickbouttrace_dict:
    time = np.linspace(timerange[0], timerange[1], len(avglickbouttrace_dict[mouse, press][2]))
    trace = avglickbouttrace_dict[mouse, press][2]
    
    # Find the indices for the interval 0 to 1 second
    start_index = np.searchsorted(time, start_time)
    end_index = np.searchsorted(time, end_time)
    
    # Select the relevant portion of the trace
    time_segment = time[start_index:end_index]
    trace_segment = trace[start_index:end_index]
    
    # Calculate the area under the curve for the interval 0 to 1 second
    area = simps(trace_segment, time_segment)
    
    auc_lick_dict[mouse,press] = area
    count = len(auc_lick_df)
    auc_lick_df.at[count,'Mouse'] = mouse
    auc_lick_df.at[count,'Bout'] = avglickbouttrace_dict[mouse, press][1]
    auc_lick_df.at[count,'AUC'] = area
    
    if mouse in mice_high:
        auc_lick_df.at[count,'Group']='High'
    elif mouse in mice_low:
        auc_lick_df.at[count,'Group']='Low'
    elif mouse in mice_comp:
        auc_lick_df.at[count,'Group']='Compulsive'
        
        
        
        
# import tdt
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.signal import find_peaks    

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/First_EtOH_Binge/'
# mice = [ 'A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A409', 'A410', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Last_EtOH_Binge/'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A410']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Sacchrin/'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_0.25Q/'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_1Q/'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/STAR_EtOH_Only/'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A212', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Withdrawal_D1'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A414']

# # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/Marie_EtOH_Data/Withdrawal_D28'
# # mice = ['A201', 'A204', 'A205', 'A207', 'A209', 'A212', 'A214', 'A215', 'A216', 'A402', 'A403', 'A404', 'A409', 'A410', 'A414']



# ############################################################################################################################
# mice_high = ['A201','A204','A209','A212','A414', 'A404']
# mice_low = ['A205','A216','A402','A409','A410']
# mice_comp =  ['A207','A214','A215','A403']

# activelever_left = ['A201','A204','A207','A209','A212','A214','A215','A403','A410']
# activelever_right =  ['A205','A216','A402','A404','A409','A414']
# ############################################################################################################################

# ######################################################################################

# trace_dict = {}
# eventrate_dict = {}
# eventamplitude_dict = {}
# for mouse in mice:
#     mouse_dir = os.path.join(folder, mouse)
#     data = tdt.read_block(mouse_dir)
#     df = pd.DataFrame()
#     df['Sig470'] = data.streams._470A.data
#     df['Dff'] = ((df['Sig470']-np.mean(df['Sig470']))/np.mean(df['Sig470']))
#     fs = round(data.streams._470A.fs)

#     split1 = str(data.epocs).split('\t')
#     y = []
#     for elements in split1:
#         x = elements.split('\n')
#         if '[struct]' in x:
#             x.remove('[struct]')
#         y.append(x)
#     z= [item for sublist in y for item in sublist]
    
#     signal=df['Dff'].values[round(data.epocs.Sir_.offset[0]*fs):round(data.epocs.Sir_.onset[-1]*fs)]
#     #spectral
  
 
#     mean_signal = np.mean(signal)
#     std_signal = np.std(signal)
#     zscore_signal = []
#     for each in signal:
#         zscore_signal.append((each-mean_signal)/std_signal)
    
#     param_prom= 1
#     peaks, properties = find_peaks(zscore_signal, prominence=param_prom, height=0.0)
   
#     if mouse in mice_high:
#         trace_dict[mouse,'High']= zscore_signal
#         eventrate_dict[mouse,'High']= len(peaks)
#         eventamplitude_dict[mouse,'High']= properties['peak_heights']
#     elif mouse in mice_low:
#         trace_dict[mouse,'Low']= zscore_signal
#         eventrate_dict[mouse,'Low']= len(peaks)
#         eventamplitude_dict[mouse,'Low']= properties['peak_heights']
#     elif mouse in mice_comp:
#         trace_dict[mouse,'Comp']= zscore_signal
#         eventrate_dict[mouse,'Comp']= len(peaks)
#         eventamplitude_dict[mouse,'Comp']= properties['peak_heights']
   


# # Define function to calculate SEM
# def sem(arr):
#     return np.std(arr, axis=0) / np.sqrt(len(arr))


# event_high_df = pd.DataFrame()
# event_low_df = pd.DataFrame()
# event_comp_df = pd.DataFrame()

# plt.figure()

# mean_events_high=[]
# mean_events_low=[]
# mean_events_comp=[]

# for mouse, group in eventrate_dict:
#     if group == 'High':
#         mean_events_high.append(eventrate_dict[mouse, group])
#         count= len(event_high_df)
#         event_high_df.at[count,'mouse']=mouse
#         event_high_df.at[count,'group']=group
#         event_high_df.at[count,'events']=eventrate_dict[mouse, group]
       
#     elif group == 'Low':
#         mean_events_low.append(eventrate_dict[mouse, group])
#         count= len(event_low_df)
#         event_low_df.at[count,'mouse']=mouse
#         event_low_df.at[count,'group']=group
#         event_low_df.at[count,'events']=eventrate_dict[mouse, group]
#     elif group == 'Comp':
#         mean_events_comp.append(eventrate_dict[mouse, group])
#         count= len(event_comp_df)
#         event_comp_df.at[count,'mouse']=mouse
#         event_comp_df.at[count,'group']=group
#         event_comp_df.at[count,'events']=eventrate_dict[mouse, group]

# plt.scatter(x=0, y=np.mean(mean_events_high), label='High')
# plt.errorbar(x=0, y=np.mean(mean_events_high), yerr=sem(mean_events_high),capsize=3)
# plt.scatter(x=1, y=np.mean(mean_events_low), label='Low')
# plt.errorbar(x=1, y=np.mean(mean_events_low), yerr=sem(mean_events_low), capsize=3)
# plt.scatter(x=2, y=np.mean(mean_events_comp),label='Comp')
# plt.errorbar(x=2, y=np.mean(mean_events_comp), yerr=sem(mean_events_comp),capsize=3)
# plt.xlabel('Group')
# plt.ylabel('Events in Session')
# plt.legend()
# plt.show()



# #### TRIALS ON HEATMAP ALIGNED TO CUE of all session sorted by latency to lick ####


# alltraces_high=[]
# alltraces_low=[]
# alltraces_comp=[]

# for subj,group in trace_dict:
#     if group == 'High':
#         alltraces_high.append(trace_dict[subj,group])
#     elif group == 'Low':
#         alltraces_low.append(trace_dict[subj,group])
#     elif group == 'Comp':
#         alltraces_comp.append(trace_dict[subj,group])

# bysession_high_df= pd.DataFrame(alltraces_high)  
# bysession_low_df= pd.DataFrame(alltraces_low)  
# bysession_comp_df= pd.DataFrame(alltraces_comp)  

# fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True)
# sns.heatmap(bysession_high_df, cmap='RdBu', vmin=-5, vmax=5, ax=ax1)
# sns.heatmap(bysession_low_df, cmap='RdBu', vmin=-5, vmax=5, ax=ax2)
# sns.heatmap(bysession_comp_df, cmap='RdBu', vmin=-5, vmax=5, ax=ax3)


# plt.xlabel('Time (sec)')
# plt.show()


  