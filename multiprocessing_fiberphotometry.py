import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy
from multiprocessing import Pool, Manager
import functools

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

def process_mouse_date(args):
    """Process a single mouse-date combination"""
    mouse, date, folder, dates_list = args
    
    # Initialize local dictionaries for this process
    local_results = {
        'alltrialtrace_dict': {},
        'avgcuetrace_dict': {},
        'avglevertrace_dict': {},
        'avgflicktrace_dict': {},
        'avgflicktrace1_dict': {},
        'avglickbouttrace_dict': {}
    }
    
    # Parameters
    timerange_cue = [-2, 5]
    timerange_lever = [-2, 5]
    timerange_lick = [-2, 10]
    active_time = [-2, 40]
    N = 100
    
    try:
        mouse_dir = os.path.join(folder, mouse)
        date_dir = os.path.join(mouse_dir, date)
        data = tdt.read_block(date_dir)
        print(f"Processing: {date_dir}")
        
        df = pd.DataFrame()
        df['Sig405'] = data.streams._405B.data
        df['Sig465'] = data.streams._465B.data[0:len(data.streams._405B.data)]
        df['Dff'] = (df['Sig465']-df['Sig405'])/df['Sig465']
        fs = round(data.streams._465B.fs)

        # Parse events
        split1 = str(data.epocs).split('\t')
        y = []
        for elements in split1:
            x = elements.split('\n')
            if '[struct]' in x:
                x.remove('[struct]')
            y.append(x)
        z = [item for sublist in y for item in sublist]

        fp_df = pd.DataFrame(columns=['Event','Timestamp'])
        events = ['Cue', 'Press', 'Licks', 'Timeout Press']
        epocs = ['Po0_','Po6_','Po4_','Po2_']
        
        for a, b in zip(events, epocs):
            if b in z:
                event_df = pd.DataFrame(columns=['Event','Timestamp'])
                event_df['Timestamp'] = data.epocs[b].onset
                event_df['Event'] = a
                fp_df = pd.concat([fp_df, event_df])
        
        # Extract timestamps
        track_cue = []
        track_lever = []
        track_licks = []
        track_to = []
        
        for i in range(len(fp_df)):
            if fp_df.iloc[i,0] == 'Cue':
                track_cue.append(fp_df.iloc[i,1])
            elif fp_df.iloc[i,0] == 'Press':
                track_lever.append(fp_df.iloc[i,1])
            elif fp_df.iloc[i,0] == 'Licks':
                track_licks.append(fp_df.iloc[i,1])
            elif fp_df.iloc[i,0] == 'Timeout Press':
                track_to.append(fp_df.iloc[i,1])
        
        date_idx = dates_list.index(date)
        cuebaseline = {}
        
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
            
            sampletrial = []
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            local_results['avgcuetrace_dict'][(mouse, date_idx, i)] = (track_cue[i], sampletrial, aligntobase)

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
            
            sampletrial = []
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for trial_num in range(len(track_cue)):
                if track_lever[i] - track_cue[trial_num] > 0 and track_lever[i] - track_cue[trial_num] < 20:
                    cue_trial = trial_num
                    break
            
            local_results['avglevertrace_dict'][(mouse, date_idx, cue_trial)] = (track_lever[i], sampletrial, aligntobase)
            cuebaseline[(mouse, date_idx, cue_trial)] = (np.mean(df.iloc[lever_baseline:lever_zero,2]), np.std(df.iloc[cue_baseline:cue_zero,2]))

        ################## FIRST LICK ALIGNMENT ################
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
            
            sampletrial = []
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for trial_num in range(len(track_cue)):
                if track_flicks[i] - track_cue[trial_num] > 0 and track_flicks[i] - track_cue[trial_num] < 30:
                    cue_trial = trial_num
                    break
            local_results['avgflicktrace_dict'][(mouse, date_idx, cue_trial)] = (track_flicks[i], sampletrial, aligntobase)

        ################## FIRST LICK ALIGNMENT WITH CUE BASELINE ################
        for i in range(len(track_flicks)):
            for trial_num in range(len(track_cue)):
                if track_flicks[i] - track_cue[trial_num] > 0 and track_flicks[i] - track_cue[trial_num] < 30:
                    cue_trial = trial_num
                    break
                    
            flick_zero = round(track_flicks[i] * fs)
            flick_baseline = flick_zero + timerange_lick[0] * fs
            flick_end = flick_zero + timerange_lick[1] * fs
            
            if (mouse, date_idx, cue_trial) in cuebaseline:
                cue_mean = cuebaseline[(mouse, date_idx, cue_trial)][0]
                cue_std = cuebaseline[(mouse, date_idx, cue_trial)][1]
                rawtrial = np.array(df.iloc[flick_baseline:flick_end,2])

                trial = []
                for each in rawtrial:
                    trial.append((each-cue_mean)/cue_std)
                
                sampletrial = []
                for k in range(0, len(trial), N):
                    sampletrial.append(np.mean(trial[k:k+N-1]))
                
                local_results['avgflicktrace1_dict'][(mouse, date_idx, cue_trial)] = (track_flicks[i], sampletrial, cue_mean)

        ############ LICKBOUT ALIGNMENT #################
        sorted_lick_bouts = identify_and_sort_lick_bouts(track_licks, max_interval)

        for length, bout in sorted_lick_bouts:
            start_time = bout[0]
            lickb_zero = round(start_time * fs)
            lickb_baseline = lickb_zero + timerange_lick[0] * fs
            lickb_end = lickb_zero + timerange_lick[1] * fs
            
            aligntobase = np.mean(df.iloc[lickb_baseline:lickb_zero,2])
            rawtrial = np.array(df.iloc[lickb_baseline:lickb_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
            
            sampletrial = []
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))
            
            for i in range(len(track_cue)):
                if start_time - track_cue[i] > 0 and start_time - track_cue[i] < 30:
                    trial_num = i
                    break
            local_results['avglickbouttrace_dict'][(mouse, date_idx, trial_num)] = (start_time, length, sampletrial)

        ########################## ALL ALIGNMENT ##########################    
        for i in range(len(track_cue)):
            cue_zero = round(track_cue[i] * fs)
            cue_baseline = cue_zero + active_time[0] * fs
            cue_end = cue_zero + active_time[1] * fs
            
            levertime = np.nan
            for l in range(len(track_lever)):
                if track_lever[l] - track_cue[i] > 0 and track_lever[l] - track_cue[i] < 20:
                    levertime = track_lever[l]
                    break
            
            flicktime = np.nan
            for m in range(len(track_flicks)):
                if track_flicks[m] - track_cue[i] > 0 and track_flicks[m] - track_cue[i] < 30:
                    flicktime = track_flicks[m]
                    break
                    
            aligntobase = np.mean(df.iloc[cue_baseline:cue_zero,2])
            rawtrial = np.array(df.iloc[cue_baseline:cue_end,2])
            
            trial = []
            for each in rawtrial:
                trial.append((each-aligntobase)/np.std(df.iloc[cue_baseline:cue_zero,2]))
            
            sampletrial = []
            for k in range(0, len(trial), N):
                sampletrial.append(np.mean(trial[k:k+N-1]))

            local_results['alltrialtrace_dict'][(mouse, date_idx, i)] = (track_cue[i], levertime, flicktime, sampletrial)

        return local_results
        
    except Exception as e:
        print(f"Error processing {mouse}/{date}: {str(e)}")
        return local_results

def run_parallel_processing(folder, mice):
    """Main function to run parallel processing"""
    
    # Create list of all mouse-date combinations to process
    tasks = []
    for mouse in mice:
        mouse_dir = os.path.join(folder, mouse)
        if os.path.exists(mouse_dir):
            dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
            dates.sort()
            for date in dates:
                tasks.append((mouse, date, folder, dates))
    
    # Initialize result dictionaries
    alltrialtrace_dict = {}
    avgcuetrace_dict = {}
    avglevertrace_dict = {}
    avgflicktrace_dict = {}
    avgflicktrace1_dict = {}
    avglickbouttrace_dict = {}
    
    # Run parallel processing
    print(f"/nProcessing {len(tasks)} mouse-date combinations in parallel...")
    
    # Determine number of processes (use all available cores minus 1)
    num_processes = max(1, os.cpu_count() - 1)
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_mouse_date, tasks)
    
    # Merge results from all processes
    print("Merging results...")
    for result in results:
        alltrialtrace_dict.update(result['alltrialtrace_dict'])
        avgcuetrace_dict.update(result['avgcuetrace_dict'])
        avglevertrace_dict.update(result['avglevertrace_dict'])
        avgflicktrace_dict.update(result['avgflicktrace_dict'])
        avgflicktrace1_dict.update(result['avgflicktrace1_dict'])
        avglickbouttrace_dict.update(result['avglickbouttrace_dict'])
    
    print("Processing complete!")
    
    return {
        'alltrialtrace_dict': alltrialtrace_dict,
        'avgcuetrace_dict': avgcuetrace_dict,
        'avglevertrace_dict': avglevertrace_dict,
        'avgflicktrace_dict': avgflicktrace_dict,
        'avgflicktrace1_dict': avgflicktrace1_dict,
        'avglickbouttrace_dict': avglickbouttrace_dict
    }


# Run the parallel processing
if __name__ == '__main__':
    # The folder and mice definitions
        
    # D2 MEDIUM SPINY NEURONS (ALCOHOL)
    folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHLearning/'
    mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321','8729','8730','8731','8732']
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_1WeekWD/'
    # mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321','8729','8730','8731','8732']
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_EtOHExtinction/'
    # mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321','8729','8730','8731','8732']
    
    
    # D2 MEDIUM SPINY NEURONS (SUCROSE)
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucLearning/'
    # mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SucExtinction/'
    # mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
    
    # D2 MEDIUM SPINY NEURONS (SUCROSE TO ETHANOL TO SUCROSE)
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_EtOHLearning/'
    # mice = ['7678', '7680', '7899']
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_AlcExtinction/'
    # mice = ['7678', '7680', '7899']
    # folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/D2_SuctoEtOH_SucExtinction2/'
    # mice = ['7678', '7680', '7899']
    
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

    results = run_parallel_processing(folder,mice)
    
    # Extract the dictionaries
    alltrialtrace_dict = results['alltrialtrace_dict']
    avgcuetrace_dict = results['avgcuetrace_dict']
    avglevertrace_dict = results['avglevertrace_dict']
    avgflicktrace_dict = results['avgflicktrace_dict']
    avgflicktrace1_dict = results['avgflicktrace1_dict']
    avglickbouttrace_dict = results['avglickbouttrace_dict']
    
    print(f"Generated {len(alltrialtrace_dict)} entries in alltrialtrace_dict")
    print(f"Generated {len(avgcuetrace_dict)} entries in avgcuetrace_dict")
    print(f"Generated {len(avglevertrace_dict)} entries in avglevertrace_dict")
    print(f"Generated {len(avgflicktrace_dict)} entries in avgflicktrace_dict")
    print(f"Generated {len(avgflicktrace1_dict)} entries in avgflicktrace1_dict")
    print(f"Generated {len(avglickbouttrace_dict)} entries in avglickbouttrace_dict")

###################################################################################################################
############################################################################################################################
### LOOKING AT THE TRACE BY ALL SESSION 
############################################################################################################################
############################################################################################################################

if __name__ == '__main__':
    # Parameters
    timerange_cue = [-2, 5]
    timerange_lever = [-2, 5]
    timerange_lick = [-2, 10]
    active_time = [-2, 40]
    session_ranges = 8
    
    #by session average
    #colors10 = ['#8da0cb','#66c2a5','#a6d854','#ffd92f','#e78ac3','#e5c494','#bc80bd','#fc8d62']
    
    
    colors10 = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen', 'mediumseagreen', 'mediumturquoise', 'deepskyblue', 'dodgerblue', 'slateblue', 'darkorchid','purple']
    
    session_data = {}  
    for key, value in avgcuetrace_dict.items():
        mouse, session, trial = key
        time, trialtrace, baseline = value
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
        #if session in [0,1,6, 7,8,9]:
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
    
    
    plt.savefig('/Users/kristineyoon/Documents/flick_sucextinct.pdf', transparent=True)
    plt.show()
    

############################################################################################################################
############################################################################################################################
### PEAK HEIGHTS WITH MAX HEIGHT
############################################################################################################################
############################################################################################################################
# plt.figure(figsize=(10, 6))
if __name__ == '__main__':
    # Parameters
    timerange_cue = [-2, 5]
    timerange_lever = [-2, 5]
    timerange_lick = [-2, 10]
    active_time = [-2, 40]
    session_ranges = 8

    
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
    
     

    
    allleverpeakheights ={}
    peakheight_lever_df = pd.DataFrame(columns=['Mouse','Session','Trial','X-Axis','PeakHeight'])
    for mouse,session,trial in avglevertrace_dict:
        if trial in range (10):
            time = np.linspace(timerange_cue[0], timerange_cue[1], len(avglevertrace_dict[mouse,session,trial][1]))  # 500 points from -2 to 10 seconds
            trace = avglevertrace_dict[mouse,session,trial][1]
            
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
            num = len(peakheight_lever_df)
            peakheight_lever_df.at[num,'Mouse']=mouse
            peakheight_lever_df.at[num,'Session']=session
            peakheight_lever_df.at[num,'Trial']=trial
            peakheight_lever_df.at[num,'X-Axis']= peakheighttime
            peakheight_lever_df.at[num,'PeakHeight']= peakheight
            allleverpeakheights[mouse,session,trial] = peakheight, peakheighttime
    

    
    
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
    


############################################################################################################################
#### TRIALS ON HEATMAP ALIGNED TO CUE by session sorted by mean trace ####

if __name__ == '__main__':
    # Parameters
    timerange_cue = [-2, 5]
    timerange_lever = [-2, 5]
    timerange_lick = [-2, 10]
    active_time = [-2, 40]
    session_ranges = 8
    
    colormap_special = sns.diverging_palette(250, 370, l=65, as_cmap=True)
    
    sess_of_interest = [1,6]
    end_time = 5.1
    alltraces=[]
    for i in sess_of_interest:
        bysession = {}
        for subj,session,trial in avgcuetrace_dict:
            if session == i and trial in range (10):
                timespace = np.linspace(timerange_cue[0], timerange_cue[1], len(avgcuetrace_dict[subj,i,trial][1]))
                trace = avgcuetrace_dict[subj,i,trial][1]
                start_index = np.searchsorted(timespace, timerange_cue[0])
                end_index = np.searchsorted(timespace, timerange_cue[1])  
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
    
    fig, axs = plt.subplots(1,len(sess_of_interest), sharex=True)
    for i in sess_of_interest:
        bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
        sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
        difference_array=np.absolute(time_segment-0)
        axs[sess_of_interest.index(i)].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
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
    
    fig, axs = plt.subplots(1,len(sess_of_interest), sharex=True)
    for i in sess_of_interest:
        bysessiondf= pd.DataFrame(alltraces[sess_of_interest.index(i)])    
        sns.heatmap(bysessiondf, cmap=colormap_special, vmin=-5, vmax=5, ax=axs[sess_of_interest.index(i)])
        difference_array=np.absolute(time_segment-0)
        axs[sess_of_interest.index(i)].axvline(x=difference_array.argmin(),linewidth=1, color='black', label='Cue Onset')
    plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(end_time-start_time)), np.arange(start_time,end_time,1, dtype=int),rotation=0)
    plt.ylabel('Trials')
    plt.xlabel('Time Alined to Lick (sec)')
    plt.show()
    
import pandas as pd
import numpy as np

# Assuming you have the dictionaries filled from the previous processing steps
# For simplicity, let's assume avgcuetrace_dict, avglevertrace_dict, avgflicktrace1_dict contain the necessary data.

def create_pca_matrix(avgcuetrace_dict, avglevertrace_dict, avgflicktrace1_dict, avglickbouttrace_dict, allcuepeakheights, allleverpeakheights, allflickpeakheights):
    pca_data = []

    # Loop through each session and trial
    for (mouse, session, trial) in avgcuetrace_dict.keys():
        if trial < 10:  
            # Retrieve data
            cue_data = avgcuetrace_dict[(mouse, session, trial)]
            lever_data = avglevertrace_dict.get((mouse, session, trial), (None, None))  # Handle missing data
            flick_data = avgflicktrace1_dict.get((mouse, session, trial), (None, None))
            lever_peak_data = allleverpeakheights.get((mouse, session, trial), (None, None)) 
            flick_peak_data = allflickpeakheights.get((mouse, session, trial), (None, None))
            
            peak_height_cue = allcuepeakheights[mouse, session, trial][0] 
            
            # if lever_peak_data == (None, None):
            #     peak_height_lever = 0
            # else:
            #     peak_height_lever = lever_peak_data[0]
                
            if flick_peak_data == (None, None):
                peak_height_flick = 0
            else:
                peak_height_flick = flick_peak_data[0]
                
            # Licks prior (counting from trials)
            licks_prior = 0
            for t in range(trial-1):
                if (mouse, session, t) in avglickbouttrace_dict:
                    licks_prior = licks_prior + avglickbouttrace_dict[mouse, session, t][1]

            
            # Latency (assuming the second value of lever_data is the time of lever press)
            if lever_data == (None, None):
                latency_press = 20
            else:
                latency_press = lever_data[0] -  cue_data[0] 
            
            if lever_data == (None, None):
                latency_lick = 30
            elif flick_data == (None, None):
                latency_lick = 10
            else:
                latency_lick = flick_data[0] -  lever_data[0]
            
            # Compile data into a row of the matrix
            pca_data.append([
                session,
                trial,
                licks_prior,  
                peak_height_cue,
                #peak_height_lever,
                peak_height_flick,
                latency_press,
                latency_lick
            ])

    # Create DataFrame for PCA analysis
    pca_matrix = pd.DataFrame(pca_data, columns=['session', 'trial', 'licks_prior', 'peak_height_cue', 'peak_height_lever', 'peak_height_flick', 
                                                 #'latency_press', 
                                                 'latency_lick'])
    
    return pca_matrix

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats
import numpy as np

if __name__ == '__main__':
    # Construct PCA matrix
    pca_matrix = create_pca_matrix(avgcuetrace_dict, avglevertrace_dict, avgflicktrace1_dict, avglickbouttrace_dict, allcuepeakheights, allleverpeakheights, allflickpeakheights)
    print(pca_matrix)


    
    # Standardize across columns (features), not rows (samples)
    org_zscore = scipy.stats.zscore(pca_matrix, axis=0)
    matrix = np.array(org_zscore)
    
    # Ensure shape matches your feature list
    feature_names = ['session', 'trial', 'licks_prior', 'peak_height_cue',
                     #'peak_height_lever', 
                     'peak_height_flick', 'latency_press', 'latency_lick']
    
    df = pd.DataFrame(matrix, columns=feature_names)
    
    # PCA
    pca = PCA(n_components=7)
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
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # Standardize the data (important for k-means)
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    
    
    # Fit PCA on the standardized data
    pca_result = pca.fit_transform(df_standardized)
    
    # Get the transformed data (with reduced dimensions)
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2','PC3','PC4','PC5', 'PC6','PC7','PC8'])
    plt.scatter(df_pca['PC1'], df_pca['PC2'], cmap='viridis', edgecolors='k')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.show()
