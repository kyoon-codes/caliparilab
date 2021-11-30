import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar
import scipy.stats as sp
import os
import pickle

###############################################################################
# Function module for plotting NPM data lined to TTLs
###############################################################################

def Plot_TTL_aligned_heatmap(Data_df, trigger, grid, grid_Xpos, grid_Ypos, Data_range, Norm='Zscored_trial',
                             Show_plot=True, X_ticks=[0, 20, 40, 60]):
    'Data_df: Data; trigger: name of TTL; grid= plt.GridSpec(len(Dates),5, wspace=0.05, hspace=0.1)'
    # Get the size of the peak for the instance you peak_normalize
    peak = np.max(Data_df['dFF'].values[100:])  # discard first 100 to avoid the noise peak at the onset of recording
    Event_timestamps = []
    Preceding_event_timestamps = []
    Following_event_timestamps = []
    Traces = []
    Plot_scatter = False
    if trigger == 'LeverPress':
        Traces = []
        # ON_idx=[i+1 for i,(a,b) in enumerate(zip(Data_df['LeverOut'].values[:-1], Data_df['LeverOut'].values[1:]))  if b>a ]
        LP_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        # if len(ON_idx)<2:
        #     return Traces, Event_timestamps
        if len(LP_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        # for stim in ON_idx:
        #     PostStimPresses=[a for a in LP_idx if a>stim]
        #     if len(PostStimPresses)==0:
        #         continue
        #     FirstPress=PostStimPresses[0]
        #     Event_timestamps.append(FirstPress)
        #     Traces.append(Data_df.loc[FirstPress+Data_range[0]:FirstPress+Data_range[1],'dFF'].values)
        for each in LP_idx:
            Traces.append(
                Data_df.loc[each + Data_range[0]:each + Data_range[1], 'dFF'].values)  # THIS IS WHERE I ADDED THE SHIFT
            Event_timestamps.append(each)
    elif trigger == 'Post_Reinforcer_FirstLick':
        # 'HeadEntry' or 'Lick' depending on when the data was recorded...ISX vs TDT
        try:
            trigger = 'Lick'
        except:
            trigger = 'HeadEntry'

        Traces = []
        ON_idx = [i + 1 for i, (a, b) in
                  enumerate(zip(Data_df['Reinforcer'].values[:-1], Data_df['Reinforcer'].values[1:])) if b > a]
        LP_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        if len(LP_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for stim in ON_idx:
            PostReward1stHE = [a for a in LP_idx if a > stim]
            if len(PostReward1stHE) == 0:
                continue
            FirstHE = PostReward1stHE[0]
            Event_timestamps.append(FirstHE)
            Preceding_event_timestamps.append(stim - FirstHE - Data_range[0])
            Traces.append(Data_df.loc[FirstHE + Data_range[0]: FirstHE + Data_range[1], 'dFF'].values)
    elif trigger == 'HeadEntry':
        Traces = []
        ON_idx = [i + 1 for i, (a, b) in
                  enumerate(zip(Data_df['Reinforcer'].values[:-1], Data_df['Reinforcer'].values[1:])) if b > a]
        LP_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 2:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        if len(LP_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for stim in ON_idx:
            PostStimPresses = [a for a in LP_idx if a > stim]
            if len(PostStimPresses) == 0:
                continue
            FirstPress = PostStimPresses[0]
            Event_timestamps.append(FirstPress)
            Traces.append(Data_df.loc[FirstPress + Data_range[0]:FirstPress + Data_range[1], 'dFF'].values)
    elif trigger == 'HouseLight':  # This is a cheat because the TTLs for all except the houselight are 'ON' when they are recorded as off. So here the TTL we are getting id the 'light off'. take one second earlier to be properly aligned
        Traces = []
        ON_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 2:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for each in ON_idx:
            # grab -2 to 5 sec [-40:100]
            Traces.append(
                Data_df.loc[each + Data_range[0]:each + Data_range[1], 'dFF'].values)  # THIS IS WHERE I ADDED THE SHIFT
            Event_timestamps.append(each)
    elif trigger == 'FullTrial':  # This is a cheat because the TTLs for all except the houselight are 'ON' when they are recorded as off. So here the TTL we are getting id the 'light off'. take one second earlier to be properly aligned
        # First get the stimulus
        Stimulus = []
        trigger = 'HouseLight'
        ON_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 2:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for each in ON_idx:
            # grab -2 to 5 sec [-40:100]
            Stimulus.append(each)
        Event_timestamps.append(Stimulus)

        # Then the lever press (last before reward)
        LPs = []
        trigger = 'LeverPress'
        ON_idx = [i + 1 for i, (a, b) in
                  enumerate(zip(Data_df['Reinforcer'].values[:-1], Data_df['Reinforcer'].values[1:])) if b > a]
        LP_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 2:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        if len(LP_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for stim in ON_idx:
            PreRewardPresses = [a for a in LP_idx if a == stim]
            if len(PreRewardPresses) == 0:
                continue
            LastPress = PreRewardPresses[-1]
            LPs.append(LastPress)

        Event_timestamps.append(LPs)

        # First lick post reward
        Licks = []
        trigger = 'HeadEntry'
        Traces = []
        ON_idx = [i + 1 for i, (a, b) in
                  enumerate(zip(Data_df['Reinforcer'].values[:-1], Data_df['Reinforcer'].values[1:])) if b > a]
        LP_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        if len(LP_idx) < 1:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for stim in ON_idx:
            PostReward1stHE = [a for a in LP_idx if a > stim]
            if len(PostReward1stHE) == 0:
                continue
            FirstHE = PostReward1stHE[0]
            Traces.append(Data_df.loc[FirstHE + Data_range[0]: FirstHE + Data_range[1], 'dFF'].values)
            Licks.append(FirstHE)
        Event_timestamps.append(Licks)
    else:
        Traces = []
        ON_idx = [i + 1 for i, (a, b) in enumerate(zip(Data_df[trigger].values[:-1], Data_df[trigger].values[1:])) if
                  b > a]
        if len(ON_idx) < 2:
            return Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps
        for each in ON_idx:
            # grab -2 to 5 sec [-40:100]
            Traces.append(Data_df.loc[each + Data_range[0]:each + Data_range[1], 'dFF'].values)
            Event_timestamps.append(each)
    if Traces:  # check that Traces is not empty, which can happen depending on what you are asking for
        while len(Traces[-1]) != len(Traces[int(len(ON_idx)/2)]):  # this is to catch the case when the last trial was truncated
            Traces = Traces[:-1]
        while len(Traces[0]) != len(Traces[int(len(ON_idx)/2)]):  # this is to catch the case when the last trial was truncated
            Traces = Traces[1:]
    Norm_Traces = []
    for each in Traces:
        if Norm == 'Zscored_trial':
            Norm_Traces.append(sp.stats.zscore(each))
        elif Norm == 'Prestim':
            s = np.std(each[:40])
            m = abs(np.mean(each[:40]))
            Norm_Traces.append([(x - m) / s for x in each])  # the +5 is to guaranty the value stay positive
        elif Norm == 'Peak':
            Norm_Traces.append(each / peak)
    if Show_plot:
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(Norm_Traces, aspect= 'auto', cmap='bwr')  # ,norm=MidpointNormalize(midpoint=1.,vmin=0, vmax=4), cmap='bwr')
        plt.autoscale(axis='y', tight=True)

    return Norm_Traces, Event_timestamps, Preceding_event_timestamps, Following_event_timestamps, im

###############################################################################
# Plotting the Heat Map
###############################################################################

Home_dir = '/Volumes/SAMSUNG USB/TDT Fiber Photometry'
mice = ['4039'] 
#,'4032','4033','4034','4035','4036','4037','4038','4039','4041'
ALL_TEMP = []
grid = plt.GridSpec(1,1)
grid_Xpos = 0
grid_Ypos = 0
for i, mouse in enumerate(mice):
    mouse_dir = os.path.join(Home_dir, mouse)
    Dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
    Dates.sort()
    for date in Dates:
        date_dir = os.path.join(mouse_dir, date)
        file_to_open = os.path.join(date_dir, 'TDTlog_' + mouse + '_' + date + '.pkl')
        try:  # skip through folders that have not been processed
            with open(file_to_open, 'rb') as f:
                Data_df = pickle.load(f, encoding='latin1')
        except:
            continue
        trigger = 'HouseLight'  # 'LP_poststim', 'LP_prereward', 'Post_Reinforcer_FirstLick', 'LP_prereward_HighMag','First_of_Five_LP'
        Norm = 'Zscored_trial'
        start = -100
        stop = 100
        Data_range = [start, stop]
        Norm_Traces, ts, pre_ts, post_ts, im = Plot_TTL_aligned_heatmap(Data_df, trigger, grid, grid_Xpos, grid_Ypos,
                                                                        Data_range, Norm, Show_plot=True)
        plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], ['-5', '-4', '-3', '-2', '-1', '0', '1', '2','3','4','5'], size=16)

        plt.vlines(Data_range[0] * (-1), -0.5, len(Norm_Traces), color='b')
        plt.ylabel('Trials (n)', size=16)
        plt.xlabel('Time From Stimulus Onset (s)', size=16)
        plt.title('Mouse: ' + str(mouse) + 
                   ' | Date: ' + str(date) + 
                   ' | PF-heatmap\n', 
                   size=16, fontweight='bold')
        plt.colorbar(im, orientation='vertical')
        plt.show()