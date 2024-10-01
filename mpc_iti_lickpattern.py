
import pandas as pd
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


def read_med(file, finfo, var_cols, col_n='C:'):
    """ Function to read-write MED raw data to csv
    :param file: String. The name of the file: path/to/file_name.
    :param finfo: A list with the subject, session number in a list
    :param path_tidy: String. Where to store processed data frame.
    :param var_cols: The number of columns in PRINTCOLUMNS.
    :param col_n: String. Column to extract information. By default 'C:'.
    :return: dataframe
    """
    # names parameter takes into account a data frame with ncols columns
    ncols = var_cols + 2
    df = pd.read_csv(file, delim_whitespace=True, header=None,
                      skiprows=3, names=['x' + str(i) for i in range(1, ncols)])
    subj = df.x2[2]
    subjf = finfo[0]
    a = np.where(df.x1 == "0:")[0]
    col_pos = np.where(df.x1 == col_n)[0]
    # Check whether subj name in fname and inside file is the same,
    # otherwise break and check fname for errors
    if subj != subjf or len(col_pos) != 1:
        print(f"Subject name in file {finfo} is wrong; or")
        print(f'{col_n} is not unique, possible error in {finfo}. Dup file?')
        stop = True
    else:
        stop = False
    while not stop:
        if sum(a == col_pos + 1)==0:
            vC=pd.DataFrame(columns=['x'])
            return vC
        else:
            col_idx = int(np.where(a == col_pos + 1)[0])
            start = a[col_idx]
            if a[col_idx] == a[-1]:
                end = len(df.index)
            else:
                end = a[col_idx + 1] - 2
            vC = df[start:(end + 1)].reset_index(drop=True)
            vC = vC.drop(columns='x1').stack().reset_index(drop=True)  # dropna default True
            #FutureWarning: In a future version of pandas all arguments of DataFrame.drop 
            #except forres the argument 'labels' will be keyword-only
            vC = np.asarray(vC, dtype='float64')
            vC = pd.DataFrame({'vC': vC.astype(str)})
            reach = True
            if reach:
                return vC

def load_formattedMedPC_df(home_dir, mice):
    Columns=['Mouse', 'Date', 'Event', 'Timestamp']
    events=[
        'ActiveLever', 
        'Lick'
        ] 
    arrays=[ 
        'J:', 
        'O:'
        ]
    Med_log=pd.DataFrame(columns=Columns)
    
    for mouse in mice:
        directory=os.path.join(home_dir, mouse)
        files = [f for f in os.listdir(directory) 
                  if (os.path.isfile(os.path.join(directory, f)) and f[0]!='.')]
        
        for i,f in enumerate(files):
            print(f)
            date=f[:16]
            
            for event,col_n in zip(events, arrays):
                Timestamps=read_med(os.path.join(directory, f),[f[-9:-4], f[:10]], var_cols=5,col_n=col_n) 
                #Timestamps is a dataframe
                if len(Timestamps)!=0:
                    Timestamps.columns=['Timestamp']
                    Timestamps.at[:,'Mouse']=mouse
                    Timestamps.at[:,'Date']=date
                    Timestamps.at[:,'Event']=event
                    Med_log=pd.concat([Med_log,Timestamps],ignore_index=True) 

    #Format 'Timestamp' from str to float
    Med_log['Timestamp'] = Med_log['Timestamp'].astype(float)
    # Add a column called 'Session'
    for mouse in mice:
        mouse_log=Med_log.loc[np.equal(Med_log['Mouse'], mouse)]
        for i,day in enumerate(np.unique(mouse_log['Date'])):
            day_log=mouse_log.loc[np.equal(mouse_log['Date'], day)]
            Med_log.at[day_log.index, 'Session']=i
    return Med_log

# home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/ITI'
# mice = ['SD1-5','SD2-2', 'SD2-5','SD3-4', 'SD3-5', 'SD4-1','SD4-3']

# home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/ITI'
# mice = ['SD1-1','SD1-2','SD1-5','SD2-1','SD2-2','SD2-3','SD2-4','SD2-5','SD3-2','SD3-4','SD3-5','SD4-1','SD4-3','SD4-4','SD5-2']


# home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/3%_ITI'
# mice = ['SD1-5','SD2-3','SD4-1','SD2-5','SD2-2','SD3-4','SD3-5','SD4-4','SD4-3']

home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/ITI'
mice = ['SD1-1','SD1-2','SD1-4','SD1-5','SD2-1','SD2-2','SD2-3','SD2-4','SD2-5','SD3-1','SD3-2','SD3-4','SD3-5','SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD5-1','SD5-2','SD6-1','SD6-2']

sip_access = 10
#['F1', 'F2', 'F3', 'F4', 'M1', 'M2', 'M3', 'M4']
#, 'F2', 'F3', 'F4', 'M1', 'M2', 'M3', 'M4'
#'4032','4033','4034','4035','4036','4037','4038','4039','4041'
mpc_df = load_formattedMedPC_df(home_dir, mice)
mpc_df.to_csv('Med_log.csv', index = False)

lick_latency = {}
allsessionlicks = {}
for mouse in mice:
    directory=os.path.join(home_dir, mouse)
    files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and f[0]!='.')]
    files.sort()
    lickspermouse = []
        
    for i,f in enumerate(files):
        print(f)
        date=f[:16]
        print(date)
        lick_time = []
        lever_time = []
        latency = []
        
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,1] == date and mpc_df.iloc[i,2] == 'Lick':
                lick_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,1] == date and mpc_df.iloc[i,2] == 'ActiveLever':
                lever_time.append(mpc_df.iloc[i,3])
        
        k=0
        while k<len(lever_time):
            lickpersession = []
            for i in range (len(lick_time)):
                if lick_time[i] - lever_time[k] <= sip_access and lick_time[i] - lever_time[k] > 0:
                    lickpersession.append(lick_time[i]- lever_time[k])
            latency.append(lickpersession)
            for items in lickpersession:
                lickspermouse.append(items)
            k += 1
        lick_latency[str(mouse),str(date[0:10])]=latency
    
    allsessionlicks[mouse]=lickspermouse


########################### IDENTIFIERS ##############################
# females = ['SD1-1','SD1-2','SD1-5','SD3-2','SD3-4','SD3-5','SD5-2']
# males = ['SD2-1','SD2-2','SD2-3','SD2-4','SD2-5', 'SD4-1','SD4-3','SD4-4']
# highdrinker = ['SD1-5','SD2-2','SD2-3','SD2-4','SD2-5','SD3-5','SD4-1']
# lowdrinker = ['SD1-1','SD1-2','SD2-1','SD3-2','SD3-4','SD4-3','SD4-4','SD5-2']

########################### IDENTIFIERS ##############################
females = ['SD1-1','SD1-2','SD1-4','SD1-5', 'SD3-1','SD3-2','SD3-4','SD3-5','SD5-1','SD5-2' ]
males = ['SD2-1','SD2-2','SD2-3','SD2-4','SD2-5', 'SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD6-1','SD6-2']


########################### INDIVIDUAL ANIMAL HEATMAPS OF EACH SESSION ##############################
################ ALL ANIMALS ################
subplotnum = 3
for mouse in mice:
    mousedict = {}
    keys = []
    for subj, date in lick_latency:
        if mouse == subj:
            keys.append(date)
            mousedict[date] = lick_latency.get((str(subj),str(date)))
            onemouse = {}
            
            for date in mousedict:
                sessioncounts = []
                newdata = (mousedict[date])
                for i in range (0,len(newdata)):
                    counts, bin_edges = np.histogram(newdata[i], bins=sip_access, range=(0,sip_access))
                    sessioncounts.append(counts)
                onemouse[date] = sessioncounts
    fig, axs = plt.subplots(subplotnum, figsize=(10,12))
    plt.suptitle(f'Mouse: {mouse}\n')
    if len(onemouse) <= subplotnum:
        for i in range(0,len(onemouse)):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='plasma', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    if len(onemouse) > subplotnum:
        for i in range(0,subplotnum):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='plasma', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    #hid axes
    if len(onemouse) < subplotnum:
        all_axes = fig.get_axes()
        for ax in all_axes[len(onemouse):subplotnum]:
            for spines in ax.spines.values():
                spines.set_visible(False)
            ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    
################ FEMALES ################
for mouse in females:
    mousedict = {}
    keys = []
    for subj, date in lick_latency:
        if mouse == subj:
            keys.append(date)
            mousedict[date] = lick_latency.get((str(subj),str(date)))
            onemouse = {}
            
            for date in mousedict:
                sessioncounts = []
                newdata = (mousedict[date])
                for i in range (0,len(newdata)):
                    counts, bin_edges = np.histogram(newdata[i], bins=sip_access, range=(0,sip_access))
                    sessioncounts.append(counts)
                onemouse[date] = sessioncounts
    fig, axs = plt.subplots(7, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    if len(onemouse) <= 7:
        for i in range(0,len(onemouse)):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='RdPu', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    if len(onemouse) > 7:
        for i in range(0,7):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='RdPu', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    #hid axes
    if len(onemouse) < 7:
        all_axes = fig.get_axes()
        for ax in all_axes[len(onemouse):7]:
            for spines in ax.spines.values():
                spines.set_visible(False)
            ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)
    fig.tight_layout()

################ MALES ################
for mouse in males:
    mousedict = {}
    keys = []
    for subj, date in lick_latency:
        if mouse == subj:
            keys.append(date)
            mousedict[date] = lick_latency.get((str(subj),str(date)))
            onemouse = {}
            
            for date in mousedict:
                sessioncounts = []
                newdata = (mousedict[date])
                for i in range (0,len(newdata)):
                    counts, bin_edges = np.histogram(newdata[i], bins=sip_access, range=(0,sip_access))
                    sessioncounts.append(counts)
                onemouse[date] = sessioncounts
    fig, axs = plt.subplots(7, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    if len(onemouse) <= 7:
        for i in range(0,len(onemouse)):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='PuBu', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    if len(onemouse) > 7:
        for i in range(0,7):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='PuBu', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    #hid axes
    if len(onemouse) < 7:
        all_axes = fig.get_axes()
        for ax in all_axes[len(onemouse):7]:
            for spines in ax.spines.values():
                spines.set_visible(False)
            ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    
################ LOW DRINKERS ################
for mouse in lowdrinker:
    mousedict = {}
    keys = []
    for subj, date in lick_latency:
        if mouse == subj:
            keys.append(date)
            mousedict[date] = lick_latency.get((str(subj),str(date)))
            onemouse = {}
            
            for date in mousedict:
                sessioncounts = []
                newdata = (mousedict[date])
                for i in range (0,len(newdata)):
                    counts, bin_edges = np.histogram(newdata[i], bins=sip_access, range=(0,sip_access))
                    sessioncounts.append(counts)
                onemouse[date] = sessioncounts
    fig, axs = plt.subplots(7, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    if len(onemouse) <= 7:
        for i in range(0,len(onemouse)):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='YlGn', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    if len(onemouse) > 7:
        for i in range(0,7):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='YlGn', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    #hid axes
    if len(onemouse) < 7:
        all_axes = fig.get_axes()
        for ax in all_axes[len(onemouse):7]:
            for spines in ax.spines.values():
                spines.set_visible(False)
            ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    
################ HIGH DRINKER ################
for mouse in highdrinker:
    mousedict = {}
    keys = []
    for subj, date in lick_latency:
        if mouse == subj:
            keys.append(date)
            mousedict[date] = lick_latency.get((str(subj),str(date)))
            onemouse = {}
            
            for date in mousedict:
                sessioncounts = []
                newdata = (mousedict[date])
                for i in range (0,len(newdata)):
                    counts, bin_edges = np.histogram(newdata[i], bins=sip_access, range=(0,sip_access))
                    sessioncounts.append(counts)
                onemouse[date] = sessioncounts
    fig, axs = plt.subplots(7, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    if len(onemouse) <= 7:
        for i in range(0,len(onemouse)):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='OrRd', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    if len(onemouse) > 7:
        for i in range(0,7):
            dicttodf = pd.DataFrame.from_dict(onemouse[keys[i]])
            sns.heatmap(dicttodf, cmap='OrRd', vmin=0, vmax=10, ax=axs[i])
            axs[i].set_title(keys[i])
            axs[i].set_xlabel('Sipper Access Time')
    #hid axes
    if len(onemouse) < 7:
        all_axes = fig.get_axes()
        for ax in all_axes[len(onemouse):7]:
            for spines in ax.spines.values():
                spines.set_visible(False)
            ax.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False)
    fig.tight_layout()


########################### GROUPED BY CONSECUTIVE TRIALS ##############################
################ ALL ANIMALS ################







########################### INDIVIDUAL ANIMAL HEATMAPS OF ALL SESSIONS ##############################
################ ALL ANIMALS ################
for mouse in mice:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
            
    individmouse = pd.DataFrame()
    for key, values in eachdatelicks.items():
        counts, bin_edges = np.histogram(values, bins=sip_access, range=(0,sip_access))
        individmouse[key]=counts
    individmouse = individmouse.swapaxes("index", "columns") 
    fig = plt.figure()
    sns.heatmap(individmouse, cmap='plasma', square=True, vmin=0, vmax=30)
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ FEMALES ################
for mouse in females:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
            
    individmouse = pd.DataFrame()
    for key, values in eachdatelicks.items():
        counts, bin_edges = np.histogram(values, bins=sip_access, range=(0,sip_access))
        individmouse[key]=counts
    individmouse = individmouse.swapaxes("index", "columns") 
    fig = plt.figure()
    sns.heatmap(individmouse, cmap='RdPu', square=True, vmin=0, vmax=30)
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ MALES ################
for mouse in males:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
            
    individmouse = pd.DataFrame()
    for key, values in eachdatelicks.items():
        counts, bin_edges = np.histogram(values, bins=sip_access, range=(0,sip_access))
        individmouse[key]=counts
    individmouse = individmouse.swapaxes("index", "columns") 
    fig = plt.figure()
    sns.heatmap(individmouse, cmap='PuBu', square=True, vmin=0, vmax=30)
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')
    
################ LOWDRINKER ################
for mouse in lowdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
            
    individmouse = pd.DataFrame()
    for key, values in eachdatelicks.items():
        counts, bin_edges = np.histogram(values, bins=sip_access, range=(0,sip_access))
        individmouse[key]=counts
    individmouse = individmouse.swapaxes("index", "columns") 
    fig = plt.figure()
    sns.heatmap(individmouse, cmap='YlGn', square=True, vmin=0, vmax=30)
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ HIGH DRINKER ################
for mouse in highdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
            
    individmouse = pd.DataFrame()
    for key, values in eachdatelicks.items():
        counts, bin_edges = np.histogram(values, bins=sip_access, range=(0,sip_access))
        individmouse[key]=counts
    individmouse = individmouse.swapaxes("index", "columns") 
    fig = plt.figure()
    sns.heatmap(individmouse, cmap='OrRd', square=True, vmin=0, vmax=30)
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')
    
########################### INDIVIDUAL ANIMAL SURVIVAL PLOT OF ALL SESSIONS ##############################
################ ALL ANIMALS ################
for mouse in mice:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.ecdfplot(data=eachdatelicks, palette='plasma')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')
    
################ FEMALES ################
for mouse in females:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.ecdfplot(data=eachdatelicks, palette='RdPu')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')   

################ MALES ################
for mouse in males:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.ecdfplot(data=eachdatelicks, palette='PuBu')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ LOW DRINKERS ################
for mouse in lowdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.ecdfplot(data=eachdatelicks, palette='YlGn')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ HIGH DRINKERS ################
for mouse in highdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.ecdfplot(data=eachdatelicks, palette='OrRd')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

########################### INDIVIDUAL ANIMAL PROBABILITY DENSITY CURVE OF ALL SESSIONS ##############################
################ ALL ANIMALS ################
for mouse in mice:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.kdeplot(data=eachdatelicks, palette='plasma')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ FEMALES ################
for mouse in females:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.kdeplot(data=eachdatelicks, palette='RdPu')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')

################ MALES ################
for mouse in males:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.kdeplot(data=eachdatelicks, palette='PuBu')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')
    
    
################ LOW DRINKERS ################
for mouse in lowdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.kdeplot(data=eachdatelicks, palette='YlGn')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')


################ HIGH DRINKERS ################
for mouse in highdrinker:
    mouselickdata = pd.DataFrame()
    eachdatelicks ={}
    for mouseid, date in lick_latency:
        workworklist=[]
        if mouseid == mouse:
            workinglist = lick_latency.get((str(mouseid),str(date)))
            workworklist.append(workinglist)
            for i in range(0,len(workworklist)):
                flat_list = [num for sublist in workworklist[i] for num in sublist]
            eachdatelicks[date] = (flat_list)
    fig = plt.figure()
    sns.kdeplot(data=eachdatelicks, palette='OrRd')
    plt.xlabel('Sipper Access Time')
    plt.title(f'Mouse: {mouse}')


########################### ALL ANIMAL HEATMAPS OF ALL SESSIONS ##############################
################ ALL ANIMALS ################
allsessionpd = pd.DataFrame(columns = mice)
for mouse in mice:
    counts, bin_edges = np.histogram(allsessionlicks.get(mouse), bins=sip_access, range=(0,sip_access))
    allsessionpd[mouse]=counts
allsessionpd = allsessionpd.swapaxes("index", "columns") 
sns.heatmap(allsessionpd, vmin=0, vmax=300, cmap='plasma')
plt.xlabel('Sipper Access Time')

################ FEMALES ################
allsessionpd = pd.DataFrame(columns = females)
for mouse in females:
    counts, bin_edges = np.histogram(allsessionlicks.get(mouse), bins=sip_access, range=(0,sip_access))
    allsessionpd[mouse]=counts
allsessionpd = allsessionpd.swapaxes("index", "columns") 
sns.heatmap(allsessionpd, vmin=0, vmax=300, cmap='RdPu')
plt.xlabel('Sipper Access Time')

################ MALES ################
allsessionpd = pd.DataFrame(columns = males)
for mouse in males:
    counts, bin_edges = np.histogram(allsessionlicks.get(mouse), bins=sip_access, range=(0,sip_access))
    allsessionpd[mouse]=counts
allsessionpd = allsessionpd.swapaxes("index", "columns") 
sns.heatmap(allsessionpd, vmin=0, vmax=300, cmap='PuBu')
plt.xlabel('Sipper Access Time')

################ HIGH DRINKERS ################
allsessionpd = pd.DataFrame(columns = highdrinker)
for mouse in highdrinker:
    counts, bin_edges = np.histogram(allsessionlicks.get(mouse), bins=sip_access, range=(0,sip_access))
    allsessionpd[mouse]=counts
allsessionpd = allsessionpd.swapaxes("index", "columns") 
sns.heatmap(allsessionpd, vmin=0, vmax=500, cmap='OrRd')
plt.xlabel('Sipper Access Time')


################ LOW DRINKERS ################
allsessionpd = pd.DataFrame(columns = lowdrinker)
for mouse in lowdrinker:
    counts, bin_edges = np.histogram(allsessionlicks.get(mouse), bins=sip_access, range=(0,sip_access))
    allsessionpd[mouse]=counts
allsessionpd = allsessionpd.swapaxes("index", "columns") 
sns.heatmap(allsessionpd, vmin=0, vmax=500, cmap='YlGn')
plt.xlabel('Sipper Access Time')






## raster plotting throughout days
mouse_df = pd.DataFrame.from_dict(lick_latency, orient='index')
allkeys = list(lick_latency.keys())

for mouse in mice:
    plotting={}
    for mouse_date in lick_latency:
        title=str(mouse_date)
        if title[2:7] == mouse:
            plotting[title[11:21]]=lick_latency[mouse_date]
            continue
    
    fig, axs = plt.subplots(len(plotting),1, figsize=(10,10), sharex=True, sharey=False)
    plt.suptitle(f'Mouse: {mouse}')
  
    for i in range(0,len(plotting)):
        raster_data=[]
        each_date = list(plotting.keys())
        raster_data= plotting[each_date[i]]
        axs[i].eventplot(raster_data)
        axs[i].set(xlim=(-0.5,10.5),xticks=np.arange(0,11))
        axs[i].axvline(x = 0, color = 'grey', linestyle='dashed')
        axs[i].set_title(each_date[i], fontsize=12)
    plt.xlabel('Sipper Access Time', fontsize=12)
    fig.text(0.005, 0.5, 'Trials Per Session', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout()



##### for specific raster plots #####
#mice = ['M4']
for mouse in mice:
    plotting={}
    for mouse_date in lick_latency:
        title=str(mouse_date)
        if title[2:7] == mouse:
            plotting[title[8:18]]=lick_latency[mouse_date]
            continue
            
    each_date = list(plotting.keys())
    raster_data= plotting[each_date[-1]]
    plt.eventplot(raster_data)
    #plt.set(xlim=(-0.5,10.5),xticks=np.arange(0,11))
    plt.axvline(x = 0, color = 'grey', linestyle='dashed')
    
    plt.savefig('/Users/kristineyoon/Desktop/rasterplot2.pdf',dpi=1200, transparent = True)


### for kde plots ###
high_drink_list = []
low_drink_list = []
highdrinkers = ['F3','F4', 'M1', 'M3','M4']
lowdrinkers = ['F1','F2','M2']
for animal in highdrinkers:
    high_drink_list += allsessionlicks[animal]
for animal in lowdrinkers:
    low_drink_list += allsessionlicks[animal]
    
sns.kdeplot(high_drink_list, multiple='stack', clip=[0,12])
sns.kdeplot(low_drink_list, multiple='stack', clip=[0,12])
plt.savefig('/Users/kristineyoon/Desktop/kdeplot.pdf',dpi=1200, transparent = True)

### for kde plots ###
high_drink_list = []
low_drink_list = []
highdrinkers = ['F3','F4', 'M1', 'M3','M4']
lowdrinkers = ['F1','F2','M2']
for animal in highdrinkers:
    high_drink_list += allsessionlicks[animal]
for animal in lowdrinkers:
    low_drink_list += allsessionlicks[animal]
    
sns.kdeplot(high_drink_list, multiple='stack', clip=[0,12])
sns.kdeplot(low_drink_list, multiple='stack', clip=[0,12])
plt.savefig('/Users/kristineyoon/Desktop/kdeplot.pdf',dpi=1200, transparent = True)


## for histogram
for mouseid in allsessionlicks:
    fig = plt.figure()
    n, bins, patches = plt.hist(x=allsessionlicks[mouseid], bins=10, color='#0504aa', alpha=0.5, rwidth=0.5)
    plt.xlabel('Sipper Access Time')
    plt.ylabel('All Session Frequency')
    plt.title(f'Mouse: {mouseid}')
