
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
        'Lick',
        'No Response',
        'Cue'
        ] 
    arrays=[ 
        'J:', 
        'O:',
        'P:',
        'L:'
        ]
    Med_log=pd.DataFrame(columns=Columns)
    
    for mouse in mice:
        directory=os.path.join(home_dir, mouse)
        files = [f for f in os.listdir(directory) 
                  if (os.path.isfile(os.path.join(directory, f)) and f[0]!='.')]
        files.sort()
        for i,f in enumerate(files):
            print(f)
            date=f[:16]
            
            for event,col_n in zip(events, arrays):
                Timestamps=read_med(os.path.join(directory, f),[f[-8:-4], f[:10]], var_cols=5,col_n=col_n) 
                #Timestamps is a dataframe
                if len(Timestamps)!=0:
                    Timestamps.columns=['Timestamp']
                    Timestamps.at[:,'Mouse']=mouse
                    Timestamps.at[:,'Date']=date
                    Timestamps.at[:,'Event']=event
                    Med_log=pd.concat([Med_log,Timestamps],ignore_index=True) 
                elif len(Timestamps)== 0:
                    Timestamps.columns=['Timestamp']
                    Timestamps.at[:,'Timestamp']= [0]
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


home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/Med Associates/'

mice = ['7647','7649','7650','7651','7654','7655','7657','7658','7661','7663','7664','7666']
sip_access = 10
mpc_df = load_formattedMedPC_df(home_dir, mice)
mpc_df.to_csv('Med_log.csv', index = False)


########################### IDENTIFIERS ##############################
halo = ['7647','7649','7650','7651','7654','7655','7657','7658']
ctrl = ['7661','7663','7664','7666']


##################################################################################
################### ALL LICK LATENCY BY TRIALS IN EACH SESSION ###################
##################################################################################
leverlickpercent_df=pd.DataFrame(columns=['Mouse','Session','Levers','Licks'])
row = 0
lick_latency_dict = {}
lick_latency_dict = {}
lever_latency_dict = {}
lever_latency_dict1 = {}
maxsession = int(max(mpc_df['Session'].values))
for j in range (0,int(max(mpc_df['Session'].values))+1):
    lickspermouse = []
    for mouse in mice:
        cue_time = []
        lick_time = []
        lever_time = []
        leverlatency = []
        licklatency = {}
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Cue':
                cue_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Lick':
                lick_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'ActiveLever':
                lever_time.append(mpc_df.iloc[i,3])
        print (str('Mouse: ' ) + str(mouse) + str(' Session: ') + str(j))
        leverlickpercent_df.at[row,'Mouse'] = str(mouse)
        leverlickpercent_df.at[row,'Session'] = j
        leverlickpercent_df.at[row,'Levers'] = (len(lever_time))
        ################### AVERAGE LATENCY FROM CUE TO LEVER PRESS ###################
        for i in range (len(cue_time)):
            for k in range (len(lever_time)):
                if lever_time[k] - cue_time[i] <= 20 and lever_time[k] - cue_time[i] > 0:
                    leverlatency.append(lever_time[k] - cue_time[i])
                    
        if len(leverlatency) != 0:
               avg_latency = sum(leverlatency)/len(leverlatency)
        lever_latency_dict[str(mouse),str(j)]=avg_latency
        lever_latency_dict1[str(mouse),str(j)]=leverlatency
        
        ###################### LATENCY FROM LEVER PRESS TO LICK ######################
        for k in range (0,len(lever_time)):
            lickpertrial = []
            for l in range (len(lick_time)):
                if lick_time[l] - lever_time[k] <= sip_access and lick_time[l] - lever_time[k] > 0:
                    lickpertrial.append(lick_time[l]- lever_time[k])
            licklatency[k] = lickpertrial
        licktrial = 0
        for i in range(len(licklatency)):
            if len(licklatency[i])!= 0:
                licktrial = licktrial + 1
        leverlickpercent_df.at[row,'Licks'] = licktrial
        lick_latency_dict[str(mouse),str(j)]=licklatency    
        row = row+1
    
leverdf=pd.DataFrame.from_dict(lever_latency_dict1, orient='index')

########################## LICK BOUTS ##########################
tracklicks = {}
for j in range (0,int(max(mpc_df['Session'].values))+1):
    lickspermouse = []
    for mouse in mice:
        lick_time = []
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Lick':
                lick_time.append(mpc_df.iloc[i,3])
        print (str('Mouse: ' ) + str(mouse) + str(' Session: ') + str(j))
        tracklicks[str(mouse),str(j)] = lick_time
#############################

lickbout_dict ={}
intertriallick_dict = {}
#### Interlick Interval ####
for subj, session in tracklicks:
    # subj = 'SD4-2'
    # session = 1
    lick_time = tracklicks[str(subj),str(session)]
    interlickint_all = [lick_time[i + 1] - lick_time[i] for i in range(len(lick_time)-1)]
    intertriallick_dict[str(subj),str(session)] = interlickint_all
    #### Lick Bout ####
    boutbreaks = np.array(interlickint_all) > 1
    boutindex = np.where(boutbreaks == True)
    boutindex = boutindex[0]
    start = 0
    boutall = []
    if len(boutindex) > 0:
        for i in range(len(boutindex)):
            end = boutindex[i]
            boutall.append(lick_time[start:end])
            start = boutindex[i]+1
    boutall.append(lick_time[start:-1])
    lickbout_dict[str(subj),str(session)] = boutall
        
lickbout_edit = {}
for subj, session in lickbout_dict:
    newlickbout = []
    for num in range(0, len(lickbout_dict[subj,session])):
        if len(lickbout_dict[subj,session][num]) >= 3:
            newlickbout.append(lickbout_dict[subj,session][num])
    lickbout_edit[str(subj),str(session)]=newlickbout

        
for subj in mice:
    fig,axs = plt.subplots(8, figsize=(8,8))
    plt.suptitle(f'Mouse: {subj}\n')
    for j in range (0,int(max(mpc_df['Session'].values))+1):
        axs[j].eventplot(tracklicks[str(subj),str(j)], lineoffsets=0)
        #axs[j].eventplot(lickbouts_justlicks_dict[str(subj),str(j)],lineoffsets=0, linelengths=0.75, colors='orange')
        axs[j].eventplot(lickbout_edit[str(subj),str(j)],lineoffsets=0, linelengths=0.5, colors='yellow')
        axs[j].set_xlim(0,3600)
        
       
bouts_df = pd.DataFrame()
allbouts = pd.DataFrame()
for mouse, session in lickbout_edit:
    bouts_df.at[mouse,'Session'] = session
    boutlicks = 0
    for i in range(len(lickbout_edit[str(mouse),str(session)])):
        bouts_df.at[mouse,f'Bout {i}'] = len(lickbout_edit[str(mouse),str(session)][i])
        boutlicks = boutlicks + len(lickbout_edit[str(mouse),str(session)][i])
    allbouts.at[mouse,f'Session{session}'] = boutlicks

#########################################
### Lick Rates
fig, axs = plt.subplots(2, figsize=(4,6), sharex=True, sharey=True)
alllicklengths=[]
for mouse in halo:
    for subj,session in lick_latency_dict:
        if subj == mouse:
            licklength = {}
            for i in range(len(lick_latency_dict[subj,session])):
                licks = len(lick_latency_dict[subj,session][i])
                licklength[i]=licks
            alllicklengths.append(licklength)
maxtrial = 1
for dictionary in alllicklengths:
    if maxtrial < len(dictionary):
        maxtrial = len(dictionary)
dfhalo = pd.DataFrame(alllicklengths)
dfhalo1 = np.where(dfhalo==0, np.nan, dfhalo)
for i in range(len(dfhalo1)):
    axs[0].scatter(np.arange(0,maxtrial,1), dfhalo1[i],c='#108080',alpha=0.5) 

alllicklengths=[]
for mouse in ctrl:
    for subj,session in lick_latency_dict:
        if subj == mouse:
            licklength = {}
            for i in range(len(lick_latency_dict[subj,session])):
                licks = len(lick_latency_dict[subj,session][i])
                licklength[i]=licks
            alllicklengths.append(licklength)
maxtrial = 1
for dictionary in alllicklengths:
    if maxtrial < len(dictionary):
        maxtrial = len(dictionary)
dfctrl = pd.DataFrame(alllicklengths)
dfctrl1 = np.where(dfctrl==0, np.nan, dfctrl)
for i in range(len(dfctrl1)):
    axs[1].scatter(np.arange(0,maxtrial,1), dfctrl1[i],c='#DE004A',alpha=0.5) 
plt.xlabel('Nth Trial in Session', size=12)
plt.ylabel('Lick Rate Per Trial', size=12)
plt.savefig('/Users/kristineyoon/Documents/lickrate10s.pdf', transparent=True)



### KDE Plots
fig, axs = plt.subplots(2, figsize=(10.08,16), sharey=True, sharex=True)
sns.heatmap(dfctrl1, cmap=sns.color_palette("light:#DE004A", as_cmap=True),square=True, ax=axs[0])
sns.heatmap(dfhalo1, cmap=sns.color_palette("light:#108080", as_cmap=True),square=True, ax=axs[1])
plt.savefig('/Users/kristineyoon/Documents/lickratehm10s.pdf', transparent=True)

from scipy.stats import gaussian_kde
from scipy.integrate import quad

# Estimate the KDE
kde_halo = gaussian_kde(dfhalo1)
kde_ctrl = gaussian_kde(dfctrl1)
# Define integration bounds
lower_bound = 0
upper_bound = 5

# Define the KDE function
kde_func_halo = lambda x: kde_halo(x)[0]
kde_func_ctrl = lambda x: kde_ctrl(x)[0]

# Perform numerical integration
area_halo, _ = quad(kde_func_halo, lower_bound, upper_bound)
area_ctrl, _ = quad(kde_func_ctrl, lower_bound, upper_bound)
print("Halo: Area under the KDE curve:", area_halo)
print("Ctrl: Area under the KDE curve:", area_ctrl)


### Individual Traces

for mouse in halo:
    fig, axs = plt.subplots(8, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for subj,session in lick_latency_dict:
        if subj == mouse:
            
            #heatmapdf = pd.DataFrame.from_dict(lick_latency_dict[subj,session])
            mouseheatmap=[]
            for i in range(len(lick_latency_dict[subj,session])):
                counts, bin_edges = np.histogram(lick_latency_dict[subj,session][i], bins=sip_access, range=(0,sip_access))
                mouseheatmap.append(counts)

            sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#DE004A"), vmin=0, vmax=10, ax=axs[int(session)])
            axs[int(session)].set_title(f'Session {session}')
            axs[int(session)].set_xlabel('Sipper Access Time')
    fig.tight_layout()


for mouse in ctrl:
    fig, axs = plt.subplots(8, figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for subj,session in lick_latency_dict:
        if subj == mouse:
            
            #heatmapdf = pd.DataFrame.from_dict(lick_latency_dict[subj,session])
            mouseheatmap=[]
            for i in range(len(lick_latency_dict[subj,session])):
                counts, bin_edges = np.histogram(lick_latency_dict[subj,session][i], bins=sip_access, range=(0,sip_access))
                mouseheatmap.append(counts)

            sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#108080"), vmin=0, vmax=10, ax=axs[int(session)])
            axs[int(session)].set_title(f'Session {session}')
            axs[int(session)].set_xlabel('Sipper Access Time')
    fig.tight_layout()