
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
                Timestamps=read_med(os.path.join(directory, f),[f[-9:-4], f[:10]], var_cols=5,col_n=col_n) 
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


home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/'
path = 'ITI'
allfiles = os.path.join(home_dir, path)
mice = ['SD1-1','SD1-2','SD1-4','SD1-5','SD2-1','SD2-2','SD2-3','SD2-4','SD2-5','SD3-1','SD3-2','SD3-4','SD3-5','SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD5-1','SD5-2','SD6-1','SD6-2']

sip_access = 10
mpc_df = load_formattedMedPC_df(allfiles, mice)
mpc_df.to_csv('Med_log.csv', index = False)


########################### IDENTIFIERS ##############################
females = ['SD1-1','SD1-2','SD1-4','SD1-5', 'SD3-1','SD3-2','SD3-4','SD3-5','SD5-1','SD5-2' ]
males = ['SD2-1','SD2-2','SD2-3','SD2-4','SD2-5', 'SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD6-1','SD6-2']


##################################################################################
################### ALL LICK LATENCY BY TRIALS IN EACH SESSION ###################
##################################################################################
lickpattern_df=pd.DataFrame()
lick_latency_dict = {}
lever_latency_dict = {}
by_session = {}
maxsession = int(max(mpc_df['Session'].values))
for j in range (0,int(max(mpc_df['Session'].values))+1):
    lickspermouse = []
    for mouse in mice:
        cue_time = []
        lick_time = []
        lever_time = []
        leverlatency = []
        licklatency = []
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Cue':
                cue_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Lick':
                lick_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'ActiveLever':
                lever_time.append(mpc_df.iloc[i,3])
        print (str('Mouse: ' ) + str(mouse) + str(' Session: ') + str(j))

        ################### AVERAGE LATENCY FROM CUE TO LEVER PRESS ###################
        for i in range (len(cue_time)):
            for k in range (len(lever_time)):
                if lever_time[k] - cue_time[i] <= 20 and lever_time[k] - cue_time[i] > 0:
                    leverlatency.append(lever_time[k] - cue_time[i])
                    
        if len(leverlatency) != 0:
               avg_latency = sum(leverlatency)/len(leverlatency)
        lever_latency_dict[str(mouse),str(j)]=avg_latency
        
        ###################### LATENCY FROM LEVER PRESS TO LICK ######################
        for k in range (0,len(lever_time)):
            lickpertrial = []
            for i in range (len(lick_time)):
                if lick_time[i] - lever_time[k] <= sip_access and lick_time[i] - lever_time[k] > 0:
                    lickpertrial.append(lick_time[i]- lever_time[k])
            licklatency.append(lickpertrial)
            for items in lickpertrial:
                lickspermouse.append(items)
        lick_latency_dict[str(mouse),str(j)]=licklatency    
        
        ###################### INTERLICK INTERVAL ######################
        interlickint_all = [lick_time[i + 1] - lick_time[i] for i in range(len(lick_time)-1)]
        interlickint_bytrial = [k for k in interlickint_all if k < 10]
        if len(interlickint_bytrial) > 0:
            interlickint_bytrial_avg = sum(interlickint_bytrial)/len(interlickint_bytrial)
        else:
            interlickint_bytrial_avg = 0
        lickpattern_df.at[f'Session {j} Interlick Interval',mouse]=interlickint_bytrial_avg
        
        ###################### LONGEST LICK BOUT ######################
        boutbreaks = np.array(interlickint_all) > 1
        boutindex = np.where(boutbreaks == True)
        boutindex = boutindex[0]
        boutlength = [0]
        if len(boutindex) > 0:
            boutall = []
            start = lick_time[0]
            for i in range(len(boutindex)):
                boutall.append([start,lick_time[boutindex[i]]])
                start = lick_time[boutindex[i]+1]
            
            for beg,end in boutall:
                boutlength.append(end-beg)
        lickpattern_df.at[f'Session {j} Longest Lick Bout Time',mouse] = max(boutlength)
        
        boutcount = [0]
        for beg,end in boutall:
            boutindicate = []
            for tstamp in lick_time:
                if tstamp >= beg and tstamp <= end:
                    boutindicate.append(tstamp)
            boutcount.append(len(boutindicate))
       
        lickpattern_df.at[f'Session {j} Longest Lick Bout #',mouse] = max(boutcount)
    by_session[j]=lickspermouse

lickpattern_df.to_csv('/Users/kristineyoon/Documents/lickpattern.csv')
################### AVERAGE LATENCY FROM CUE TO LEVER PRESS ###################
leverdf_female=pd.DataFrame()
leverdf_male=pd.DataFrame()
for mouse, session in lever_latency_dict:
    for i in range (maxsession+1):
        femalelever=[]
        malelever=[]
        if mouse in females:
            if int(session) == i:
                femalelever.append(lever_latency_dict[mouse,session])
                leverdf_female.at[i,mouse] = lever_latency_dict[mouse,session]
        if mouse in males:
            if int(session) == i:
                malelever.append(lever_latency_dict[mouse,session])
                leverdf_male.at[i,mouse] = lever_latency_dict[mouse,session]
leverdf_female.to_csv('/Users/kristineyoon/Documents/leverdf_female.csv')
leverdf_male.to_csv('/Users/kristineyoon/Documents/leverdf_male.csv')
###################### FIRST LICK LATENCY FROM LEVERPRESS ######################
flickdf_female=pd.DataFrame()
flickdf_male=pd.DataFrame()
for mouse, session in lick_latency_dict:
    for i in range (maxsession+1):
        if mouse in females:
            if int(session) == i:
                femaleflick=lick_latency_dict.get((str(mouse),str(session)))
                allflick=[]
                for k in range(len(femaleflick)):
                    if len(femaleflick[k]) > 0:
                        allflick.append(femaleflick[k][0])
                if len(allflick) > 0:
                    avgflick=sum(allflick)/len(allflick)
                    flickdf_female.at[i,mouse]=avgflick
        if mouse in males:
            if int(session) == i:
                maleflick=lick_latency_dict.get((str(mouse),str(session)))
                allflick=[]
                for k in range(len(maleflick)):
                    if len(maleflick[k]) > 0:
                        allflick.append(maleflick[k][0])
                if len(allflick) > 0:
                    avgflick=sum(allflick)/len(allflick)
                    flickdf_male.at[i,mouse]=avgflick
flickdf_female.to_csv('/Users/kristineyoon/Documents/flickdf_female.csv')
flickdf_male.to_csv('/Users/kristineyoon/Documents/flickdf_male.csv')
###################### GRAPHING LATENCY FROM LEVER PRESS TO LICK ######################
fig = plt.figure()
sns.kdeplot(data=by_session, palette='magma')
plt.xlabel('Sipper Access Time')
plt.title('All Mice')
    
##################################### FEMALES #####################################
lick_latency_female = {}
by_session_female = {}
session_num=0
for mouse, session in lick_latency_dict:
    licksperfemale=[]
    if mouse in females:
        lick_latency_female[mouse,session]=lick_latency_dict[mouse,session]

for i in range(0,len(by_session)):
    lickspermouse=[]
    for mouse, session in lick_latency_female:  
        if session == str(i):
            for lists in lick_latency_female[mouse,session]:
                for items in lists:
                    lickspermouse.append(items)
    by_session_female[i]=lickspermouse

plt.figure(figsize=(16, 12), dpi=100)
ax = sns.kdeplot(data=by_session_female, palette='light:#DE004A')
plt.xlabel('Sipper Access Time (sec)', size = 20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.ylabel('Density', size = 20, labelpad=5)
plt.title('Females', size = 20)
plt.setp(ax.get_legend().get_texts(), fontsize='20')  
plt.setp(ax.get_legend().get_title(), fontsize='20')
plt.xlim(0,sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysessionfem.pdf', transparent=True)

###################################### MALES ######################################
lick_latency_male = {}
by_session_male = {}
session_num=0
for mouse, session in lick_latency_dict:
    lickspermale=[]
    if mouse in males:
        lick_latency_male[mouse,session]=lick_latency_dict[mouse,session]

for i in range(0,len(by_session)):
    lickspermouse=[]
    for mouse, session in lick_latency_male:  
        if session == str(i):
            for lists in lick_latency_male[mouse,session]:
                for items in lists:
                    lickspermouse.append(items)
    by_session_male[i]=lickspermouse

plt.figure(figsize=(16, 12), dpi=100)
ax = sns.kdeplot(data=by_session_male, palette='light:#108080')
plt.xlabel('Sipper Access Time (sec)', size = 20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.ylabel('Density', size = 20, labelpad=5)
plt.setp(ax.get_legend().get_texts(), fontsize='20')  
plt.setp(ax.get_legend().get_title(), fontsize='20')
plt.title('Males', size = 20)
plt.xlim(0,sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysessionfmal.pdf', transparent=True)


from scipy.stats import gaussian_kde
from scipy.integrate import quad

# Define integration bounds
lower_bound = 0
upper_bound = 4
# Estimate the KDE
for i in range (len(by_session_female)):
    kde_fe = gaussian_kde(by_session_female[i])
    kde_func_fe = lambda x: kde_fe(x)[0]
    area_fe, _ = quad(kde_func_fe, lower_bound, upper_bound)
    print(f"Female at Session {i} AUC:", area_fe)

for i in range (len(by_session_male)):
    kde_ma = gaussian_kde(by_session_male[i])
    kde_func_male = lambda x: kde_ma(x)[0]
    area_male, _ = quad(kde_func_male, lower_bound, upper_bound)
    print(f"Male at Session {i} AUC:", area_male)

#################################################################################
######################### HEATMAP OF LICKING BY ANIMALS #########################
#################################################################################
eachanimalcounts =[]
for subj in females:
    sessioncounts = []
    for mouse,session in lick_latency_female:
        if mouse == subj:
            sessioncounts.append(lick_latency_female[mouse,session])
    session_cts = [item for sublist in sessioncounts for item in sublist]
    session_cts1 = [item for sublist in session_cts for item in sublist]
    counts, bin_edges = np.histogram(session_cts1, bins=sip_access, range=(0,sip_access))
    eachanimalcounts.append(counts)
plt.figure(figsize=(16, 12), dpi=100)
ax = sns.heatmap(eachanimalcounts, cmap=sns.color_palette("light:#DE004A", as_cmap=True),square=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Sipper Access Time (sec)', size = 20)
plt.ylabel('Mouse Number', size = 20)
plt.title('Females', size = 20)
plt.savefig('/Users/kristineyoon/Documents/heatmapbyfem.pdf', transparent=True)

eachanimalcounts =[]
for subj in males:
    sessioncounts = []
    for mouse,session in lick_latency_male:
        if mouse == subj:
            sessioncounts.append(lick_latency_male[mouse,session])
    session_cts = [item for sublist in sessioncounts for item in sublist]
    session_cts1 = [item for sublist in session_cts for item in sublist]
    counts, bin_edges = np.histogram(session_cts1, bins=sip_access, range=(0,sip_access))
    eachanimalcounts.append(counts)
plt.figure(figsize=(16, 12), dpi=100)
ax = sns.heatmap(eachanimalcounts, cmap=sns.color_palette("light:#108080", as_cmap=True),square=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Sipper Access Time (sec)', size = 20)
plt.ylabel('Mouse Number', size = 20)
plt.title('Males', size = 20)
plt.savefig('/Users/kristineyoon/Documents/heatmapbymal.pdf', transparent=True)

#################################################################################
######################  HEATMAP OF ALL ANIMALS BY SESSION ####################### 
#################################################################################
allsessioncounts=[]
for sess in by_session_female:
    counts, bin_edges = np.histogram(by_session[sess], bins=sip_access, range=(0,sip_access))
    allsessioncounts.append(counts)
    
plt.figure(figsize=(16, 12), dpi=100)
ax = sns.heatmap(allsessioncounts, cmap=sns.color_palette("light:#DE004A", as_cmap=True),square=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Sipper Access Time (sec)', size = 20)
plt.ylabel('Session', size = 20)
plt.title('Females', size = 20)
plt.savefig('/Users/kristineyoon/Documents/heatmapbyseshfem.pdf', transparent=True)


allsessioncounts=[]
for sess in by_session_male:
    counts, bin_edges = np.histogram(by_session[sess], bins=sip_access, range=(0,sip_access))
    allsessioncounts.append(counts)
    
plt.figure(figsize=(16, 12), dpi=100)
ax = sns.heatmap(allsessioncounts, cmap=sns.color_palette("light:#108080", as_cmap=True),square=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Sipper Access Time (sec)', size = 20)
plt.ylabel('Session', size = 20)
plt.title('Males', size = 20)
plt.savefig('/Users/kristineyoon/Documents/heatmapbyseshmal.pdf', transparent=True)

##################################################################################
########################### LICK RATE OF EACH TRIAL ##############################
##################################################################################
lickratedf=pd.DataFrame()
plt.figure(figsize=(16, 12), dpi=100)
lickratedict = {}
for mouse, session in lick_latency_dict:
    if mouse in males:
        lickratelist = []
        for i in range (0,len(lick_latency_dict[mouse,session])):
            if len(lick_latency_dict[mouse,session][i])/sip_access > 0:
                lickratelist.append(len(lick_latency_dict[mouse,session][i])/sip_access)
        lickratedict[mouse,session]=lickratelist
        maxtrials = len(lickratelist)
        plt.scatter(np.arange(0,maxtrials,1), lickratedict[mouse,session],c='#108080',alpha=0.5,s=10**2) 
lickratedict1 = {}
for mouse, session in lick_latency_dict:
    if mouse in females:
        lickratelist = []
        for i in range (0,len(lick_latency_dict[mouse,session])):
            if len(lick_latency_dict[mouse,session][i])/10 > 0:
                lickratelist.append(len(lick_latency_dict[mouse,session][i])/10)
        lickratedict1[mouse,session]=lickratelist
        maxtrials = len(lickratelist)
        plt.scatter(np.arange(0,maxtrials,1), lickratedict1[mouse,session],c='#DE004A', alpha=0.5,s=10**2)  


plt.yticks(ticks=[-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16], 
           labels=[16,14,12,10,8,6,4,2,0,2,4,6,8,10,12,14,16], size=20)
plt.xlabel('Nth Trial in Session', size=20)
plt.ylabel('Lick Rate Per Trial', size=20)
plt.savefig('/Users/kristineyoon/Documents/triallickrate.pdf', transparent=True)

#####################################################################################################
########################### INDIVIDUAL ANIMAL HEATMAPS OF ALL SESSIONS ##############################
#####################################################################################################
mousedict={}
for mouse in females:
    for subj, session in lick_latency_dict:
        if mouse == subj:
            mousedict[session] = lick_latency_dict.get((str(subj),str(session)))
         
    fig, axs = plt.subplots(len(mousedict), figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for sesh, trials in mousedict.items():
        mouseheatmap=[]
        for i in range(len(trials)):
            counts, bin_edges = np.histogram(trials[i], bins=sip_access, range=(0,sip_access))
            mouseheatmap.append(counts)

        sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#DE004A"), vmin=0, vmax=10, ax=axs[int(sesh)])
        axs[int(sesh)].set_title(f'Session {sesh}')
        axs[int(sesh)].set_xlabel('Sipper Access Time')
    fig.tight_layout()
    
mousedict={}
for mouse in males:
    for subj, session in lick_latency_dict:
        if mouse == subj:
            mousedict[session] = lick_latency_dict.get((str(subj),str(session)))
         
    fig, axs = plt.subplots(len(mousedict), figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for sesh, trials in mousedict.items():
        mouseheatmap=[]
        for i in range(len(trials)):
            counts, bin_edges = np.histogram(trials[i], bins=sip_access, range=(0,sip_access))
            mouseheatmap.append(counts)

        sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#108080"), vmin=0, vmax=10, ax=axs[int(sesh)])
        axs[int(sesh)].set_title(f'Session {sesh}')
        axs[int(sesh)].set_xlabel('Sipper Access Time')
    fig.tight_layout()

########################### NEW IDENTIFIERS ##############################
highdrinker = ['SD1-5', 'SD3-2', 'SD3-5',                               #females
               'SD2-1', 'SD2-2', 'SD2-3' , 'SD2-4', 'SD2-5', 'SD4-1']   #males

#######################################################################################
###################### GRAPHING LATENCY FROM LEVER PRESS TO LICK ######################
#######################################################################################

lick_latency_high = {}
lick_latency_low = {}
by_session_high = {}
by_session_low = {}
session_num=0
for mouse, session in lick_latency_dict:
    licksperhigh=[]
    if mouse in highdrinker:
        lick_latency_high[mouse,session]=lick_latency_dict[mouse,session]
    else:
        lick_latency_low[mouse,session]=lick_latency_dict[mouse,session]

for i in range(0,len(by_session)):
    licksperhigh=[]
    licksperlow=[]
    for mouse, session in lick_latency_high:  
        if session == str(i):
            for lists in lick_latency_high[mouse,session]:
                for items in lists:
                    licksperhigh.append(items)
        by_session_high[i]=licksperhigh
    for mouse, session in lick_latency_low:  
        if session == str(i):
            for lists in lick_latency_low[mouse,session]:
                for items in lists:
                    licksperlow.append(items)
        by_session_low[i]=licksperlow

plt.figure(figsize=(16, 12), dpi=100)
ax = sns.kdeplot(data=by_session_high, palette='light:#AA4B5F')
plt.xlabel('Sipper Access Time (sec)', size = 20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.ylabel('Density', size = 20, labelpad=5)
plt.title('Females', size = 20)
plt.setp(ax.get_legend().get_texts(), fontsize='20')  
plt.setp(ax.get_legend().get_title(), fontsize='20')
plt.xlim(0,sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysessionhigh.pdf', transparent=True)


plt.figure(figsize=(16, 12), dpi=100)
ax = sns.kdeplot(data=by_session_low, palette='light:#778958')
plt.xlabel('Sipper Access Time (sec)', size = 20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.ylabel('Density', size = 20, labelpad=5)
plt.setp(ax.get_legend().get_texts(), fontsize='20')  
plt.setp(ax.get_legend().get_title(), fontsize='20')
plt.title('Males', size = 20)
plt.xlim(0,sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysessionflow.pdf', transparent=True)


##################################################################################
########################### LICK RATE OF EACH TRIAL ##############################
##################################################################################
plt.figure(figsize=(16, 12), dpi=100)
lickratedict = {}
lickratedict1 = {}
for mouse, session in lick_latency_dict:
    if mouse in highdrinker:
        lickratelist = []
        for i in range (0,len(lick_latency_dict[mouse,session])):
            if len(lick_latency_dict[mouse,session][i])/sip_access:
                lickratelist.append(len(lick_latency_dict[mouse,session][i])/sip_access)
        lickratedict[mouse,session]=lickratelist
        maxtrials = len(lickratelist)
        plt.scatter(np.arange(0,maxtrials,1), lickratedict[mouse,session],c='#AA4B5F',alpha=0.5,s=10**2) 
    else:    
        lickratelist = []
        for i in range (0,len(lick_latency_dict[mouse,session])):
            if len(lick_latency_dict[mouse,session][i])/10 > 0:
                lickratelist.append(-len(lick_latency_dict[mouse,session][i])/10)
        lickratedict1[mouse,session]=lickratelist
        maxtrials = len(lickratelist)
        plt.scatter(np.arange(0,maxtrials,1), lickratedict1[mouse,session],c='#778958', alpha=0.5,s=10**2)  
plt.yticks(ticks=[-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16], 
           labels=[16,14,12,10,8,6,4,2,0,2,4,6,8,10,12,14,16], size=20)
plt.xlabel('Nth Trial in Session', size=20)
plt.ylabel('Lick Rate Per Trial', size=20)
plt.savefig('/Users/kristineyoon/Documents/triallickratehighlow.pdf', transparent=True)

#########################switch this out to high and low drinkers############################################################################
########################### INDIVIDUAL ANIMAL HEATMAPS OF ALL SESSIONS ##############################
#####################################################################################################
mousedict={}
for mouse in females:
    for subj, session in lick_latency_dict:
        if mouse == subj:
            mousedict[session] = lick_latency_dict.get((str(subj),str(session)))
         
    fig, axs = plt.subplots(len(mousedict), figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for sesh, trials in mousedict.items():
        mouseheatmap=[]
        for i in range(len(trials)):
            counts, bin_edges = np.histogram(trials[i], bins=sip_access, range=(0,sip_access))
            mouseheatmap.append(counts)

        sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#DE004A"), vmin=0, vmax=10, ax=axs[int(sesh)])
        axs[int(sesh)].set_title(f'Session {sesh}')
        axs[int(sesh)].set_xlabel('Sipper Access Time')
    fig.tight_layout()
    
mousedict={}
for mouse in males:
    for subj, session in lick_latency_dict:
        if mouse == subj:
            mousedict[session] = lick_latency_dict.get((str(subj),str(session)))
         
    fig, axs = plt.subplots(len(mousedict), figsize=(6,25))
    plt.suptitle(f'Mouse: {mouse}\n')
    for sesh, trials in mousedict.items():
        mouseheatmap=[]
        for i in range(len(trials)):
            counts, bin_edges = np.histogram(trials[i], bins=sip_access, range=(0,sip_access))
            mouseheatmap.append(counts)

        sns.heatmap(mouseheatmap, cmap=sns.color_palette("light:#108080"), vmin=0, vmax=10, ax=axs[int(sesh)])
        axs[int(sesh)].set_title(f'Session {sesh}')
        axs[int(sesh)].set_xlabel('Sipper Access Time')
    fig.tight_layout()


