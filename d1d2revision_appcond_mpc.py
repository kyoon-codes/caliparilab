#for d1/d2 revisions

import pandas as pd
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt


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
        'Trial', 
        'NosePoke'
        ] 
    arrays=[ 
        'G:', 
        'D:'
        ]
    Med_log=pd.DataFrame(columns=Columns)
    
    for mouse in mice:
        directory=os.path.join(home_dir, mouse)
        files = [f for f in os.listdir(directory) 
                  if (os.path.isfile(os.path.join(directory, f)) and f[0]=='2')]
        
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

    #Format 'Timestamp' from str to float
    Med_log['Timestamp'] = Med_log['Timestamp'].astype(float)
    # Add a column called 'Session'
    for mouse in mice:
        mouse_log=Med_log.loc[np.equal(Med_log['Mouse'], mouse)]
        for i,day in enumerate(np.unique(mouse_log['Date'])):
            day_log=mouse_log.loc[np.equal(mouse_log['Date'], day)]
            Med_log.at[day_log.index, 'Session']=i
    return Med_log


home_dir = '/Volumes/Kristine/D1D2Revisions/AppetitiveConditioningMedPC'
mice = ['5355',
 '5447',
 '5138',
 '5139',
 '5356',
 '5436',
 '5437',
 '5448',
 '5449',
 '5450',
 '5462',
 '5463',
 '5464',
 '5634',
 '5635',
 '5961',
 '5962']
#, 'F2', 'F3', 'F4', 'M1', 'M2', 'M3', 'M4'
#'4032','4033','4034','4035','4036','4037','4038','4039','4041'
mpc_df = load_formattedMedPC_df(home_dir, mice)
mpc_df.to_csv('Med_log.csv', index = False)

poke_latency = {}
allsessionpokes = {}
for mouse in mice:
    directory=os.path.join(home_dir, mouse)
    files = [f for f in os.listdir(directory) if f[0]=='2' and (os.path.isfile(os.path.join(directory, f)))]
    files.sort()
    pokespermouse = []
        
    for i,f in enumerate(files):
        print(f)
        date=f[:16]
        print(date)
        nosepoke_time = []
        trial_time = []
        latency = []
        
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,1] == date and mpc_df.iloc[i,2] == 'NosePoke':
                nosepoke_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,1] == date and mpc_df.iloc[i,2] == 'Trial':
                trial_time.append(mpc_df.iloc[i,3])
        
        k=0
        while k<len(trial_time):
            pokespersession = []
            for i in range (len(nosepoke_time)):
                if nosepoke_time[i] - trial_time[k] <= 30 and nosepoke_time[i] - trial_time[k] > 0:
                    pokespersession.append(nosepoke_time[i]- trial_time[k])
            if len(pokespersession) != 0:
                latency.append(pokespersession)
            for items in pokespersession:
                pokespermouse.append(items)
            k += 1
        poke_latency[str(mouse),str(date[0:10])]=latency
    
    allsessionpokes[mouse]=pokespermouse


mouse_df = pd.DataFrame.from_dict(poke_latency, orient='index')
allkeys = list(poke_latency.keys())

