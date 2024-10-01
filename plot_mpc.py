import pandas as pd
import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 500
matplotlib.rcParams['savefig.pad_inches'] = 0.2

home_dir = '/Volumes/Kristine/EtOH_SA_Pilot'
mice = ['M3']
#,'4032','4033','4034','4035','4036','4037','4038','4039','4041'

Columns=['Mouse', 'Date', 'Event', 'Timestamp']
events=['ActiveLever', 'Cue','LeverPress', 'Lick'] 
arrays=[ 'J:', 'P:', 'N:', 'O:']

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


for mouse in mice:
    directory=os.path.join(home_dir, mouse)
    files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and f[0]!='.')]
    files.sort()
    fig, axs = plt.subplots(len(files), 1, figsize = (8,15), sharey=True)
    for i,f in enumerate(files):
        print(f)
        date=f[:10]
         
        for event,col_n in zip(events, arrays):
            Timestamps=read_med(os.path.join(directory, f),[f[-6:-4], f[:10]], var_cols=5,col_n=col_n) 
            if len(Timestamps) == 0:
                Timestamps.at[0,'vC']=0
            elif len(Timestamps)!=0:
                if event == 'Cue':
                    cue_data = Timestamps['vC'].to_list()
                    # hl_data = Timestamps.to_numpy()
                    # hl_data = np.concatenate((hl_data), axis=None)
                    continue
                elif event == 'ActiveLever':
                    press_data = Timestamps['vC'].to_list()
                    # lick_data = Timestamps.to_numpy()
                    # lick_data = np.concatenate((lick_data), axis=None)
                    continue
                elif event == 'Lick':
                    lick_data = Timestamps['vC'].to_list()
                    # rein_data = Timestamps.to_numpy()
                    # rein_data = np.concatenate((rein_data), axis=None)
                    continue

        data_np = [#[float(i) for i in cue_data],
                   [float(i) for i in lick_data],
                   [float(i) for i in press_data]
                   ]
        colors1 = ['gold','royalblue']
        lineoffsets1 = [-2,2]
        linelengths1 = [3, 3]
        max_value = 200
        #max([lastelement[-1] for lastelement in data_np])
        plt.setp(axs, 
                 xticks=np.arange(0, max_value + 100.0, 300), 
                 yticks=lineoffsets1, 
                 yticklabels= [#'Cue',
                               'Licks','LeverPress'])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_title(f'{date}')
        axs[i].eventplot(data_np, colors=colors1, lineoffsets=lineoffsets1,
                            linelengths=linelengths1)
        
    fig.suptitle(f'{mouse}', fontweight="bold")
    fig.tight_layout()
    plt.show()


