import tdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy
import scipy.signal as spsignal
from scipy.signal import find_peaks    

# folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/All/'
# mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321']

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/Omission2/'
mice = ['7098','7099','7108','7296','7311', '7319', '7321']


spectrumdens_dict={}
trace_dict = {}
eventrate_dict = {}
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
        
        signal=df['Dff'].values[2000:900000]
        #spectral
        (f, S) = spsignal.periodogram(signal, fs=fs, scaling='density')
        logbins = np.logspace(np.log2(0.003),np.log2(10),20, base=2)
        logbins = np.insert(logbins, 0,0)
        Binned_S=[]
        for a,b in zip(logbins[:-1], logbins[1:]):
            Binned_S.append(sum([x for x,ff in zip(S,f) if (ff>a) & (ff<=b)]))
        
        spectrumdens_dict[mouse,Dates.index(date)]= Binned_S
        
        
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        zscore_signal = []
        for each in signal:
            zscore_signal.append((each-mean_signal)/std_signal)
        trace_dict[mouse,Dates.index(date)]= zscore_signal
        
        param_thresh= 0.0
        param_prom= 0.6
        peaks, properties = find_peaks(zscore_signal, threshold = param_thresh, prominence=param_prom, height=.01)
        eventrate_dict[mouse,Dates.index(date)]= len(peaks)
       
       

colors10 = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61','#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']

plt.figure()
for i in range (9):
    session_psd=[]
    for mouse, session in spectrumdens_dict:
        if session ==i:
            session_psd.append(spectrumdens_dict[mouse, session])
    
    mean_session = np.mean(session_psd, axis=0)
    sem_session = np.std(session_psd, axis=0)/np.sqrt(len(session_psd)-1) 
    
    # for x,point in enumerate(session_psd):
    #     plt.plot(point, color=colors10[i], alpha=0.8, label=i)
    plt.plot(range(len(mean_session)), mean_session, color=colors10[i], label=i)
    plt.fill_between(range(len(mean_session)), mean_session-sem_session, mean_session+sem_session, color=colors10[i], alpha=0.3)
    
plt.legend()
plt.ylabel('Power')
plt.xlabel('Frenquency')


# Define function to calculate SEM
def sem(arr):
    return np.std(arr, axis=0) / np.sqrt(len(arr))


event_df = pd.DataFrame()
plt.figure()
for i in range (8):
    mean_session=[]
    for mouse, session in eventrate_dict:
        
        if session == i:
            mean_session.append(eventrate_dict[mouse, session])
            count= len(event_df)
            event_df.at[count,'mouse']=mouse
            event_df.at[count,'session']=session
            event_df.at[count,'events']=eventrate_dict[mouse, session]
            
            plt.scatter(x=session,y= eventrate_dict[mouse, session],color=colors10[i], alpha=0.1)
    plt.scatter(x=i, y=np.mean(mean_session), color=colors10[i])
    plt.errorbar(x=i, y=np.mean(mean_session), yerr=sem(mean_session), ecolor=colors10[i],capsize=3)
plt.xlabel('Session')
plt.ylabel('Events in Session')
plt.show()



#### TRIALS ON HEATMAP ALIGNED TO CUE of all session sorted by latency to lick ####


alltraces=[]
for i in range(8):
    for subj,session in trace_dict:
        if session == i:
            alltraces.append(trace_dict[subj,session])
plt.figure()
bysessiondf= pd.DataFrame(alltraces)    
sns.heatmap(bysessiondf, cmap='RdBu', vmin=-5, vmax=5, )
# plt.xticks(np.arange(0,bysessiondf.shape[1]+1,(bysessiondf.shape[1]+1)/(active_time[1]-active_time[0])*2), 
#             np.arange(active_time[0], active_time[1],2, dtype=int),
#             rotation=0)
plt.xlabel('Time (sec)')
plt.show()

