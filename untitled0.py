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
mice = ['7098','7099','7107','7108','7296', '7310', '7311', '7319', '7321']

spectrumdens_dict={}

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
        
        signal=df['Dff'].values[:30000]
        #spectral
        (f, S) = spsignal.periodogram(signal, fs=20, scaling='density')
        logbins = np.logspace(np.log2(0.003),np.log2(10),20, base=2)
        logbins = np.insert(logbins, 0,0)
        Binned_S=[]
        for a,b in zip(logbins[:-1], logbins[1:]):
            Binned_S.append(sum([x for x,ff in zip(S,f) if (ff>a) & (ff<=b)]))
        
        spectrumdens_dict[mouse,Dates.index(date)]= Binned_S



mean_fed=np.mean(Fed_spectrum, axis=0)     
mean_hungry=np.mean(Hungry_spectrum, axis=0)    
sem_fed=np.std(Fed_spectrum, axis=0)/np.sqrt(len(Fed_spectrum)-1) 
sem_hungry=np.std(Hungry_spectrum, axis=0)/np.sqrt(len(Hungry_spectrum)-1) 
# colors=['cornflowerblue','mediumblue']
colors=['orange', 'tomato']
fig=plt.figure(figsize=(1,2))
ax=fig.add_axes([0.5,0.25,0.5,0.75])
for p,(fed, hungry) in enumerate(zip(Fed_spectrum, Hungry_spectrum)):
    plt.plot(fed, color=colors[0], alpha=0.2, linewidth=1)
    plt.plot(hungry, color=colors[1], alpha=0.2, linewidth=1)  
# plt.bar(np.arange(len(mean_fed))-0.1, mean_fed, color='b', width=0.2)
# plt.bar(np.arange(len(mean_fed))+0.1,mean_hungry, color='r', width=0.2)
# plt.plot(np.arange(len(mean_fed))-0.1, mean_fed, color='b', width=0.2)
# plt.plot(np.arange(len(mean_fed))+0.1,mean_hungry, color='r', width=0.2)
plt.fill_between(range(len(mean_fed)), mean_fed-sem_fed, mean_fed+sem_fed, color=colors[0], alpha=0.3)
plt.fill_between(range(len(mean_fed)), mean_hungry-sem_hungry, mean_hungry+sem_hungry, color=colors[1], alpha=0.3)
plt.ylabel('Power', size=7)
plt.xlabel('Frenquency', size=7)
# plt.xticks(np.arange(-0.5,15,1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(width=1)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1)
plt.title('N='+str(len(mice)))
plt.ylim(0,3000)