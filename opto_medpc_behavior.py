
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
        'Cue'
        ] 
    arrays=[ 
        'J:', 
        'O:',
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
            date=f[:10]
            
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
                    continue
    #Format 'Timestamp' from str to float
    Med_log['Timestamp'] = Med_log['Timestamp'].astype(float)
    # Add a column called 'Session'
    
    for mouse in mice:
        mouse_log = Med_log.loc[Med_log['Mouse'] == mouse]
        for i, day in enumerate(np.unique(mouse_log['Date'])):
            day_idx = mouse_log.loc[mouse_log['Date'] == day].index
            Med_log.loc[day_idx, 'Session'] = i
        
    return Med_log


home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D2_Opto_EtOH_MPC'
files = os.listdir(home_dir)
files.sort()
print(files)
ommited_mice = ['8343','8345','8959', '8963',] #due to placement
mice = ['8339', '8340', '8341', '8342',  '8344',  '8346', '8347', '8957', '8958', '8960', '8961', '8962',  '8964', '8965',  '8967', '8968', '8969', '8970', '8971', '8972']
maxsession = 9 # ommited session 6 because two ctrl were not recorded correctly
sip_access = 10
mpc_df = load_formattedMedPC_df(home_dir, mice)
mpc_df.to_csv('Med_log.csv', index = False)


##################################################################################
################### ALL LICK LATENCY BY TRIALS IN EACH SESSION ###################
##################################################################################
lickpattern_df = pd.DataFrame()
lick_latency_dict = {}
session_levers = {}
session_licks = {}
collapselicksby_session = {}

grouped = mpc_df.groupby(['Mouse', 'Session'])

for j in range(maxsession + 1):
    lickspermouse = []

    for mouse in mice:
        if (mouse, j) not in grouped.groups:
            continue

        df = grouped.get_group((mouse, j))

        cue_time   = df.loc[df['Event'] == 'Cue', 'Timestamp'].values
        lick_time  = df.loc[df['Event'] == 'Lick', 'Timestamp'].values
        lever_time = df.loc[df['Event'] == 'ActiveLever', 'Timestamp'].values

        print(f'Mouse: {mouse} Session: {j}')

        # -------------------- LICK COUNTS --------------------
        session_licks[(mouse, j)] = len(lick_time)
        
        # -------------------- LEVER COUNTS --------------------
        session_levers[(mouse, j)] = len(lever_time)

        # ---------------- CUE → LICK LATENCY ----------------
        licklatency = []
        for lt in cue_time:
            diffs = lick_time - lt
            valid = diffs[(diffs > 0) & (diffs <= sip_access)]
            licklatency.append(valid.tolist())
            lickspermouse.extend(valid.tolist())
            
        if len(licklatency) > 0:
            for i in range(len(licklatency)):
                if len(licklatency[i]) > 0:
                    lick_latency_dict[(mouse,j,i)] = licklatency[i][0]


        # ---------------- INTERLICK INTERVAL ----------------
        if len(lick_time) > 1:
            interlick = np.diff(lick_time)
            interlick_short = interlick[interlick < 10]
            ili_avg = interlick_short.mean() if len(interlick_short) else 0
        else:
            interlick = np.array([])
            ili_avg = 0

        lickpattern_df.at[f'Session {j} Interlick Interval', mouse] = ili_avg

        # ---------------- LONGEST LICK BOUT ----------------
        if len(interlick):
            breaks = np.where(interlick > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends   = np.r_[breaks, len(lick_time) - 1]

            bout_lengths = lick_time[ends] - lick_time[starts]
            bout_counts  = ends - starts + 1

            lickpattern_df.at[f'Session {j} Longest Lick Bout Time', mouse] = bout_lengths.max()
            lickpattern_df.at[f'Session {j} Longest Lick Bout #', mouse] = bout_counts.max()
        else:
            lickpattern_df.at[f'Session {j} Longest Lick Bout Time', mouse] = 0
            lickpattern_df.at[f'Session {j} Longest Lick Bout #', mouse] = 0

    collapselicksby_session[j] = lickspermouse


########################### IDENTIFIERS ##############################
ephnr = ['8339', '8341', '8342', '8343', '8344', '8345', '8346',  '8959', '8960', '8961', '8963', '8964', '8965']
ctrl  = ['8340', '8957', '8958', '8962', '8966', '8967', '8968', '8969', '8970', '8971', '8972']



# -------------------------------------------------------------------
# Plot ephnr vs ctrl across sessions
# -------------------------------------------------------------------
# Convert session_licks dict → tidy dataframe with group labels
group_map = {m: 'ephnr' for m in ephnr} | {m: 'ctrl' for m in ctrl}

licks_df = pd.DataFrame([(mouse, int(session), count) for (mouse, session), count in session_licks.items() if mouse in group_map], columns=['mouse', 'session', 'licks'])
licks_df['group'] = licks_df['mouse'].map(group_map)

levers_df = pd.DataFrame([(mouse, int(session), count) for (mouse, session), count in session_levers.items() if mouse in group_map], columns=['mouse', 'session', 'levers'])
levers_df['group'] = levers_df['mouse'].map(group_map)

# Mean ± SEM per session per group
licks_summary = (licks_df.groupby(['group', 'session'])['licks'].agg(mean='mean', sem='sem').reset_index())
levers_summary = (levers_df.groupby(['group', 'session'])['levers'].agg(mean='mean', sem='sem').reset_index())

sessions_to_plot = [2, 3, 4, 5]
licks_summary_filtered = licks_summary[licks_summary['session'].isin(sessions_to_plot)]
levers_summary_filtered = levers_summary[levers_summary['session'].isin(sessions_to_plot)]

# Plotting
plt.figure(figsize=(8, 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licks_summary_filtered[licks_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session', fontsize=14)
plt.ylabel('Licks per Session', fontsize=14)
plt.title('Session Lick Counts: ephnr vs ctrl', fontsize=16)
plt.legend(title='Group')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = levers_summary_filtered[levers_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session', fontsize=14)
plt.ylabel('Presses per Session', fontsize=14)
plt.title('Session Presses Counts: ephnr vs ctrl', fontsize=16)
plt.legend(title='Group')
plt.tight_layout()
plt.show()


# Run Stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

licks_df['group'] = licks_df['group'].astype('category')
licks_df['session'] = licks_df['session'].astype(int)

lme = smf.mixedlm("licks ~ group * session",licks_df,groups=licks_df["mouse"])
lme_res = lme.fit(reml=False)
print(lme_res.summary())

posthoc_results = []

for sesh in sorted(licks_df['session'].unique()):
    df_s = licks_df[licks_df['session'] == sesh]

    # Skip sessions without both groups
    if df_s['group'].nunique() < 2:
        continue

    model = smf.mixedlm("licks ~ group", df_s, groups=df_s["mouse"])

    res = model.fit(reml=False)

    coef = res.params.get('group[T.ephnr]', np.nan)
    pval = res.pvalues.get('group[T.ephnr]', np.nan)

    posthoc_results.append({'session': sesh,'coef_ephnr_vs_ctrl': coef,'p_uncorrected': pval})

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df['p_fdr'] = multipletests(posthoc_df['p_uncorrected'], method='fdr_bh')[1]

print(posthoc_df)


levers_df['group'] = levers_df['group'].astype('category')
levers_df['session'] = levers_df['session'].astype(int)

lme = smf.mixedlm("levers ~ group * session",levers_df,groups=levers_df["mouse"])
lme_res = lme.fit(reml=False)
print(lme_res.summary())

posthoc_results = []

for sesh in sorted(levers_df['session'].unique()):
    df_s = levers_df[levers_df['session'] == sesh]

    # Skip sessions without both groups
    if df_s['group'].nunique() < 2:
        continue

    model = smf.mixedlm("levers ~ group", df_s, groups=df_s["mouse"])

    res = model.fit(reml=False)

    coef = res.params.get('group[T.ephnr]', np.nan)
    pval = res.pvalues.get('group[T.ephnr]', np.nan)

    posthoc_results.append({'session': sesh,'coef_ephnr_vs_ctrl': coef,'p_uncorrected': pval})

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df['p_fdr'] = multipletests(posthoc_df['p_uncorrected'], method='fdr_bh')[1]

print(posthoc_df)


########################### AVERAGE LATENCY: CUE → LICK ##############################
licklatency_df = pd.DataFrame([(mouse, int(session), int(trial), count) for (mouse, session, trial), count in lick_latency_dict.items() if mouse in group_map], columns=['mouse', 'session','' ,'latency'])
licklatency_df['group'] = licklatency_df['mouse'].map(group_map)
licklatency_summary = (licklatency_df.groupby(['group', 'session'])['latency'].agg(mean='mean', sem='sem').reset_index())

# Sessions to plot
licklatency_summary_filtered = licklatency_summary[licklatency_summary['session'].isin(sessions_to_plot)]

# Plotting
plt.figure(figsize=(8, 6))
colors = {'ctrl': '#108080', 'ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licklatency_summary_filtered[licklatency_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session', fontsize=14)
plt.ylabel('Latency per Session', fontsize=14)
plt.title('Session Latency Counts: ephnr vs ctrl', fontsize=16)
plt.legend(title='Group')
plt.xticks(sessions_to_plot)  # Only show the sessions we're plotting
plt.tight_layout()
plt.show()


########################### FIRST LICK LATENCY AFTER LEVER ##############################
flickdf_ephnr = pd.DataFrame()
flickdf_ctrl  = pd.DataFrame()


licks_df = pd.DataFrame([(mouse, int(session), count) for (mouse, session), count in session_licks.items() if mouse in group_map], columns=['mouse', 'session', 'licks'])
licks_df['group'] = licks_df['mouse'].map(group_map)

for mouse, session in lick_latency_dict:
    for i in range(maxsession + 1):

        if mouse in ephnr and int(session) == i:
            flicks = lick_latency_dict[(mouse, session)]
            firstlicks = [trial[0] for trial in flicks if len(trial) > 0]
            if firstlicks:
                flickdf_ephnr.at[i, mouse] = np.mean(firstlicks)

        if mouse in ctrl and int(session) == i:
            flicks = lick_latency_dict[(mouse, session)]
            firstlicks = [trial[0] for trial in flicks if len(trial) > 0]
            if firstlicks:
                flickdf_ctrl.at[i, mouse] = np.mean(firstlicks)

flickdf_ephnr.to_csv('/Users/kristineyoon/Documents/flickdf_ephnr.csv')
flickdf_ctrl.to_csv('/Users/kristineyoon/Documents/flickdf_ctrl.csv')


########################### KDE BY SESSION (EPHNR) ##############################
lick_latency_ephnr = {}
by_session_ephnr = {}

for mouse, session in lick_latency_dict:
    if mouse in ephnr:
        lick_latency_ephnr[mouse, session] = lick_latency_dict[mouse, session]

for i in range(len(by_session)):
    licks = []
    for mouse, session in lick_latency_ephnr:
        if session == str(i):
            for trial in lick_latency_ephnr[mouse, session]:
                licks.extend(trial)
    by_session_ephnr[i] = licks
    
plt.figure(figsize=(16,12), dpi=100)
ax = sns.kdeplot(data=by_session_ephnr, palette='light:#AA4B5F')
plt.title('ephnr', size=20)
plt.xlim(0, sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysession_ephnr.pdf', transparent=True)


########################### KDE BY SESSION (CTRL) ##############################
lick_latency_ctrl = {}
by_session_ctrl = {}

for mouse, session in lick_latency_dict:
    if mouse in ctrl:
        lick_latency_ctrl[mouse, session] = lick_latency_dict[mouse, session]

for i in range(len(by_session)):
    licks = []
    for mouse, session in lick_latency_ctrl:
        if session == str(i):
            for trial in lick_latency_ctrl[mouse, session]:
                licks.extend(trial)
    by_session_ctrl[i] = licks
    
plt.figure(figsize=(16,12), dpi=100)
ax = sns.kdeplot(data=by_session_ctrl, palette='light:#108080')
plt.title('ctrl', size=20)
plt.xlim(0, sip_access)
plt.savefig('/Users/kristineyoon/Documents/bysession_ctrl.pdf', transparent=True)

########################### LICK RATE PER TRIAL (EPHNR vs CTRL) ##############################

plt.figure(figsize=(16,12), dpi=100)

for mouse, session in lick_latency_dict:
    rates = [len(trial)/sip_access for trial in lick_latency_dict[(mouse,session)] if len(trial)>0]

    if mouse in ephnr:
        plt.scatter(range(len(rates)), rates, c='#AA4B5F', alpha=0.5, s=100)

    if mouse in ctrl:
        plt.scatter(range(len(rates)), [-r for r in rates], c='#108080', alpha=0.5, s=100)

plt.xlabel('Nth Trial', size=20)
plt.ylabel('Lick Rate Per Trial', size=20)
plt.savefig('/Users/kristineyoon/Documents/triallickrate_ephnr_ctrl.pdf', transparent=True)

########################### INDIVIDUAL ANIMAL HEATMAPS (GENERIC TEMPLATE) ##############################

def plot_individual_heatmaps(group, cmap):
    for mouse in group:
        mousedict = {}
        for subj, session in lick_latency_dict:
            if subj == mouse:
                mousedict[int(session)] = lick_latency_dict[(subj, session)]

        fig, axs = plt.subplots(len(mousedict), figsize=(6,25))
        plt.suptitle(f'Mouse: {mouse}')

        for sesh, trials in mousedict.items():
            heat = [np.histogram(trial, bins=sip_access, range=(0,sip_access))[0]
                    for trial in trials]
            sns.heatmap(heat, cmap=cmap, vmin=0, vmax=10, ax=axs[sesh])
            axs[sesh].set_title(f'Session {sesh}')
plot_individual_heatmaps(ephnr, sns.color_palette("light:#AA4B5F"))
plot_individual_heatmaps(ctrl,  sns.color_palette("light:#108080"))
