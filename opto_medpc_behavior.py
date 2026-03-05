
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
mice = ['8339', '8340', '8341', '8342',  '8344',  '8346', '8347', '8957', '8958', '8960', '8961', '8962',  '8964', '8965',  '8967', '8968', '8969', '8970', '8971', '8972', '8343','8345','8959', '8963']
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
plt.figure(figsize=(len(sessions_to_plot), 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licks_summary_filtered[licks_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session')
plt.ylabel('Licks per Session')
plt.title('Session Lick Counts: ephnr vs ctrl')
plt.legend(title='Group')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(len(sessions_to_plot), 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = levers_summary_filtered[levers_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session',)
plt.ylabel('Presses per Session')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.title('Session Presses Counts: ephnr vs ctrl')
plt.legend(title='Group')
plt.tight_layout()
plt.show()


# -------- Baseline is average of sessions 0-2
def make_baseline_and_sessions(summary_df,baseline_sessions=[0,1,2], sessions=[3,4,5]):
    # Baseline = mean of sessions 0–2 (already normalized)
    baseline = (summary_df[summary_df['session'].isin(baseline_sessions)].groupby('group').agg(mean=('mean','mean'), sem=('sem','mean')).reset_index())
    baseline['session'] = baseline_sessions[-1]  # fake session index for plotting
    # Keep sessions of interest
    stim = summary_df[summary_df['session'].isin(sessions)]
    return pd.concat([baseline, stim], ignore_index=True)

licks_df = pd.DataFrame([(mouse, int(session), count) for (mouse, session), count in session_licks.items() if mouse in group_map], columns=['mouse', 'session', 'licks'])
licks_df['group'] = licks_df['mouse'].map(group_map)
baseline_licks = (licks_df[licks_df['session'].isin([0,1,2])].groupby('mouse')['licks'].mean())

levers_df = pd.DataFrame([(mouse, int(session), count) for (mouse, session), count in session_levers.items() if mouse in group_map], columns=['mouse', 'session', 'levers'])
levers_df['group'] = levers_df['mouse'].map(group_map)
baseline_levers = (levers_df[levers_df['session'].isin([0,1,2])].groupby('mouse')['levers'].mean())

# Mean ± SEM per session per group
licks_summary = (licks_df.groupby(['group', 'session'])['licks'].agg(mean='mean', sem='sem').reset_index())
levers_summary = (levers_df.groupby(['group', 'session'])['levers'].agg(mean='mean', sem='sem').reset_index())

sessions_to_plot = [2, 3, 4, 5]
licks_plot_df = make_baseline_and_sessions(licks_summary, baseline_sessions=[0,1,2], sessions= [3,4,5])
levers_plot_df = make_baseline_and_sessions(levers_summary, baseline_sessions=[0,1,2], sessions= [3,4,5])

baseline = (levers_df[levers_df['session'].isin([0,1,2])].groupby('mouse')['levers'].mean())
# Plotting
plt.figure(figsize=(4, 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licks_plot_df[licks_plot_df['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session')
plt.ylabel('Licks per Session')
plt.title('Session Lick Counts: ephnr vs ctrl')
plt.legend(title='Group')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 6))
colors = {'ctrl':  '#108080','ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = levers_plot_df[levers_plot_df['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session',)
plt.ylabel('Presses per Session')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.title('Session Presses Counts: ephnr vs ctrl')
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
plt.figure(figsize=(4, 6))
colors = {'ctrl': '#108080', 'ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licklatency_summary_filtered[licklatency_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session')
plt.ylabel('Latency per Session')
plt.title('Session Latency Counts: ephnr vs ctrl')
plt.legend(title='Group')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xticks(sessions_to_plot)  # Only show the sessions we're plotting
plt.tight_layout()
plt.show()



# -------- Percentage change
# 

licks_pct_df = licks_df.copy()
baseline = (licks_df[licks_df['session'] == 2].set_index('mouse')['licks'])
licks_pct_df['baseline_session2'] = licks_pct_df['mouse'].map(baseline)
licks_pct_df['pct_change_from_s2'] = ((licks_pct_df['licks'] - licks_pct_df['baseline_session2'])/ licks_pct_df['baseline_session2']) * 100
licks_pct_summary = (licks_pct_df.groupby(['group', 'session'])['pct_change_from_s2'].agg(mean='mean', sem='sem').reset_index())
licks_pct_summary_filtered = licks_pct_summary[licks_pct_summary['session'].isin(sessions_to_plot)]


# Plotting
plt.figure(figsize=(4, 6))
colors = {'ctrl': '#108080', 'ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licks_pct_summary_filtered[licks_pct_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xlabel('Session')
plt.ylabel('Licks per Session')
plt.title('Lick Counts: ephnr vs ctrl')
plt.legend(title='Group')
plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xticks(sessions_to_plot)  # Only show the sessions we're plotting
plt.tight_layout()
plt.show()

levers_pct_df = levers_df.copy()
baseline = (levers_df[levers_df['session'] == 2].set_index('mouse')['levers'])
levers_pct_df['baseline'] = levers_pct_df['mouse'].map(baseline)
levers_pct_df['pct_change'] = ((levers_pct_df['levers'] - levers_pct_df['baseline']) / levers_pct_df['baseline']) * 100
levers_pct_summary = (levers_pct_df.groupby(['group','session'])['pct_change'].agg(mean='mean', sem='sem').reset_index())
levers_pct_summary_filtered = licks_pct_summary[licks_pct_summary['session'].isin(sessions_to_plot)]


plt.figure(figsize=(4, 6))
for grp in ['ctrl', 'ephnr']:
    df = levers_pct_summary_filtered[levers_pct_summary_filtered['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xlabel('Session')
plt.ylabel('% Change in Lever Presses')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

def make_baseline_and_sessions(summary_df,baseline_sessions=[0,1,2], sessions=[3,4,5]):
    # Baseline = mean of sessions 0–2 (already normalized)
    baseline = (summary_df[summary_df['session'].isin(baseline_sessions)].groupby('group').agg(mean=('mean','mean'), sem=('sem','mean')).reset_index())
    baseline['session'] = baseline_sessions[-1]  # fake session index for plotting
    # Keep sessions of interest
    stim = summary_df[summary_df['session'].isin(sessions)]
    return pd.concat([baseline, stim], ignore_index=True)

### this is by average of three baslines
licks_pct_df = licks_df.copy()
#baseline = (licks_df[licks_df['session'] == 2].set_index('mouse')['licks'])
baseline = (licks_df[licks_df['session'].isin([0, 1, 2])].groupby('mouse')['licks'].mean())
licks_pct_df['baseline_session2'] = licks_pct_df['mouse'].map(baseline)
licks_pct_df['pct_change_from_s2'] = ((licks_pct_df['licks'] - licks_pct_df['baseline_session2'])/ licks_pct_df['baseline_session2']) * 100
# Mean ± SEM per session per group
licks_pct_summary = (licks_pct_df.groupby(['group', 'session'])['pct_change_from_s2'].agg(mean='mean', sem='sem').reset_index())

licks_plot_df = make_baseline_and_sessions(licks_pct_summary, baseline_sessions=[0,1,2], sessions= [3,4,5])

plt.figure(figsize=(4, 6))
colors = {'ctrl': '#108080', 'ephnr': '#AA4B5F'}

for grp in ['ctrl', 'ephnr']:
    df = licks_plot_df[licks_plot_df['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'],marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xlabel('Session')
plt.ylabel('% Change in Licks')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

baseline = (levers_df[levers_df['session'].isin([0,1,2])].groupby('mouse')['levers'].mean())
levers_pct_df = levers_df.copy()
levers_pct_df['baseline'] = levers_pct_df['mouse'].map(baseline)
levers_pct_df['pct_change'] = ((levers_pct_df['levers'] - levers_pct_df['baseline']) / levers_pct_df['baseline']) * 100
levers_pct_summary = (levers_pct_df.groupby(['group','session'])['pct_change'].agg(mean='mean', sem='sem').reset_index())

levers_plot_df = make_baseline_and_sessions(levers_pct_summary,baseline_sessions=[0,1,2], sessions= [3,4,5])

plt.figure(figsize=(4, 6))
for grp in ['ctrl', 'ephnr']:
    df = levers_plot_df[levers_plot_df['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xlabel('Session')
plt.ylabel('% Change in Lever Presses')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# 
licklatency_pct_df = licklatency_df.copy()
baseline = (licklatency_pct_df[licklatency_pct_df['session'].isin([0, 1, 2])].groupby('mouse')['latency'].mean())
licklatency_pct_df['baseline_session2'] = licklatency_pct_df['mouse'].map(baseline)
licklatency_pct_df['pct_change_from_s2'] = ((licklatency_pct_df['latency'] - licklatency_pct_df['baseline_session2'])/ licklatency_pct_df['baseline_session2']) * 100
licklatency_pct_summary = (licklatency_pct_df.groupby(['group', 'session'])['pct_change_from_s2'].agg(mean='mean', sem='sem').reset_index())
licklatency_plot_df = make_baseline_and_sessions(licklatency_pct_summary,  baseline_sessions, sessions_to_plot)

plt.figure(figsize=(4, 6))
for grp in ['ctrl', 'ephnr']:
    df = licklatency_plot_df[licklatency_plot_df['group'] == grp]
    plt.errorbar(df['session'], df['mean'], yerr=df['sem'], marker='o', linewidth=1, capsize=2, color=colors[grp], label=grp)

plt.xticks([2,3,4,5], ['Baseline', 'Opto 1', 'Opto 2', 'Opto 3'])
plt.xlabel('Session')
plt.ylabel('% Change in Latency')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# -------- For Putting into Prism
def make_mouse_wide_pct_df(df,value_col, baseline_sessions=(0, 1, 2), sessions_of_interest=(3, 4, 5), mouse_col='mouse', session_col='session', group_col='group'):
    df = df.copy()
    baseline = (df[df[session_col].isin(baseline_sessions)].groupby(mouse_col)[value_col].mean())
    # df['pct_change'] = ((df[value_col] - df[mouse_col].map(baseline))/ df[mouse_col].map(baseline)) * 100

    # ---- Baseline column (in pct-change space)
    # baseline_pct = (df[df[session_col].isin(baseline_sessions)].groupby(mouse_col)['pct_change'].mean().rename('baseline'))

    # ---- Sessions 3–5 columns
    sessions = (df[df[session_col].isin(sessions_of_interest)].pivot(index=mouse_col, columns=session_col, values=value_col).rename(columns={s: f'session{s}' for s in sessions_of_interest}))

    # ---- Combine
    wide_df = pd.concat([baseline, sessions], axis=1)

    # ---- Add group column
    wide_df[group_col] = (df.drop_duplicates(mouse_col).set_index(mouse_col)[group_col])

    return wide_df

licks_mouse_wide = make_mouse_wide_pct_df(licks_df, value_col='licks')
levers_mouse_wide = make_mouse_wide_pct_df(levers_df, value_col='levers')
latency_mouse_wide = make_mouse_wide_pct_df(licklatency_df, value_col='latency')


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
