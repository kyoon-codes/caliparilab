import numpy as np
import matplotlib.pyplot as plt


colors = sns.color_palette("husl", 10)

def plot_trace_and_derivative_individual(df, label, timerange, ylim, ogtime, experiment, colors):
    session_list = sorted(df['Session'].unique())
    plt.figure(figsize=(10, 8))

    for i, session in enumerate(session_list):
        if i in [7]:  # select specific session index, or just directly specify session ID instead
            traces = np.stack(df.loc[df['Session']==session, 'Trace'])
            time = np.linspace(ogtime[0], ogtime[1], len(traces[0]))
    
            # restrict to time window of interest
            start_idx = np.searchsorted(time, timerange[0])
            end_idx = np.searchsorted(time, timerange[1])
            timesegment = time[start_idx:end_idx]

            # plot each trace and its derivative separately
            for trial_i, trace in enumerate(traces):
                if trial_i < 10:
                    segment = trace[start_idx:end_idx]
                    deriv = np.diff(segment) / np.diff(time[start_idx:end_idx])[0]
    
                    plt.plot(timesegment, segment,
                             color=colors[i % len(colors)], alpha=0.5, label=f'Trace (Trial {trial_i+1})' if trial_i==0 else "")
                    plt.plot(timesegment[:-1], deriv,
                             linestyle='--', color=colors[i % len(colors)], alpha=0.5, label=f'Derivative (Trial {trial_i+1})' if trial_i==0 else "")

    plt.xlabel('Time (s)')
    plt.ylabel('z-score / derivative')
    plt.title(f'{label}-Aligned Traces and Derivatives (Individual Trials)')
    plt.axhline(y=0, linestyle=':', color='black')
    plt.axvline(x=0, linewidth=1, color='black')
    plt.legend()
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


newavglickbouttrace_df = avglickbouttrace_df[(avglickbouttrace_df['Mouse']== '8732')]
plot_trace_and_derivative_bysession(newavglickbouttrace_df, 'Lick', [-2,15], (-10,15), timerange_lick, experiment, colors)
