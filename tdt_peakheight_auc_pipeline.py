import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.stats import sem

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
colors = ['indianred', 'orange', 'goldenrod', 'gold', 'yellowgreen',
          'mediumseagreen', 'mediumturquoise', 'deepskyblue',
          'dodgerblue', 'slateblue', 'darkorchid', 'purple']

# -------------------------------------------------------------
# CORE FUNCTIONS
# -------------------------------------------------------------
def compute_metrics(df, timerange, auc_interval, peak_interval, trial_limit=10):
    """
    Compute AUC and peak height within given intervals.
    
    df: DataFrame with ['Mouse', 'Session', 'Trial', 'Trace']
    timerange: tuple (start, end) of full trace window
    auc_interval: tuple (start, end) for area calculation
    peak_interval: tuple (start, end) for peak height
    """
    results = []

    for _, row in df.iterrows():
        if row['Trial'] < trial_limit:
            trace = np.array(row['Trace'])
            time = np.linspace(timerange[0], timerange[1], len(trace))

            # --- AUC computation ---
            auc_start_idx = np.searchsorted(time, auc_interval[0])
            auc_end_idx = np.searchsorted(time, auc_interval[1])
            auc_segment = trace[auc_start_idx:auc_end_idx]
            auc_time = time[auc_start_idx:auc_end_idx]
            auc_val = simpson(auc_segment, auc_time)

            # --- Peak computation ---
            peak_start_idx = np.searchsorted(time, peak_interval[0])
            peak_end_idx = np.searchsorted(time, peak_interval[1])
            peak_segment = trace[peak_start_idx:peak_end_idx]
            peak_time = time[peak_start_idx:peak_end_idx]
            peak_val = np.max(peak_segment)
            peak_time_val = peak_time[np.argmax(peak_segment)]

            # --- store results ---
            results.append({
                "Mouse": row["Mouse"],
                "Session": row["Session"],
                "Trial": row["Trial"],
                "AUC": auc_val,
                "PeakHeight": peak_val,
                "PeakTime": peak_time_val
            })
    
    return pd.DataFrame(results)

def plot_metric(df, metric, total_sessions, ylabel, title):
    """
    Plot metric (AUC or PeakHeight) by session with mean ± SEM.
    """
    plt.figure(figsize=(10, 6))
    for session in range(total_sessions):
        session_data = df[df["Session"] == session][metric].values
        if len(session_data) > 0:
            plt.scatter([session]*len(session_data), session_data,
                        color=colors[session % len(colors)], alpha=0.05)
            mean_val = np.mean(session_data)
            err_val = sem(session_data)
            plt.scatter(session, mean_val, color=colors[session % len(colors)])
            plt.errorbar(session, mean_val, yerr=err_val, color=colors[session % len(colors)],
                         capsize=3)
    plt.xlabel("Session")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# RUN FOR EACH TRACE TYPE
# -------------------------------------------------------------
# Cue
cue_metrics_df = compute_metrics(avgcuetrace_df,
                                 timerange=timerange_cue,
                                 auc_interval=(0, 2),
                                 peak_interval=(0, 2),
                                 trial_limit=10)

plot_metric(cue_metrics_df, "AUC", total_sessions=10,
            ylabel="Area Under Curve", title="Cue-Aligned AUC by Session")

plot_metric(cue_metrics_df, "PeakHeight", total_sessions=10,
            ylabel="Peak Height", title="Cue-Aligned Peak Height by Session")

# Lever
lever_metrics_df = compute_metrics(avglevertrace_df,
                                   timerange=timerange_lever,
                                   auc_interval=(-1, 1),
                                   peak_interval=(-1, 1),
                                   trial_limit=10)

plot_metric(lever_metrics_df, "AUC", total_sessions=8,
            ylabel="Area Under Curve", title="Lever-Aligned AUC by Session")

plot_metric(lever_metrics_df, "PeakHeight", total_sessions=8,
            ylabel="Peak Height", title="Lever-Aligned Peak Height by Session")

# First Lick
flick_metrics_df = compute_metrics(avgflicktrace_df,
                                   timerange=timerange_lick,
                                   auc_interval=(0, 2),
                                   peak_interval=(0, 2),
                                   trial_limit=10)

plot_metric(flick_metrics_df, "AUC", total_sessions=8,
            ylabel="Area Under Curve", title="First Lick-Aligned AUC by Session")

plot_metric(flick_metrics_df, "PeakHeight", total_sessions=8,
            ylabel="Peak Height", title="First Lick-Aligned Peak Height by Session")
