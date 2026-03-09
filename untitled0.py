
# ========================= IMPORTS =========================
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tdt
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import linregress

from scipy.stats import sem, kruskal
from scipy.integrate import simpson


# -------------------------- ALL FILES --------------------------

folder = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/D1D2_FiberPhotometry/'



# ------- D2 MEDIUM SPINY NEURONS (ALCOHOL) -------
mice = ['7098','7099','7107','7108', '7310', '7311', '7319', '7321','8729','8730','8731','8732','9299','9302','9325','9326','9327'] #'7296','9325','9326',
male = ['7107','7108','7319', '7321','8729','8730','8731','8732','9299','9302']
female = ['7098','7099','7310', '7311','9325','9326','9327' ]
experiment = 'D2_EtOHLearning'


mice = ['9299','9302','9325','9326','9327']
experiment = 'D2_EtOHLongExtinction'
# experiment = 'D2_1WeekWD'



# mice = ['7098','7099','7108', '7311', '7319', '7321','8729','8730','8731','8732'] #'7296',
# experiment = 'D2_EtOHExtinction'

# ------- D2 MEDIUM SPINY NEURONS (SUCROSE) -------

mice = ['7678', '7680', '8733','8742','8743','8747','8748','8750', '7899']
experiment = 'D2_SucLearning' #7899 ommitted for this because they did not lick on session 3
male = ['7678', '7680','8742','8743','8747','8748']
female = ['7899', '8733','8750']
# mice = ['7678', '7680', '7899','8733','8742','8743','8747','8748','8750']
# experiment = 'D2_SucExtinction'
# experiment = 'D2_SuctoEtOH_EtOHLearning'
# experiment = 'D2_SuctoEtOH_AlcExtinction'
# experiment = 'D2_SuctoEtOH_SucExtinction2'


# ------- D1 MEDIUM SPINY NEURONS -------
mice = ['676', '679', '849', '873', '874', '917']
# experiment = 'D1_EtOHLearning'
# experiment = 'D1_1WeekWD'
male = ['676', '679', '849']
female = ['873', '874', '917']
# experiment = 'D1_EtOHExtinction'
experiment = 'D1_SucLearning'
# experiment = 'D1_SucExtinction'



# ========================= FAST HELPERS =========================
def extract_events_fast(data, events, epocs):
    """
    Faster, no string parsing. Safely tries to read epoc.onset.
    Returns DataFrame with columns: Event, Timestamp
    """
    out_event = []
    out_ts = []
    for ev_name, ep in zip(events, epocs):
        try:
            ts = getattr(data.epocs, ep).onset
        except Exception:
            continue
        if ts is None or len(ts) == 0:
            continue
        out_event.append(np.full(len(ts), ev_name, dtype=object))
        out_ts.append(np.asarray(ts, dtype=float))

    if not out_event:
        return pd.DataFrame({"Event": [], "Timestamp": []})

    return pd.DataFrame({
        "Event": np.concatenate(out_event),
        "Timestamp": np.concatenate(out_ts)
    })


def identify_firstlicks_fast(track_lever, track_licks):
    """
    Returns:
      firstlick_indices (indices into track_licks)
      lick_counts (number of licks between lever presses)
    """
    track_lever = np.asarray(track_lever, dtype=float)
    track_licks = np.asarray(track_licks, dtype=float)

    first_idx = []
    counts = []

    for i, press in enumerate(track_lever):
        upper = track_lever[i + 1] if i < len(track_lever) - 1 else np.inf
        mask = (track_licks > press) & (track_licks < upper)
        idx = np.flatnonzero(mask)
        if idx.size:
            first_idx.append(idx[0])
            counts.append(idx.size)

    return np.asarray(first_idx, dtype=int), np.asarray(counts, dtype=int)


def identify_lick_bouts_fast(lick_timestamps, max_interval=0.4):
    """
    Returns list of (bout_len, bout_times_list) sorted by bout_len ascending
    """
    licks = np.asarray(lick_timestamps, dtype=float)
    if licks.size == 0:
        return []

    # boundaries where interval > max_interval
    gaps = np.flatnonzero(np.diff(licks) > max_interval)
    # bout start indices in licks
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps + 1, licks.size]

    bouts = []
    for s, e in zip(starts, ends):
        bout = licks[s:e].tolist()
        bouts.append((e - s, bout))

    bouts.sort(key=lambda x: x[0])
    return bouts


def find_trial_for_events_searchsorted(event_times, cue_times, window):
    """
    Vectorized mapping of each event to the most recent cue.
    Returns:
      cue_idx (int array), valid_mask (bool array)
    """
    event_times = np.asarray(event_times, dtype=float)
    cue_times = np.asarray(cue_times, dtype=float)
    cue_idx = np.searchsorted(cue_times, event_times, side="right") - 1
    valid = (cue_idx >= 0) & ((event_times - cue_times[cue_idx]) < window)
    return cue_idx, valid


# ========================= SIGNAL PROCESSING =========================

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np

def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow*np.exp(-t/tau_slow) + amp_fast*np.exp(-t/tau_fast)

def process_signal_dff_fast(sig405, sig465, fs,
                            lowpass_hz=20,
                            fs_ds=100,     # output sampling rate
                            fs_fit=10,     # bleaching-fit sampling rate
                            maxfev=800):
    sig405 = np.asarray(sig405, dtype=float)
    sig465 = np.asarray(sig465, dtype=float)
    n = min(sig405.size, sig465.size)
    sig405 = sig405[:n]
    sig465 = sig465[:n]

    # 1) Low-pass at native fs (zero-phase)
    b, a = butter(2, lowpass_hz, btype="low", fs=fs)
    s465 = filtfilt(b, a, sig465)
    s405 = filtfilt(b, a, sig405)

    # 2) Downsample for downstream processing (speed)
    ds_step = int(round(fs / fs_ds))
    s465_ds = s465[::ds_step]
    s405_ds = s405[::ds_step]
    t_ds = np.arange(s465_ds.size) / fs_ds

    # 3) Fit bleaching on an even lower rate
    fit_step = int(round(fs_ds / fs_fit))
    s465_fit = s465_ds[::fit_step]
    s405_fit = s405_ds[::fit_step]
    t_fit = np.arange(s465_fit.size) / fs_fit

    def fit_bleach(y, t):
        m = float(np.max(y))
        p0 = [m/2, m/4, m/4, 3600, 0.1]
        bounds = ([0, 0, 0, 600, 0], [m, m, m, 36000, 1])
        try:
            p, _ = curve_fit(double_exponential, t, y, p0=p0, bounds=bounds, maxfev=maxfev)
        except Exception:
            p = p0
        return p

    p465 = fit_bleach(s465_fit, t_fit)
    p405 = fit_bleach(s405_fit, t_fit)

    # evaluate fits back at ds resolution
    fit465_ds = double_exponential(t_ds, *p465)
    fit405_ds = double_exponential(t_ds, *p405)

    # 4) Detrend + motion regression at ds resolution
    d465 = s465_ds - fit465_ds
    d405 = s405_ds - fit405_ds

    slope, intercept, _, _, _ = linregress(d405, d465)
    motion_est = intercept + slope * d405
    corrected = d465 - motion_est

    # 5) dF/F at ds resolution
    dff_ds = 100.0 * corrected / fit465_ds

    return dff_ds, fs_ds


# ========================= MAIN PIPELINE =========================
def run_pipeline(
    folder,
    experiment,
    mice,
    male=(),
    female=(),
    events=('Cue', 'Press', 'Licks', 'Timeout Press'),
    epocs=('Po0_', 'Po6_', 'Po4_', 'Po2_'),
    timerange_cue=(-2, 5),
    timerange_lever=(-2, 5),
    timerange_lick=(-2, 10),
    active_time=(-2, 30),
    timerange_iti=(-40, -10),
    lick_bout_interval=0.4,
    max_trials_per_session=10,
):
    # storage (list-of-dicts; converted to DF at end)
    all_cue_trials = []
    all_lever_trials = []
    all_firstlick_trials = []
    all_lickbouts = []
    all_trials = []
    all_iti_trials = []

    lickspersession = []
    responsepersession = []
    timeoutpersession = []

    for mouse in tqdm(mice, desc="Processing mice"):
        mouse_dir = os.path.join(folder, experiment, mouse)
        dates = sorted([x for x in os.listdir(mouse_dir) if x.isnumeric()])
        session_map = {date: i for i, date in enumerate(dates)}

        sex = "U"
        if mouse in male:
            sex = "M"
        elif mouse in female:
            sex = "F"

        for date in dates:
            session_idx = session_map[date]
            date_dir = os.path.join(mouse_dir, date)

            try:
                data = tdt.read_block(date_dir)
            except Exception as e:
                print(f"[WARN] failed to read {date_dir}: {e}")
                continue

            # -------- Signal --------
            try:
                sig405 = data.streams._405B.data
                sig465 = data.streams._465B.data
                fs = float(data.streams._465B.fs)
            except Exception as e:
                print(f"[WARN] missing streams in {date_dir}: {e}")
                continue

            # compute dff (numpy array)
            
            trace, fs_used = process_signal_dff_fast(sig405, sig465, fs,
                                         lowpass_hz=20,
                                         fs_ds=100,
                                         fs_fit=10)
            fs = fs_used  # IMPORTANT: use this fs for all timestamp->index conversions

            

            # -------- Events --------
            fp_df = extract_events_fast(data, events, epocs)
            if fp_df.empty:
                continue

            track_cue = fp_df.loc[fp_df["Event"] == "Cue", "Timestamp"].to_numpy(dtype=float)
            track_lever = fp_df.loc[fp_df["Event"] == "Press", "Timestamp"].to_numpy(dtype=float)
            track_licks = fp_df.loc[fp_df["Event"] == "Licks", "Timestamp"].to_numpy(dtype=float)
            track_to = fp_df.loc[fp_df["Event"] == "Timeout Press", "Timestamp"].to_numpy(dtype=float)

            if track_cue.size == 0:
                continue

            # cap session to first N cues (your original behavior)
            if track_cue.size > max_trials_per_session:
                cutoff_cue_time = track_cue[max_trials_per_session]
                totallicks = track_licks[track_licks < cutoff_cue_time]
                totalresponse = track_lever[track_lever < cutoff_cue_time]
                # your original used cue[9] for timeout; keep safe:
                totimeout_cut = track_cue[max_trials_per_session - 1]
                totaltimeout = track_to[track_to < totimeout_cut]
                track_cue = track_cue[:max_trials_per_session]
            else:
                totallicks = track_licks
                totalresponse = track_lever
                totaltimeout = track_to[track_to < track_cue[-1]]

            lickspersession.append({"Mouse": mouse, "Session": session_idx, "Licks": int(totallicks.size), "Sex": sex})
            responsepersession.append({"Mouse": mouse, "Session": session_idx, "Responses": int(totalresponse.size), "Sex": sex})
            timeoutpersession.append({"Mouse": mouse, "Session": session_idx, "TimeoutPresses": int(totaltimeout.size), "Sex": sex})

            # -------- Precompute cue baseline stats (vector loop; uses numpy slices) --------
            cue_zero = np.rint(track_cue * fs).astype(int)
            cue_base_start = cue_zero + int(timerange_cue[0] * fs)
            cue_end = cue_zero + int(timerange_cue[1] * fs)

            cue_baseline_mean = np.empty(track_cue.size, dtype=float)
            cue_baseline_std = np.empty(track_cue.size, dtype=float)

            n_trace = trace.size
            for i in range(track_cue.size):
                bs = max(0, cue_base_start[i])
                bz = min(n_trace, cue_zero[i])
                if bz <= bs:
                    cue_baseline_mean[i] = np.nan
                    cue_baseline_std[i] = np.nan
                else:
                    seg = trace[bs:bz]
                    cue_baseline_mean[i] = float(seg.mean())
                    cue_baseline_std[i] = float(seg.std(ddof=0)) if seg.size > 1 else np.nan

                # store cue-aligned trace
                es = max(0, cue_base_start[i])
                ee = min(n_trace, cue_end[i])
                tr = trace[es:ee]
                mu = cue_baseline_mean[i]
                sd = cue_baseline_std[i]
                ztr = (tr - mu) / sd if np.isfinite(sd) and sd > 0 else np.full(tr.shape, np.nan)

                all_cue_trials.append({
                    "Mouse": mouse, "Session": session_idx, "Trial": i, "CueTime": float(track_cue[i]),
                    "Trace": ztr, "BaselineMean": mu, "BaselineStd": sd, "Sex": sex
                })

            # -------- LEVER ALIGNMENT (fast trial lookup) --------
            lever_cue_idx, lever_valid = find_trial_for_events_searchsorted(track_lever, track_cue, window=20.0)
            for lever_time, cue_i, ok in zip(track_lever, lever_cue_idx, lever_valid):
                if not ok:
                    continue
                mu = cue_baseline_mean[cue_i]
                sd = cue_baseline_std[cue_i]
                if not (np.isfinite(sd) and sd > 0):
                    continue

                z0 = int(np.rint(lever_time * fs))
                bs = max(0, z0 + int(timerange_lever[0] * fs))
                ee = min(n_trace, z0 + int(timerange_lever[1] * fs))
                tr = trace[bs:ee]
                ztr = (tr - mu) / sd

                all_lever_trials.append({
                    "Mouse": mouse, "Session": session_idx, "Trial": int(cue_i),
                    "LeverTime": float(lever_time), "Trace": ztr,
                    "BaselineMean": float(mu), "BaselineStd": float(sd), "Sex": sex
                })

            # -------- FIRST LICK ALIGNMENT --------
            firstlick_indices, lick_counts = identify_firstlicks_fast(track_lever, track_licks)
            track_flicks = track_licks[firstlick_indices] if firstlick_indices.size else np.array([], dtype=float)

            flick_cue_idx, flick_valid = find_trial_for_events_searchsorted(track_flicks, track_cue, window=30.0)
            for flick_time, cue_i, ok, nlick in zip(track_flicks, flick_cue_idx, flick_valid, lick_counts):
                if not ok:
                    continue
                mu = cue_baseline_mean[cue_i]
                sd = cue_baseline_std[cue_i]
                if not (np.isfinite(sd) and sd > 0):
                    continue

                z0 = int(np.rint(flick_time * fs))
                bs = max(0, z0 + int(timerange_lick[0] * fs))
                ee = min(n_trace, z0 + int(timerange_lick[1] * fs))
                tr = trace[bs:ee]
                ztr = (tr - mu) / sd

                all_firstlick_trials.append({
                    "Mouse": mouse, "Session": session_idx, "Trial": int(cue_i),
                    "FlickTime": float(flick_time), "Trace": ztr,
                    "BaselineMean": float(mu), "BaselineStd": float(sd),
                    "Licks": int(nlick), "Sex": sex
                })

            # -------- LICK BOUT ALIGNMENT --------
            sorted_bouts = identify_lick_bouts_fast(track_licks, max_interval=lick_bout_interval)
            if sorted_bouts:
                bout_starts = np.array([b[1][0] for b in sorted_bouts], dtype=float)
                bout_lengths = np.array([b[0] for b in sorted_bouts], dtype=int)

                bout_cue_idx, bout_valid = find_trial_for_events_searchsorted(bout_starts, track_cue, window=30.0)
                for start_time, blen, cue_i, ok in zip(bout_starts, bout_lengths, bout_cue_idx, bout_valid):
                    if not ok:
                        continue
                    mu = cue_baseline_mean[cue_i]
                    sd = cue_baseline_std[cue_i]
                    if not (np.isfinite(sd) and sd > 0):
                        continue

                    z0 = int(np.rint(start_time * fs))
                    bs = max(0, z0 + int(timerange_lick[0] * fs))
                    ee = min(n_trace, z0 + int(timerange_lick[1] * fs))
                    tr = trace[bs:ee]
                    ztr = (tr - mu) / sd

                    all_lickbouts.append({
                        "Mouse": mouse, "Session": session_idx, "Trial": int(cue_i),
                        "BoutLength": int(blen), "StartTime": float(start_time),
                        "Trace": ztr, "BaselineMean": float(mu), "BaselineStd": float(sd), "Sex": sex
                    })

            # -------- ALL TRIAL ALIGNMENT (cue -> active_time) --------
            for trial_num, cue_time in enumerate(track_cue):
                z0 = int(np.rint(cue_time * fs))
                bs = max(0, z0 + int(active_time[0] * fs))
                ee = min(n_trace, z0 + int(active_time[1] * fs))

                tr = trace[bs:ee]
                mu = float(np.mean(tr)) if tr.size else np.nan
                sd = float(np.std(tr, ddof=0)) if tr.size > 1 else np.nan
                ztr = (tr - mu) / sd if np.isfinite(sd) and sd > 0 else np.full(tr.shape, np.nan)

                # lever / flick times within windows (vector slices)
                lever_candidates = track_lever[(track_lever > cue_time) & (track_lever < cue_time + 20)]
                lever_time = float(lever_candidates[0]) if lever_candidates.size else np.nan

                flick_candidates = track_flicks[(track_flicks > cue_time) & (track_flicks < cue_time + 30)]
                flick_time = float(flick_candidates[0]) if flick_candidates.size else np.nan

                all_trials.append({
                    "Mouse": mouse, "Session": session_idx, "Trial": int(trial_num),
                    "CueTime": float(cue_time), "LeverTime": lever_time, "FlickTime": flick_time,
                    "Trace": ztr, "BaselineMean": mu, "BaselineStd": sd,
                    "Latency": (flick_time - cue_time) if np.isfinite(flick_time) else np.nan,
                    "Sex": sex
                })

            # -------- ITI ALIGNMENT --------
            for trial_num, cue_time in enumerate(track_cue):
                if trial_num == 0:
                    continue
                z0 = int(np.rint(cue_time * fs))
                bs = max(0, z0 + int(timerange_iti[0] * fs))
                ee = min(n_trace, z0 + int(timerange_iti[1] * fs))

                tr = trace[bs:ee]
                mu = float(np.mean(tr)) if tr.size else np.nan
                sd = float(np.std(tr, ddof=0)) if tr.size > 1 else np.nan
                ztr = (tr - mu) / sd if np.isfinite(sd) and sd > 0 else np.full(tr.shape, np.nan)

                # timeouts within ITI (vectorized)
                to_samp = track_to * fs
                mask_to = (to_samp >= bs) & (to_samp <= ee)
                trial_to = ((to_samp[mask_to] - bs) / fs).tolist()

                all_iti_trials.append({
                    "Mouse": mouse, "Session": session_idx, "Trial": int(trial_num),
                    "CueTime": float(cue_time),
                    "Trace": ztr,
                    "TimeOuts": trial_to,
                    "TimeoutLength": int(len(trial_to)),
                    "Sex": sex
                })

    # ========================= CONVERT TO DATAFRAMES =========================
    avgcuetrace_df = pd.DataFrame(all_cue_trials)
    avglevertrace_df = pd.DataFrame(all_lever_trials)
    avgflicktrace_df = pd.DataFrame(all_firstlick_trials)
    avglickbouttrace_df = pd.DataFrame(all_lickbouts)
    alltrialtrace_df = pd.DataFrame(all_trials)
    allititrace_df = pd.DataFrame(all_iti_trials)

    # ========================= BEHAVIORAL DATA =========================
    if not alltrialtrace_df.empty:
        alltrialtrace_df["LeverLate"] = alltrialtrace_df["LeverTime"] - alltrialtrace_df["CueTime"]
        alltrialtrace_df["LickLate"] = alltrialtrace_df["FlickTime"] - alltrialtrace_df["LeverTime"]

        leverlatency_matrix_df = alltrialtrace_df.pivot_table(
            index=["Mouse", "Trial"], columns="Session", values="LeverLate", aggfunc="first"
        )
        licklatency_matrix_df = alltrialtrace_df.pivot_table(
            index=["Mouse", "Trial"], columns="Session", values="LickLate", aggfunc="first"
        )
        behavioral_licklatency_matrix_df = alltrialtrace_df.pivot_table(
            index=["Mouse"], columns="Session", values="LickLate", aggfunc="mean"
        )
    else:
        leverlatency_matrix_df = pd.DataFrame()
        licklatency_matrix_df = pd.DataFrame()
        behavioral_licklatency_matrix_df = pd.DataFrame()

    behavioral_licks_df = pd.DataFrame(lickspersession)
    behavioral_response_df = pd.DataFrame(responsepersession)
    behavioral_timeout_df = pd.DataFrame(timeoutpersession)

    behavioral_licks_matrix_df = behavioral_licks_df.pivot_table(
        index=["Mouse"], columns="Session", values="Licks", aggfunc="first"
    ) if not behavioral_licks_df.empty else pd.DataFrame()

    behavioral_response_matrix_df = behavioral_response_df.pivot_table(
        index=["Mouse"], columns="Session", values="Responses", aggfunc="first"
    ) if not behavioral_response_df.empty else pd.DataFrame()

    behavioral_timeout_matrix_df = behavioral_timeout_df.pivot_table(
        index=["Mouse"], columns="Session", values="TimeoutPresses", aggfunc="first"
    ) if not behavioral_timeout_df.empty else pd.DataFrame()

    return {
        "avgcuetrace_df": avgcuetrace_df,
        "avglevertrace_df": avglevertrace_df,
        "avgflicktrace_df": avgflicktrace_df,
        "avglickbouttrace_df": avglickbouttrace_df,
        "alltrialtrace_df": alltrialtrace_df,
        "allititrace_df": allititrace_df,
        "leverlatency_matrix_df": leverlatency_matrix_df,
        "licklatency_matrix_df": licklatency_matrix_df,
        "behavioral_licklatency_matrix_df": behavioral_licklatency_matrix_df,
        "behavioral_licks_df": behavioral_licks_df,
        "behavioral_response_df": behavioral_response_df,
        "behavioral_timeout_df": behavioral_timeout_df,
        "behavioral_licks_matrix_df": behavioral_licks_matrix_df,
        "behavioral_response_matrix_df": behavioral_response_matrix_df,
        "behavioral_timeout_matrix_df": behavioral_timeout_matrix_df,
    }


# ========================= RUN (example) =========================
results = run_pipeline(folder=folder, experiment=experiment, mice=mice, male=male, female=female)
avgcuetrace_df = results["avgcuetrace_df"]
alltrialtrace_df = results["alltrialtrace_df"]