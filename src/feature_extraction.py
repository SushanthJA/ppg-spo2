import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def detect_systolic_peaks(ppg, fs=500):
    distance = int(0.5 * fs)  # enforce ~0.5s between beats
    systolic_peaks, _ = find_peaks(ppg, distance=distance, prominence=0.5)
    return systolic_peaks


def detect_dicrotic_notch(ppg, systolic_peaks):
    notches = []
    for i in range(len(systolic_peaks) - 1):
        segment = ppg[systolic_peaks[i]:systolic_peaks[i+1]]
        if len(segment) > 0:
            notch_idx = np.argmin(segment) + systolic_peaks[i]
            notches.append(notch_idx)
    return np.array(notches)


# def detect_diastolic_peaks(ppg, notches, systolic_peaks):
#     diastolic_peaks = []
#     for i in range(len(notches)):
#         rr_interval = systolic_peaks[i+1] - systolic_peaks[i]
#         local_idx = None
#         encountered_else = False
#         search_end = systolic_peaks[i+1] - int(0.15*rr_interval) # unrealistic to have diastolic peak after this
#         segment = ppg[notches[i]:search_end]

#         if len(segment) > 0:
#             for j in range(len(segment)-1):
#                 if segment.iloc[j+1] > segment.iloc[j]:
#                     local_idx = j+1
#                 else:
#                     encountered_else = True
#                     break
#             if encountered_else:
#                 dia_idx = local_idx + notches[i]
#             else:
#                 dia_idx = np.nan
#             diastolic_peaks.append(dia_idx)
#     return np.array(diastolic_peaks)


def detect_diastolic_peaks(ppg, notches, systolic_peaks, fs=500, min_prom=0.02):
    diastolic_peaks = []
    for i in range(len(notches)):
        segment = ppg.iloc[notches[i]:systolic_peaks[i+1]]

        # search peaks in this window
        peaks, _ = find_peaks(segment, prominence=min_prom)

        if len(peaks) > 0:
            # take the first peak after notch
            dia_idx = peaks[0] + notches[i]
            diastolic_peaks.append(dia_idx)
        else:
            diastolic_peaks.append(np.nan)

    return np.array(diastolic_peaks)


def build_feature_dataframe(ppg, fs=500):
    """
    Extracts beat-level features from PPG and returns a DataFrame.
    
    Columns include:
    - systolic_amp, diastolic_amp, notch_amp
    - systolic_time, diastolic_time, notch_time
    - pulse_width, rise_time, HR
    """
    systolic_peaks = detect_systolic_peaks(ppg, fs)
    notches = detect_dicrotic_notch(ppg, systolic_peaks)
    diastolic_peaks = detect_diastolic_peaks(ppg, notches, systolic_peaks)

    features = []

    for i in range(len(notches)):  
        sys_idx = systolic_peaks[i]          # current systolic
        next_sys_idx = systolic_peaks[i+1]   # next systolic (for HR interval)
        notch_idx = notches[i]
        dia_idx = diastolic_peaks[i]
    
        # Amplitudes
        systolic_amp = ppg[sys_idx]
        notch_amp = ppg[notch_idx] if notch_idx is not None else np.nan
        diastolic_amp = ppg[dia_idx] if not np.isnan(dia_idx) else np.nan
    
        # Times
        systolic_time = sys_idx / fs
        notch_time = notch_idx / fs if notch_idx is not None else np.nan
        diastolic_time = dia_idx / fs if not np.isnan(dia_idx) else np.nan
    
        # Derived features
        rise_time = (notch_time - systolic_time) if notch_idx is not None else np.nan
        pulse_width = (diastolic_time - systolic_time) if not np.isnan(diastolic_time) else np.nan
    
        # Heart rate for this beat = interval to next systolic
        rr_interval = (next_sys_idx - sys_idx) / fs
        hr = 60.0 / rr_interval if rr_interval > 0 else np.nan
    
        features.append({
            "sys_idx": sys_idx,
            "notch_idx": notch_idx,
            "dia_idx": dia_idx,
            "systolic_amp": systolic_amp,
            "notch_amp": notch_amp,
            "diastolic_amp": diastolic_amp,
            "systolic_time": systolic_time,
            "notch_time": notch_time,
            "diastolic_time": diastolic_time,
            "rise_time": rise_time,
            "pulse_width": pulse_width,
            "heart_rate": hr
        })

    df_features = pd.DataFrame(features)
    return df_features


def plot_ppg_with_features(ppg, fs, systolic_peaks, diastolic_peaks, notches, seconds=5, title="PPG (first {seconds} seconds)"):
    
    max_samples = int(seconds * fs)
    t = np.arange(len(ppg)) / fs
    ppg = ppg[:max_samples]
    t = t[:max_samples]

    systolic_peaks = systolic_peaks[systolic_peaks < max_samples]
    diastolic_peaks = diastolic_peaks[diastolic_peaks < max_samples]
    notches = notches[notches < max_samples]

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(t, ppg, label="Filtered PPG", color="black", linewidth=1)

    plt.plot(systolic_peaks / fs, ppg[systolic_peaks], "ro", label="Systolic peaks")
    plt.plot(diastolic_peaks / fs, ppg[diastolic_peaks], "go", label="Diastolic peaks")
    plt.plot(notches / fs, ppg[notches], "bo", label="Dicrotic notches")

    plt.title(f"{title} (first {seconds}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

