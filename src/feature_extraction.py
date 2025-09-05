import numpy as np
import pandas as pd
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


def detect_diastolic_peaks(ppg, notches, systolic_peaks):
    diastolic_peaks = []
    for i in range(len(notches)):
        if i+1 < len(systolic_peaks):
            segment = ppg[notches[i]:systolic_peaks[i+1]]
            if len(segment) > 0:
                dia_idx = np.argmax(segment) + notches[i]
                diastolic_peaks.append(dia_idx)
    return np.array(diastolic_peaks)


def compute_heart_rate(systolic_peaks, fs=500):
    if len(systolic_peaks) < 2:
        return np.nan
    rr_intervals = np.diff(systolic_peaks) / fs  # seconds
    hr = 60.0 / np.mean(rr_intervals)
    return hr


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

    for i in range(len(diastolic_peaks)):
        sys_idx = systolic_peaks[i]
        notch_idx = notches[i] if i < len(notches) else None
        dia_idx = diastolic_peaks[i]

        if notch_idx is None:
            continue

        # Amplitudes
        systolic_amp = ppg[sys_idx]
        notch_amp = ppg[notch_idx]
        diastolic_amp = ppg[dia_idx]

        # Times
        systolic_time = sys_idx / fs
        notch_time = notch_idx / fs
        diastolic_time = dia_idx / fs

        # Derived features
        rise_time = notch_time - systolic_time
        pulse_width = diastolic_time - systolic_time
        hr = compute_heart_rate(systolic_peaks, fs)

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
            "heart_rate_bpm": hr
        })

    df_features = pd.DataFrame(features)
    return df_features
