import numpy as np
import pandas as pd

def compute_ac_dc(signal):
    """
    Compute AC (pulsatile amplitude) and DC (baseline) for a signal segment.
    """
    dc = np.mean(signal)  # baseline
    ac = np.max(signal) - np.min(signal)  # peak-to-peak amplitude
    return ac, dc


def compute_r_ratio(red_signal, ir_signal):
    """
    Compute R-ratio for given red and infrared signals.
    Both signals should be aligned and segmented (beat or window).
    """
    ac_red, dc_red = compute_ac_dc(red_signal)
    ac_ir, dc_ir = compute_ac_dc(ir_signal)
    
    if dc_red == 0 or dc_ir == 0 or ac_ir == 0:
        return np.nan  # avoid division by zero
    
    r = (ac_red / dc_red) / (ac_ir / dc_ir)
    return r


def estimate_spo2(r, A=110.0, B=25.0):
    """
    Estimate SpO2 from R-ratio using empirical calibration constants.
    Defaults: A=110, B=25.
    """
    if np.isnan(r):
        return np.nan
    spo2 = A - B * r
    return spo2


def extract_spo2_features(red_signal, ir_signal, beats, A=110.0, B=25.0):
    """
    Compute R-ratio and estimated SpO2 per beat.
    
    Parameters:
    - red_signal: np.array, raw red PPG channel
    - ir_signal: np.array, raw IR PPG channel
    - beats: list of (start_idx, end_idx) tuples from peak detection
    - A: pulsatile (AC) component of the PPG (the beat-to-beat variation).
    - B:  baseline (DC) component of the PPG (the slowly varying tissue/skin absorption).
    
    Returns:
    - DataFrame with columns: [beat_idx, R_ratio, SpO2_est]
    """
    results = []
    for i, (start, end) in enumerate(beats):
        red_seg = red_signal[start:end]
        ir_seg = ir_signal[start:end]
        
        r = compute_r_ratio(red_seg, ir_seg)
        spo2 = estimate_spo2(r, A, B)
        
        results.append({
            "beat_idx": i,
            "R_ratio": r,
            "SpO2_est": spo2
        })
    
    return pd.DataFrame(results)
