import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def remove_dc(sig, window=250):
    """
    Remove DC component using Gaussian rolling baseline.
    Args:
        sig (np.ndarray): Input signal.
        window (int): Window length (in samples). 
                      For fs=500, window=250 => 0.5s.
    """
    baseline = gaussian_filter1d(sig, sigma=window)
    return sig - baseline


def bandpass_filter(sig, fs=500, lowcut=0.75, highcut=5, order=3):
    """
    Apply Butterworth bandpass filter to retain only heart-rate related frequencies.
    Args:
        sig (np.ndarray): Input signal.
        fs (int): Sampling frequency (Hz).
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        order (int): Filter order.
    nyq: Nyquist Frequency
    """
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def preprocess_ppg(df, channels=['pleth_1', 'pleth_2'], fs=500):
    """
    Apply DC removal and bandpass filtering to PPG channels.
    Returns the DataFrame with pleth_filtered columns.
    """
    for ch in channels:
        if ch in df.columns:
            sig = df[ch].values
            sig = remove_dc(sig, window=int(0.5 * fs))
            sig = bandpass_filter(sig, fs=fs)
            df[ch + "_filtered"] = sig
    return df


def plot_ppg_filtered(df, channel='pleth_1', seconds=10, fs=500):
    """
    Plot raw vs filtered PPG.
    """
    n_samples = seconds * fs
    segment = df.iloc[:n_samples]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Raw on left axis
    if channel in df.columns:
        ax1.plot(segment.index, segment[channel], color="steelblue", alpha=0.7, label=f"{channel} (raw)")
    ax1.set_ylabel("Raw amplitude", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Filtered on right axis
    ax2 = ax1.twinx()
    if channel + "_filtered" in df.columns:
        ax2.plot(segment.index, segment[channel + "_filtered"], color="darkorange", linewidth=2, label=f"{channel} (filtered)")
    ax2.set_ylabel("Filtered amplitude", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.title(f"{channel}: Raw vs Filtered (first {seconds} seconds)")
    fig.tight_layout()
    plt.show()
