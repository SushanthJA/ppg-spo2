import os
import pandas as pd
import matplotlib.pyplot as plt

def load_signal_csv(filepath: str):
    """
    Load a single signal CSV file.
    """
    return pd.read_csv(filepath)


def load_metadata_csv(filepath: str):
    """
    Load metadata file (subjects_info.csv).
    """
    return pd.read_csv(filepath)


def plot_ppg(df, channels=['pleth_1', 'pleth_2'], seconds=10, fs=500):
    """
    Quick visualization of PPG channels.
    
    Args:
        df (pd.DataFrame): Signal dataframe.
        channels (list): List of pleth channels to plot.
        seconds (int): How many seconds to display.
        fs (int): Sampling frequency (default 500 Hz).
    """
    n_samples = seconds * fs
    segment = df.iloc[:n_samples]

    plt.figure(figsize=(12, 5))
    for ch in channels:
        if ch in df.columns:
            plt.plot(segment.index, segment[ch], label=ch, alpha=0.7)

    plt.title(f"PPG signals (first {seconds} seconds)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def check_missing_values(data_dir):
    results = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            path = os.path.join(data_dir, file)
            df = pd.read_csv(path)
            results[file] = df.isna().sum().to_dict()
            
    for fname, na_counts in results.items():
        print(f"\nFile: {fname}")
        for col, count in na_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing")
            else:
                print(f"No missing items detected")

    return results