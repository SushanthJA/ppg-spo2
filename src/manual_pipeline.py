import os
import pandas as pd
from load_data import load_signal_csv, load_metadata_csv
from preprocess import preprocess_ppg
from feature_extraction import detect_systolic_peaks, detect_dicrotic_notch, detect_diastolic_peaks, build_feature_dataframe
from spo2_estimation import extract_spo2_features

DATA_DIR = "../data/csv"           # folder containing csv files
OUTPUT_DIR = "../outputs"       # folder to save per-file feature CSVs
META_FILE = os.path.join(DATA_DIR, "subject_info.csv")
FS = 500

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_file(filepath, A=110, B=25):
    print(f"Processing {filepath}...")
    df = load_signal_csv(filepath)
    df = preprocess_ppg(df, channels=["pleth_1", "pleth_2"], fs=FS)

    # pick filtered red/IR
    red = df["pleth_1_filtered"].values
    ir = df["pleth_2_filtered"].values

    # beat detection for segmentation
    systolic_peaks = detect_systolic_peaks(pd.Series(red), fs=FS)

    # build feature dataframe
    feature_df = build_feature_dataframe(pd.Series(red), fs=FS)

    # build beat windows from systolic peaks
    beats = []
    for i in range(len(systolic_peaks) - 1):
        beats.append((systolic_peaks[i], systolic_peaks[i+1]))

    # compute R & SpO2 per beat
    spo2_df = extract_spo2_features(red, ir, beats, A=A, B=B)

    # merge features + SpO2
    merged = pd.merge(feature_df, spo2_df, left_index=True, right_on="beat_idx", how="inner")

    return merged

def run_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, A=110, B=25):
    ensure_dir(output_dir)

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "subjects_info.csv"]
    for file in files:
        filepath = os.path.join(data_dir, file)
        result = process_file(filepath, A, B)

        outname = os.path.splitext(file)[0] + "_features.csv"
        outpath = os.path.join(output_dir, outname)
        result.to_csv(outpath, index=False)
        print(f"Saved {outpath}")
