from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/csv/")

def list_recordings():
    files = sorted([p for p in DATA_DIR.glob("s*_*.csv")])
    print(f"Found {len(files)} recording CSVs. Example {files[:5]}\n\n")
    return files

def load_metadata():
    meta = pd.read_csv(DATA_DIR/"subjects_info.csv")
    # print("Metadata columns:", meta.columns.tolist())
    return meta

def load_record(rec_path):
    df = pd.read_csv(rec_path)
    return df

if __name__ == "__main__":
    recs = list_recordings()
    meta = load_metadata()
    df = load_record(recs[0])