import pandas as pd
import glob, os

# Load subject metadata
meta = pd.read_csv("../data/csv/subjects_info.csv")

rows = []
for file in glob.glob("../manualoutputs/*_features.csv"):
    df = pd.read_csv(file)

    # Aggregate features
    feats_mean = df.mean(numeric_only=True, skipna=True)
    feats_std = df.std(numeric_only=True, skipna=True)

    # keep only selected features
    keep_cols = ["systolic_amp", "diastolic_amp", "notch_amp",
                 "pulse_width", "rise_time", "heart_rate"]
    feats = {}
    for col in keep_cols:
        feats[f"{col}_mean"] = feats_mean.get(col, None)
        feats[f"{col}_std"] = feats_std.get(col, None)

    # Adding metadata
    fname = os.path.basename(file).replace("_features.csv", "")
    row_meta = meta[meta["record"] == fname].iloc[0]

    feats["spo2_ref"] = (row_meta["spo2_start"] + row_meta["spo2_end"]) / 2
    feats["activity"] = row_meta["activity"]
    feats["gender"] = row_meta["gender"]
    feats["age"] = row_meta["age"]
    feats["height"] = row_meta["height"]
    feats["weight"] = row_meta["weight"]

    rows.append(feats)

ml_df = pd.DataFrame(rows)
ml_df.to_csv("../ml_dataset.csv", index=False)
