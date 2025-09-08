# SpO2 Estimation from PPG (Photoplethysmography) Waveforms
*Dataset Used*: https://physionet.org/content/pulse-transit-time-ppg/1.1.0/

## 1. PPG Signal Preprocessing & Identifying Beats

### Results to be Achieved
- Select **good quality pulses** while rejecting noise.
- Prepare signals for feature extraction.

### Tasks
1. **Baseline Wander Removal**
   - Remove DC offset using a rolling Gaussian baseline filter (~0.5s window).

2. **Bandpass Filtering**
   - Apply a 3rd-order Butterworth bandpass filter (0.75–5 Hz) to keep only physiological heart rate components and remove high-frequency noise and motion artifacts.

3. **Beat Identification**
   - Use `scipy.signal.find_peaks` to detect **systolic peaks**.
   - Enforce a physiological minimum distance (~0.5s at 500Hz) between peaks.

4. **Good Pulse Selection**
   - Discard beats with unrealistic intervals (outliers in inter-beat intervals).
   - Optionally use accelerometer/gyroscope/load-cell signals to flag motion-corrupted windows.

---

## 2. Pulse Waveform Feature Extraction

### Results to be Achieved
Extract morphological and temporal features from each clean beat.

### Tasks
1. **Identify Key Landmarks**
   - **Systolic Peak**: maximum in each beat.
   - **Dicrotic Notch**: local minimum after systolic peak.
   - **Diastolic Peak**: local maximum after notch (if present).

2. **Feature Computation (per beat)**
   - Amplitude features: systolic amplitude, diastolic amplitude, notch amplitude.
   - Timing features: systolic time, notch time, diastolic time.
   - Derived metrics: rise time (foot → systolic), pulse width, inter-beat interval.
   - Heart rate: computed from consecutive peak intervals.

3. **SpO2 Estimation (per beat)**
   - Use red (`pleth_1`) and infrared (`pleth_2`) PPG signals.
   - Compute the ratio-of-ratios (R):

     ```
     R = (AC_red / DC_red) / (AC_ir / DC_ir)
     ```

   - Estimate SpO2 with calibration constants:

     ```
     SpO2 = A - B * R
     ```

   - Note: due to noise, unrealistic values may appear.

---

## 3. Train the ML Model for SpO2

### Results to be Achieved
Develop a model that predicts **SpO2 from PPG features** while being robust to subject variability and motion.

### Tasks
1. **Dataset Preparation**
   - Merge **subjects_info.csv** (demographics, reference SpO2) with **beat-level features**.
   - Aggregate features per file (mean, std).
   - Use reference SpO2 (average of `spo2_start` and `spo2_end`) as the ground truth.

2. **Feature Engineering**
   - Include morphological features (amp/timing).
   - Add subject metadata (age, height, weight, gender).
   - Encode categorical variables (`activity`, `gender`) via one-hot encoding.
   - Normalize numeric features (scaling to zero mean, unit variance).

3. **Model Training & Evaluation**
   - Models tested: **Linear Regression, Support Vector Regression (SVR), Random Forest Regressor**.
   - Use **5-fold cross-validation** on 80% training data to optimize hyperparameters.
   - Final evaluation on 20% hold-out test set.
   - Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

4. **Hyperparameter Tuning**
   - Apply `GridSearchCV` to optimize Random Forest parameters.
   - Save the best model and its predictions.

---

## Project Structure

manualoutputs\\                \# *Extracted beat-level features and manual SpO2 estimates (per subject, per activity)*

notebooks\\
  EDA_01.ipynb                 \# *Exploratory data analysis*
  explanations.ipynb           \# *Detailed explanations / notes*
  ml_modelling.ipynb           \# *ML model training & testing experiments*
  run_pipeline.ipynb           \# *End-to-end manual pipeline execution*

src\\
  build_dataset.py             \# *Combines extracted features + subject metadata for ML training*
  feature_extraction.py        \# *Detects systolic, diastolic, notch peaks & features*
  load_data.py                 \# *Loads raw signal data & metadata*
  manual_pipeline.py           \# *Builds feature DataFrames (manualoutputs/)*
  preprocess.py                \# *Preprocessing (filtering, DC removal, etc.)*
  spo2_estimation.py           \# *R-ratio SpO2 estimation*

.gitignore                     \# *Ignore unnecessary files*
README.md                      \# *Project description, setup, usage*
best_rf_model.pkl              \# *Saved Random Forest model (best tuned one)*
ml_dataset.csv                 \# *Aggregated dataset used for ML training*
rf_predictions.csv             \# *Predictions vs Ground Truth SpO2*
