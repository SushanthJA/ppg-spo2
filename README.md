\# SpO2 Estimation from PPG (Photoplethysmography) Waveforms



\## \*\*1. PPG Signal Preprocessing \& Identifying Beats\*\*



\### Results to be Achieved



\* Select \*\*good quality pulses\*\* while rejecting noise.

\* Prepare signals for feature extraction.



\### Tasks



1\. \*\*Baseline Wander Removal\*\*



&nbsp;  \* Remove DC offset using a rolling Gaussian baseline filter (\\~0.5s window).



2\. \*\*Bandpass Filtering\*\*



&nbsp;  \* Apply a 3rd-order Butterworth bandpass filter (0.75–5 Hz) to keep only physiological heart rate components and remove high-frequency noise and motion artifacts.



3\. \*\*Beat Identification\*\*



&nbsp;  \* Use `scipy.signal.find\_peaks` to detect \*\*systolic peaks\*\*.

&nbsp;  \* Enforce a physiological minimum distance (\\~0.5s at 500Hz) between peaks.



4\. \*\*Good Pulse Selection\*\*



&nbsp;  \* Discard beats with unrealistic intervals (outliers in inter-beat intervals).

&nbsp;  \* Optionally use accelerometer/gyroscope/load-cell signals to flag motion-corrupted windows.



---



\## \*\*2. Pulse Waveform Feature Extraction\*\*



\### Results to be Achieved



Extract morphological and temporal features from each clean beat.



\### Tasks



1\. \*\*Identify Key Landmarks\*\*



&nbsp;  \* \*\*Systolic Peak\*\*: maximum in each beat.

&nbsp;  \* \*\*Dicrotic Notch\*\*: local minimum after systolic peak.

&nbsp;  \* \*\*Diastolic Peak\*\*: local maximum after notch (if present).



2\. \*\*Feature Computation (per beat)\*\*



&nbsp;  \* Amplitude features: systolic amplitude, diastolic amplitude, notch amplitude.

&nbsp;  \* Timing features: systolic time, notch time, diastolic time.

&nbsp;  \* Derived metrics: rise time (foot → systolic), pulse width, inter-beat interval.

&nbsp;  \* Heart rate: computed from consecutive peak intervals.



3\. \*\*SpO2 Estimation (per beat)\*\*



&nbsp;  \* Use red (pleth\\\_1) and infrared (pleth\\\_2) PPG signals.

&nbsp;  \* Compute the ratio-of-ratios (R):



&nbsp;    $$

&nbsp;    R = \\frac{(AC\_{red} / DC\_{red})}{(AC\_{ir} / DC\_{ir})}

&nbsp;    $$

&nbsp;  \* Estimate SpO2 with calibration constants:



&nbsp;    $$

&nbsp;    SpO2 = A - B \\cdot R

&nbsp;    $$

&nbsp;  \* Note: due to noise, unrealistic values may appear.



---



\## \*\*3. Train the ML Model for SpO2\*\*



\### Results to be Achieved



Develop a model that predicts \*\*SpO2 from PPG features\*\* while being robust to subject variability and motion.



\### Tasks



1\. \*\*Dataset Preparation\*\*



&nbsp;  \* Merge \*\*subjects\\\_info.csv\*\* (demographics, reference SpO2) with \*\*beat-level features\*\*.

&nbsp;  \* Aggregate features per file (mean, std).

&nbsp;  \* Use reference SpO2 (average of `spo2\_start` and `spo2\_end`) as the ground truth.



2\. \*\*Feature Engineering\*\*



&nbsp;  \* Include morphological features (amp/timing).

&nbsp;  \* Add subject metadata (age, height, weight, gender).

&nbsp;  \* Encode categorical variables (`activity`, `gender`) via one-hot encoding.

&nbsp;  \* Normalize numeric features (scaling to zero mean, unit variance).



3\. \*\*Model Training \& Evaluation\*\*



&nbsp;  \* Models tested: \*\*Linear Regression, Support Vector Regression (SVR), Random Forest Regressor\*\*.

&nbsp;  \* Use \*\*5-fold cross-validation\*\* on 80% training data to optimize hyperparameters.

&nbsp;  \* Final evaluation on 20% hold-out test set.

&nbsp;  \* Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).



4\. \*\*Hyperparameter Tuning\*\*



&nbsp;  \* Apply `GridSearchCV` to optimize Random Forest parameters.

&nbsp;  \* Save the best model and its predictions.





\## Project Structure



\\manualoutputs		  # Extracted beat-level features and corresponding manual SpO2 estimates for each subject across all activities



\\notebooks

&nbsp;  EDA\_01.ipynb           # Exploratory data analysis

&nbsp;  explanations.ipynb     # Detailed explanations / notes

&nbsp;  ml\_modelling.ipynb     # ML model training \& testing experiments

&nbsp;  run\_pipeline.ipynb     # End-to-end manual pipeline execution (manual\_pipeline.py)



\\src

&nbsp;  build\_dataset.py       # Combines extracted features + subject metadata for (building data for ML model training)

&nbsp;  feature\_extraction.py  # Detects systolic, diastolic, notch peaks \& features

&nbsp;  load\_data.py           # Loads raw signals \& metadata from dataset

&nbsp;  manual\_pipeline.py     # Step-by-step pipeline to build feature dataframes (\\manualoutputs) for each subject task

&nbsp;  preprocess.py          # Preprocessing (filtering, DC removal, etc.)

&nbsp;  spo2\_estimation.py     # R-ratio SpO2 estimation



.gitignore                # Ignore unnecessary files (cache, env, etc.)

README.md                 # Project description, setup, usage

best\_rf\_model.pkl         # Saved Random Forest model (best tuned one)

ml\_dataset.csv            # Aggregated dataset used for ML training

rf\_predictions.csv        # Comparison of Predictions from the trained Random Forest model to the Ground Truth SpO2





