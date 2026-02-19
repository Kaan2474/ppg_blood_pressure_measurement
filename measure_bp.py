import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import resample
from motion_artifact_removal import butterworth_lowpass_filter, sqa_autocorrelation
from feature_extraction import extract_features


INPUT_FILE = "aligned_signals_walking_1.csv"
OUTPUT_FILE = "blood_pressure_results_walking_sqa_1.csv"
POLAR_SAMPLING_RATE = 176.0
MIMIC_SAMPLING_RATE = 125.0 # Expected from the ANN model
WINDOW_SIZE = int(5 * MIMIC_SAMPLING_RATE) # 625 samples


def load_ppg_signals(file):
    """
    Loads PPG signals obtained from the Polar Verity Sense

    Args:
        file: Contains PPG signals

    Returns:
        ppg_signals: Raw, averaged, and inverted PPG signals
    """
    print(f"Loading {file}...")
    df = pd.read_csv(file)
    ppg_signals = -1 * (df['PPG_0'] + df['PPG_1'] + df['PPG_2']) / 3.0
    return ppg_signals


def resample_ppg_signals(ppg_signals):
    """
    Adjusts the sampling rate of the PPG signals derived from the Polar Verity Sense to match the MIMIC database:
        - Changes the sampling rate from 176 Hz to 125 Hz

    Args:
        ppg_signals: PPG signals from the Polar Verity Sense

    Returns:   
        ppg_resampled: PPG signals with adjusted sampling rate of 125 Hz
    """
    num_samples = int(len(ppg_signals) * MIMIC_SAMPLING_RATE / POLAR_SAMPLING_RATE)
    print(f"Resampling from {POLAR_SAMPLING_RATE} Hz to {MIMIC_SAMPLING_RATE} Hz...")
    ppg_resampled = resample(ppg_signals, num_samples)
    return ppg_resampled


def predict_blood_pressure():
    # 1. Load PPG signals
    ppg_signals = load_ppg_signals(INPUT_FILE)

    # 3. Adjust sampling rate of PPG signals from 176 Hz to 125 Hz
    ppg_resampled = resample_ppg_signals(ppg_signals)

    # 2. Motion artifact removal with lowpass filter
    # ppg_low_pass = butterworth_lowpass_filter(ppg_resampled, 12.0, MIMIC_SAMPLING_RATE, 4)
    
    # 4. Load ANN model
    print("Loading Model...")
    MODEL_FILE = "ann_bp_model.pkl"
    SCALER_FILE = "scaler.pkl"
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    predictions = []
    
    print("Estimating blood pressure...")
    
    # 5. Blood pressure estimation
    step_size = int(MIMIC_SAMPLING_RATE * 1.0) # Steps 1 second at a time
    
    for i in range(0, len(ppg_resampled) - WINDOW_SIZE, step_size):
        window = ppg_resampled[i : i+WINDOW_SIZE]
        normalized_window = (window - np.mean(window)) / np.std(window) # Scales the signal magnitude differences
        if sqa_autocorrelation(normalized_window, MIMIC_SAMPLING_RATE):
            features = extract_features(normalized_window, MIMIC_SAMPLING_RATE)
            # D. Predict
            if features[0] != 0: # Check if extraction succeeded
                # Scale features (Must use the SAME scaler as training)
                scaled_features = scaler.transform([features])
                # Predict [SBP, DBP]
                bp_pred = model.predict(scaled_features)[0]
                # Save result (Convert sample index 'i' back to seconds)
                time_sec = i / MIMIC_SAMPLING_RATE
                predictions.append([time_sec, bp_pred[0], bp_pred[1]])

    if len(predictions) == 0:
        print("No valid windows found!")
        return

    # 6. Blood pressure results
    results_df = pd.DataFrame(predictions, columns=['Time in sec', 'SBP', 'DBP'])
    print(f"\nEstimation Complete! Generated {len(results_df)} BP readings.")
    print(results_df.head())

    # 7. Save to CSV
    results_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    predict_blood_pressure()