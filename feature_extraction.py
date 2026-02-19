from scipy.signal import find_peaks, periodogram
import numpy as np
from motion_artifact_removal import sqa_autocorrelation


def extract_features(ppg_window, sampling_frequency):
    """
    Extracts the 22 features from each PPG window based on Wang et al. (2018):
        1. 2 morphological features: systolic upstroke time (sut), diastolic time (dt)
        2. 20 spectral features: power in 0.5 Hz chunks from 0 - 10 Hz
    
    Args:
        ppg_window: The 5-second PPG segment
        sampling_frequency: The sampling frequency of the data record (125 Hz for MIMIC data)

    Returns:
        features: All 22 features combined
    """
    
    # 1. Morphological features

    systolic_peaks, _ = find_peaks(ppg_window, height=0, distance=sampling_frequency/2.5)
    diastolic_peaks, _ = find_peaks(-ppg_window, distance=sampling_frequency/2.5)
    
    sut_values = []
    dt_values = []
    
    # Evaluates if signal cycles are valid
    # Valid signal cycle: current_diastolic_peak → systolic_peak → next_diastolic_peak.
    for i in range(len(diastolic_peaks) - 1):
        current_diastolic_peak = diastolic_peaks[i]
        next_diastolic_peak = diastolic_peaks[i+1]
        validated_systolic_peaks = systolic_peaks[(systolic_peaks > current_diastolic_peak) & (systolic_peaks < next_diastolic_peak)]

        # The signal cycle is only valid if exactly one systolic peak exists
        if len(validated_systolic_peaks) == 1:
            systolic_peak = validated_systolic_peaks[0]
            
            # Calculates systolic upstroke time
            sut = (systolic_peak - current_diastolic_peak) / sampling_frequency
            sut_values.append(sut)
            
            # Calculates diastolic time
            dt = (next_diastolic_peak - systolic_peak) / sampling_frequency
            dt_values.append(dt)
    
    # Averages the systolic upstroke time and diastolic time
    if len(sut_values) > 0:
        avg_sut = np.mean(sut_values)
        avg_dt = np.mean(dt_values)
    else:
        return np.zeros(22) # Return empty feature vector if detection fails

    # 2. Spectral features
    
    # Converts the PPG window from time domain to frequency domain
    f, Pxx = periodogram(ppg_window, sampling_frequency, scaling='spectrum')
    
    spectral_features = []
    
    # Slices the signal into 0.5 Hz chunks
    # Iteration 1: 0.0 - 0.5 Hz
    # Iteration 2: 0.5 - 1.0 Hz
    # ...
    # Iteration 20: 9.5 - 10.0 Hz
    for i in range(20):
        low_freq = i * 0.5
        high_freq = (i + 1) * 0.5
        band_power = np.sum(Pxx[(f >= low_freq) & (f < high_freq)]) # Calculates the total power of every frequency chunk (e.g., 0.5 - 1.0 Hz)
        spectral_features.append(band_power)
        
    # Combines the 2 morphological features and the 20 spectral features
    features = [avg_sut, avg_dt] + spectral_features
    return features


def start_feature_extraction(ppg_signals, abp_signals, sampling_frequency):
    """
    Starts the feature extraction process:
        1. Segments PPG and corresponding ABP signals into 5-second windows
        2. Derives systolic and diastolic blood pressure values from ABP windows
        3. Applies an SQA method based on Leitner et. al (2022) to remove motion artifacts from the PPG signals
        4. Extracts features from PPG window and maps them to corresponding blood pressure values
    
    Args:
        ppg_signals: PPG signals that will be segmented into windows and used for feature extraction
        abp_signals: ABP signals that will be segmented into windows and used for blood pressure extraction
        sampling_frequency: The sampling frequency of the data record (125 Hz for MIMIC data)

    Returns:
        X_train: All 22 extracted features for each PPG window
        Y_train: All corresponding blood pressure values for each ABP window
    """

    X_train = []
    Y_train = []

    WINDOW_SIZE = sampling_frequency * 5 # 125 * 5 = 625
    for i in range(0, len(ppg_signals) - WINDOW_SIZE, WINDOW_SIZE):

        # 1. Segmentation
        ppg_window = ppg_signals[i : i+WINDOW_SIZE]
        abp_window = abp_signals[i : i+WINDOW_SIZE]

        # 2. Derivation of blood pressure
        systolic_bp = np.max(abp_window)
        diastolic_bp = np.min(abp_window)

        # 3. SQA method
        if sqa_autocorrelation(ppg_window, sampling_frequency):
            # 4. Actual feature and corresponding blood pressure extraction
            features = extract_features(ppg_window, sampling_frequency)
            X_train.append(features)
            Y_train.append([systolic_bp, diastolic_bp])
    
    # Convert to numpy arrays for model training
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train