from scipy.signal import butter, filtfilt, correlate
import numpy as np

def butterworth_lowpass_filter(ppg_signals, cutoff, sampling_frequency, order):
    """
    Applies a Butterworth lowpass filter:
        - All frequencies above the cutoff are removed
        - All frequencies below the cutoff are kept

    Args:
        ppg_signals: The PPG signals that need to be filtered
        cutoff: The specified cutoff frequency
        sampling_frequency: The sampling frequency of the data record (125 Hz for MIMIC data)
        order: The order of the Butterworth low-pass filter

    Returns:
        lowpass_filtered_ppg_signals: The filtered PPG signals
    """
    nyquist = 0.5 * sampling_frequency
    cutoff_frequency = cutoff / nyquist
    
    b, a = butter(order, cutoff_frequency, btype='lowpass', analog=False)
    
    lowpass_filtered_ppg_signals = filtfilt(b, a, ppg_signals) # Ensures that PPG and ABP signals stay aligned
    return lowpass_filtered_ppg_signals

    
def sqa_autocorrelation(ppg_window, sampling_frequency):
    """
    Implements the Autocorrelation-based SQA method based on Leitner et al. (2022)
    
    Logic:
    - Clean PPG signals are highly periodic.
    - We calculate the autocorrelation function (ACF).
    - If the maximum peak of the ACF (at the heart rate lag) is < 0.7, 
      the signal is considered 'corrupted' by motion artifacts.
      
    Args:
        window_data: 5-second PPG segment
        sampling_frequency: Sampling frequency (default 125Hz)
    
    Returns:
        True if valid (Max Autocorrelation >= 0.7), False otherwise.
    """
    # 1. Normalize the window for standardization
    normalized_window = ppg_window - np.mean(ppg_window)
    
    # 2. Calculate Autocorrelation
    corr = correlate(normalized_window, normalized_window, mode='full')
    corr = corr[len(corr)//2:] # Only positive lags
    
    # 3. Normalize Autocorrelation
    if corr[0] == 0:
        return False # No signal
    corr_norm = corr / corr[0] # Normalization to 1.0
    
    # 4. Find the peaks of the autocorrelated signal
    min_lag = int(sampling_frequency * 60 / 220) # (220 BPM) = 125 * (60/220) ~= 34 samples
    max_lag = int(sampling_frequency * 60 / 30) # (30 BPM)  = 125 * (60/30) = 250 samples
    
    # Safety check: ensure window is long enough
    if len(corr_norm) < max_lag:
        max_lag = len(corr_norm)
        
    valid_region = corr_norm[min_lag : max_lag]
    
    if len(valid_region) == 0:
        return False

    max_autocorr = np.max(valid_region)
    
    # 5. Threshold Check (Leitner et al. used 0.7)
    if max_autocorr >= 0.7:
        return True # Clean Signal
    else:
        return False # Noisy / Corrupted