import wfdb
import numpy as np


def load_data(path):
    """
    Loads MIMIC data record and additional information

    Args:
        path: Location of the data

    Returns:
        signals: All available signals (e.g., PPG, ABP, ...)
        signal_names: The header information of the signals (e.g., PLETH, ABP)
        sampling_frequency: The sampling frequency of the data record (125 Hz for MIMIC data)
    """
    record = wfdb.rdrecord(path)
    signals = record.p_signal
    signal_names = record.sig_name
    sampling_frequency = record.fs
    return signals, signal_names, sampling_frequency


def extract_signals(signal_names, signals):
    """
    Extracts PPG and ABP signals from the data based on the determined indices

    Args:
        signal_names: The header information of the signals (e.g., PLETH, ABP)
        signals: All available signals (e.g., PPG, ABP, ...)

    Returns:
        ppg_signals: Raw PPG signals
        abp_signals: Raw ABP signals
    """
    ppg_index = signal_names.index('PLETH')
    ppg_signals = signals[:, ppg_index]
    abp_index = signal_names.index('ABP')
    abp_signals = signals[:, abp_index]
    return ppg_signals, abp_signals


def preprocess_ppg_signals(ppg_signals):
    """
    Replaces invalid entries (NaNs) with 0

    Args:
        ppg_signals: Raw PPG signals

    Returns:
        valid_ppg_signals: PPG signals with exclusively valid entries
    """
    valid_ppg_signals = np.nan_to_num(ppg_signals)
    return valid_ppg_signals