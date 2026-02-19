import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

INPUT_FILE = "signals_walking_2.csv"
OUTPUT_FILE = "aligned_signals_walking_2.csv"


def distribute_timestamps(signal, sampling_rate):
    """
    Creates unique timestamps for every PPG and ACC signal

    Args:
        signal: PPG or ACC signal
        sampling_rate: Sampling rate of the signal (PPG: 176.0 / ACC: 52.0)
    
    Returns:
        signal: PPG or ACC signal with unique timestamps
    """
    signal = signal.copy().reset_index(drop=True)
    
    time_step_ns = (1.0 / sampling_rate) * 1e9 # nanoseconds between consecutive samples
    
    # Allocates same timestamps into the same group
    signal['group_id'] = (signal['Polar Timestamp'] != signal['Polar Timestamp'].shift()).cumsum()
    
    # Calculate offset from the end of the packet
    packet_sizes = signal.groupby('group_id')['Polar Timestamp'].transform('count')
    sample_indices = signal.groupby('group_id').cumcount()
    samples_from_end = (packet_sizes - 1) - sample_indices
    
    # Apply correction
    signal['Corrected_Time_ns'] = signal['Polar Timestamp'] - (samples_from_end * time_step_ns)
    return signal

# Loads the PPG and ACC data from the CSV file into a dataframe
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# Separates PPG and ACC signals
raw_ppg_signals = df[df['Type'] == 'PPG'].copy()
raw_acc_signals = df[df['Type'] == 'ACC'].copy()

# Fixes timestamps
print("Fixing timestamps...")
PPG_SAMPLING_RATE = 176.0
ACC_SAMPLING_RATE = 52.0
ppg_fixed_timestamps = distribute_timestamps(raw_ppg_signals, PPG_SAMPLING_RATE)
acc_fixed_timestamps = distribute_timestamps(raw_acc_signals, ACC_SAMPLING_RATE)

# Gets the start time
start_ns = min(ppg_fixed_timestamps['Corrected_Time_ns'].iloc[0], acc_fixed_timestamps['Corrected_Time_ns'].iloc[0])

# Converts the start time from nanoseconds into seconds
ppg_time_sec = (ppg_fixed_timestamps['Corrected_Time_ns'] - start_ns) / 1e9
acc_time_sec = (acc_fixed_timestamps['Corrected_Time_ns'] - start_ns) / 1e9

# 4. Linear interpolation to align PPG and ACC signals
print("Aligning data...")

target_time = ppg_time_sec.values

# Creates interpolators for ACC X, Y, Z
interp_x = interp1d(acc_time_sec, acc_fixed_timestamps['Val0'], kind='linear', fill_value="extrapolate")
interp_y = interp1d(acc_time_sec, acc_fixed_timestamps['Val1'], kind='linear', fill_value="extrapolate")
interp_z = interp1d(acc_time_sec, acc_fixed_timestamps['Val2'], kind='linear', fill_value="extrapolate")

# Generates aligned ACC data
aligned_acc_x = interp_x(target_time)
aligned_acc_y = interp_y(target_time)
aligned_acc_z = interp_z(target_time)

# Creates final dataframe for CSV file
final_df = pd.DataFrame({
    'Time_sec': target_time,
    # PPG Channels (Raw Optical Data)
    'PPG_0': ppg_fixed_timestamps['Val0'].values,
    'PPG_1': ppg_fixed_timestamps['Val1'].values,
    'PPG_2': ppg_fixed_timestamps['Val2'].values,
    'Ambient': ppg_fixed_timestamps['Val3'].values,
    # Synchronized Accelerometer Data
    'ACC_X': aligned_acc_x,
    'ACC_Y': aligned_acc_y,
    'ACC_Z': aligned_acc_z
})

# 6. Save aligned data to CSV file
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Full aligned data saved to {OUTPUT_FILE}")
print(final_df.head())

# 7. Visualization: Plot PPG0 vs Ambient
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(final_df['Time_sec'], final_df['PPG_0'], color='green', label='PPG_0')
plt.title("PPG Channel 0")
plt.legend()

plt.subplot(2,1,2)
plt.plot(final_df['Time_sec'], final_df['ACC_Z'], color='orange', label='Acc Z (Aligned)')
plt.title("Aligned Accelerometer Z")
plt.legend()

plt.tight_layout()
plt.show()