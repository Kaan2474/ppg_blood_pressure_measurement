from data_preparation import load_data, extract_signals, preprocess_ppg_signals
from motion_artifact_removal import butterworth_lowpass_filter
from feature_extraction import start_feature_extraction
from ann_model import scale_data, split_data, create_ann_model, evaluate_ann_model, save_model
from visualization import create_scatter_plot, create_histogram, visualize_signals


# --- 1. Dataset preparation ---
RECORD_041 = '/Users/kaanhisi/Desktop/Projekte/PPG_based_BPM/data/041/041'
signals, signal_names, sampling_frequency = load_data(RECORD_041)
ppg_signals, abp_signals = extract_signals(signal_names, signals)
valid_ppg_signals = preprocess_ppg_signals(ppg_signals)

# --- 2. Motion artifact removal using low-pass filter ---
lowpass_filtered_ppg_signals = butterworth_lowpass_filter(valid_ppg_signals, 12, sampling_frequency, 4)

# --- 3. Feature extraction ---
X_train, Y_train = start_feature_extraction(lowpass_filtered_ppg_signals, abp_signals, sampling_frequency)

# --- 4. Development of ANN model ---
scaler, X_scaled = scale_data(X_train)
X_train, X_test, Y_train, Y_test = split_data(X_scaled, Y_train)
ann_model = create_ann_model()
ann_model.fit(X_train, Y_train) # Initiate learning process
save_model(ann_model, scaler)

# --- 5. Evaluation of ANN model ---
Y_pred, mae_sbp, mae_dbp, sd_sbp, sd_dbp = evaluate_ann_model(ann_model, X_test, Y_test)

# --- 6. Visualization ---
visualize_signals(ppg_signals, abp_signals, num_samples=1000)
create_scatter_plot(Y_test, Y_pred, mae_sbp, sd_sbp, mae_dbp, sd_dbp)
create_histogram(Y_pred, Y_test)