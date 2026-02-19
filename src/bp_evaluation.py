import pandas as pd
import numpy as np

def evaluate_single_session(csv_file, cuff_sbp, cuff_dbp, skip_seconds=0):
    """
    Calculates the error for a single recording session.
    """
    # 1. Load the data
    df = pd.read_csv(csv_file)
    
    # 2. Filter out the "Warm-up" period (first 10 seconds)
    # The first few seconds often contain unstable predictions as the filter settles
    df_stable = df[df['Time in sec'] > skip_seconds]
    
    if len(df_stable) == 0:
        print("Error: File is too short or skip_seconds is too large.")
        return None

    # 3. Calculate Model's Average Prediction for this session
    pred_sbp = df_stable['SBP'].mean()
    pred_dbp = df_stable['DBP'].mean()
    
    # 4. Calculate Errors (Prediction - Reference)
    error_sbp = pred_sbp - cuff_sbp
    error_dbp = pred_dbp - cuff_dbp
    
    print(f"--- Session: {csv_file} ---")
    print(f"Reference Cuff:  {cuff_sbp} / {cuff_dbp} mmHg")
    print(f"Model Predicted: {pred_sbp:.2f} / {pred_dbp:.2f} mmHg")
    print(f"Error:           {error_sbp:+.2f} / {error_dbp:+.2f} mmHg")
    
    return error_sbp, error_dbp

# --- MAIN EVALUATION ---

# 2. Run evaluation for the current file
# We store the errors in a list. If you have multiple files, add them here.
sbp_errors = []
dbp_errors = []

# Analyze the file you uploaded
file_1 = "blood_pressure_results_walking_sqa_1.csv"
err_s, err_d = evaluate_single_session(file_1, 127, 67)
sbp_errors.append(err_s)
dbp_errors.append(err_d)

file_2 = "blood_pressure_results_walking_sqa_2.csv"
err_s, err_d = evaluate_single_session(file_2, 128, 60)
sbp_errors.append(err_s)
dbp_errors.append(err_d)

# 3. Calculate Final MAE and SD
mae_sbp = np.mean(np.abs(sbp_errors))
sd_sbp = np.std(sbp_errors, ddof=1) if len(sbp_errors) > 1 else 0.0

mae_dbp = np.mean(np.abs(dbp_errors))
sd_dbp = np.std(dbp_errors, ddof=1) if len(dbp_errors) > 1 else 0.0

print("\n=== FINAL ACCURACY RESULTS ===")
print(f"Systolic:  MAE = {mae_sbp:.2f} mmHg, SD = {sd_sbp:.2f} mmHg")
print(f"Diastolic: MAE = {mae_dbp:.2f} mmHg, SD = {sd_dbp:.2f} mmHg")