import matplotlib.pyplot as plt

def create_scatter_plot(Y_test, Y_pred, mae_sbp, sd_sbp, mae_dbp, sd_dbp):
    plt.figure(figsize=(14, 6))
    # Plot SBP Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test[:, 0], Y_pred[:, 0], color='blue', alpha=0.6)
    plt.plot([min(Y_test[:,0]), max(Y_test[:,0])], [min(Y_test[:,0]), max(Y_test[:,0])], 'r--')
    plt.title(f"Systolic blood pressure (MAE ± SD: {mae_sbp:.2f} ± {sd_sbp:.2f} mmHg)")
    plt.xlabel("Measured systolic blood pressure")
    plt.ylabel("Estimated systolic blood pressure")
    plt.grid(True)

    # Plot DBP Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test[:, 1], Y_pred[:, 1], color='green', alpha=0.6)
    plt.plot([min(Y_test[:,1]), max(Y_test[:,1])], [min(Y_test[:,1]), max(Y_test[:,1])], 'r--')
    plt.title(f"Diastolic blood pressure (MAE ± SD: {mae_dbp:.2f} ± {sd_dbp:.2f} mmHg)")
    plt.xlabel("Measured diastolic blood pressure")
    plt.ylabel("Estimated diastolic blood pressure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_histogram(Y_pred, Y_test):
    # B. Histograms of error (Replicating Fig. 7 from Wang et al. (2018))
    # Calculate the error: Estimated - Measured
    sbp_error = Y_pred[:, 0] - Y_test[:, 0]
    dbp_error = Y_pred[:, 1] - Y_test[:, 1]
    plt.figure(figsize=(14, 6))

    # SBP Error Histogram
    plt.subplot(1, 2, 1)
    plt.hist(sbp_error, bins=30, color='maroon', edgecolor='black', alpha=0.7)
    plt.title("Estimation error distribution for systolic blood pressure")
    plt.xlabel("Error in mmHg")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y', alpha=0.5)

    # DBP Error Histogram
    plt.subplot(1, 2, 2)
    plt.hist(dbp_error, bins=30, color='maroon', edgecolor='black', alpha=0.7)
    plt.title("Estimation error distribution for diastolic blood pressure")
    plt.xlabel("Error in mmHg")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.show()


def visualize_signals(ppg_signals, abp_signals, num_samples=1000):
    """
    Visualizes the aligned PPG and ABP signals.
    - Top: PPG (Black)
    - Bottom: ABP (Red)
    """
    plt.figure(figsize=(12, 8))

    # Top Plot: PPG (Input)
    plt.subplot(2, 1, 1)
    plt.plot(ppg_signals[:num_samples], color='black', label='PPG Signal')
    plt.title("Photoplethysmogram (Input)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Bottom Plot: ABP (Target)
    plt.subplot(2, 1, 2)
    plt.plot(abp_signals[:num_samples], color='red', label='ABP Signal')
    plt.title("Arterial Blood Pressure (Target)")
    plt.xlabel("Samples")
    plt.ylabel("Pressure (mmHg)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()