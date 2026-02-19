from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

def scale_data(X_train):
    """
    Standardizes the differences of the units in morphological (seconds) and spectral features (units of power)

    Args:
        X_train: The morphological and spectral features that need to standardized

    Returns:
        scaler: The object that standardizes the morphological and spectral features
        X_scaled: The scaled feature set
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    return scaler, X_scaled


def split_data(X_scaled, Y_train):
    """
    Split data for training (85%) and testing (15%)

    Args:
        X_scaled: The scaled feature set
        Y_train: The corresponding blood pressure values

    Returns:
        X_train: Feature set for training the ML model
        X_test: Feature set for testing the ML model
        Y_train: Corresponding blood pressure values for training the ML model
        Y_test: Corresponding blood pressure values for testing the ML model
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_train, test_size=0.15, random_state=42
        )
    return X_train, X_test, Y_train, Y_test


def create_ann_model():
    """
    Creates the Feed-Forward ANN model (Multilayer Perceptron) based on Wang et al. (2018)

    Returns:
        model: The ANN model
    """
    model = MLPRegressor( # Multilayer Perceptron
    hidden_layer_sizes=(10,),
    activation='tanh',
    solver='lbfgs', # solver='lbfgs' is similar to Levenberg-Marquardt
    max_iter=2000,
    random_state=42
    )
    return model


def save_model(model, scaler):
    """
    Saves the ANN model and the scaler object

    Args:
        model: The ANN model
        scaler: The object that standardizes the morphological and spectral features
    """
    joblib.dump(model, 'ann_bp_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')


def evaluate_ann_model(model, X_test, Y_test):
    """
    Evaluates the ANN model according to the AAMI standard:
        - Calculates mean absolute error (MAE)
        - Calculates standard deviation (SD)

    Args:
        model: The ANN model
        X_test: Features for testing
        Y_test: Blood pressure values for testing

    Returns:
        Y_pred: The predicted blood pressure values
        mae_sbp: The mean absolute error of systolic blood pressure
        mae_dbp: The mean absolute error of diastolic blood pressure
        sd_sbp: The standard deviation of systolic blood pressure
        sd_dbp: The standard deviation of diastolic blood pressure
    """
    Y_pred = model.predict(X_test)
    # Calculate MAE
    mae_sbp = mean_absolute_error(Y_test[:, 0], Y_pred[:, 0])
    mae_dbp = mean_absolute_error(Y_test[:, 1], Y_pred[:, 1])
    # Calculate SD
    sd_sbp = np.std(np.abs(Y_test[:, 0] - Y_pred[:, 0]))
    sd_dbp = np.std(np.abs(Y_test[:, 1] - Y_pred[:, 1]))
    print(f"Systolic BP: MAE = {mae_sbp:.2f} ± {sd_sbp:.2f} mmHg")
    print(f"Diastolic BP: MAE = {mae_dbp:.2f} ± {sd_dbp:.2f} mmHg")
    return Y_pred, mae_sbp, mae_dbp, sd_sbp, sd_dbp