# PPG-Based Blood Pressure Measurements With Polar Verity Sense

This repository provides a Python-based prototype for non-invasive blood pressure estimation using photoplethysmography (PPG) signals. The system supports real-time signal acquisition from a Polar Verity Sense device and uses an Artificial Neural Network (ANN) model trained on the MIMIC database for blood pressure prediction. The project includes methods for motion artifact removal, making it suitable for measurements during physical activities like walking. The development of the prototype had 8 key phases: 

<img width="4484" height="564" alt="image" src="https://github.com/user-attachments/assets/e060ed3c-2e4f-4287-abc8-0e315d452151" />

## Features
- **Real-time Acquisition**: Simultaneous streaming of PPG and Accelerometer (ACC) signals via Bluetooth Low Energy (BLE)
- **Signal Alignment**: Synchronization of differing sampling rates (PPG at 176 Hz, ACC at 52 Hz) using linear interpolation and assignment of unique timestamps
- **Motion Artifact Mitigation**:
  1. 4th-order Butterworth low-pass filter
  2. Signal quality assessment using an autocorrelation-based filtering method
- **Feature Engineering**: Extraction of 22 distinct features from 5-second PPG windows
- **Machine Learning**: Multi-layer Perceptron (MLP) regressor trained on the MIMIC database
- **Performance Analytics**: Evaluation by calculating Mean Absolute Error (MAE) and Standard Deviation (SD) according to the AAMI standard

## Project Structure

File | Description
--- | ---
main.py | Primary entry point for data preparation, feature extraction, and development of the ANN model using MIMIC data
data_preparation.py | Loads, extracts, and preprocesses PPG and arterial blood pressure signals from the MIMIC data records
motion_artifact_removal.py | Contains the Butterworth low-pass filter and the autocorrelation-based filtering
feature_extraction.py | Extracts 2 morphological (Systolic Upstroke Time, Diastolic Time) and 20 spectral features (power bands)
ann_model.py | Creates and defines the MLP architecture
signal_acquisition.py | Connects to the Polar Verity Sense with BLE and handles data streaming of PPG and ACC signals
aligning_signals.py | Aligns PPG and ACC signals through linear interpolation and assigns a unique timestamp for each alignment
measure_bp.py | Predicts systolic and diastolic blood pressure values from aligned Polar Verity Sense data using the trained ANN model
bp_evaluation.py | Compares model predictions against reference blood pressure readings from a cuff
visualization.py | Creates scatter plots, histograms, and signal visualizations

## Results

## Installation & Usage

**Prerequisites**
- Python 3.8+
- Polar Verity Sense (for data streaming)
- Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, bleak, bleakheart, joblib, wfdb

**Installation**
```console
pip install numpy pandas scipy scikit-learn matplotlib bleak joblib wfdb
```
```console
python3 -m pip install "bleakheart"
```

**Quick Start**
1. **Load Data**: Download data records from the MIMIC database and store the data in a separate directory
2. **Data Path**: Add the path to the MIMIC data directory within src/main.py (line 9)
3. **Model Training**: Run src/main.py to train the ANN model or skip this step and use the already existing models/ann_bp_model.pkl 
4. **Data Collection**: Run src/signal_acquisition.py while wearing the turned on Polar Verity Sense on the arm to record PPG/ACC data to a CSV
5. **Signal Alignment**: Run src/aligning_signals.py to synchronize PPG and ACC signals
6. **Blood Pressure Estimation**: Run src/measure_bp.py to predict systolic and diastolic blood pressure from the collected data
7. **Reference Values**: Measure reference blood pressure values using a cuff and add these values within src/bp_evaluation.py (line 43 and 44)
8. **Evaluation**: Run src/bp_evaluation.py to compare your results against the reference values


**Disclaimer**: This software is for research purposes only and is not a certified medical device.
