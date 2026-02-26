# PPG-Based Blood Pressure Measurements With Polar Verity Sense

This repository provides a Python-based prototype for non-invasive and cuffless blood pressure estimation using photoplethysmography (PPG) signals. The system supports real-time signal acquisition from a Polar Verity Sense device and uses an Artificial Neural Network (ANN) model trained on the MIMIC database. The project includes methods for motion artifact removal, making it suitable for measurements during physical activities like walking. The development of the prototype is based on Wang et al. (2018) and had seven key phases: 

<img width="3924" height="564" alt="image" src="https://github.com/user-attachments/assets/7d46a451-9e38-4c8b-8650-8dfe2f23d06f" />

## Features
- **Real-time Signal Acquisition**: Simultaneous streaming of PPG and Accelerometer (ACC) signals via Bluetooth Low Energy (BLE)
- **Signal Alignment**: Synchronization of different sampling rates (PPG at 176 Hz, ACC at 52 Hz) using linear interpolation
- **Motion Artifact Mitigation**:
  1. 4th-order Butterworth low-pass filter
  2. Signal quality assessment using an autocorrelation-based filtering method based on Leitner et al. (2022)
- **Feature Engineering**: Extraction of 22 distinct features from 5-second PPG windows
- **Machine Learning**: Multi-layer Perceptron (MLP) regressor trained on the MIMIC database
- **Blood Pressure Measurement**: Predicting Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) from PPG signals
- **Performance Analytics**: Evaluation by calculating Mean Absolute Error (MAE) and Standard Deviation (SD) according to the AAMI standard

## Project Structure

File | Description
--- | ---
main.py | Primary entry point for data preparation, feature extraction, and development of the ANN model using MIMIC data
data_preparation.py | Loads, extracts, and preprocesses PPG and arterial blood pressure signals from the MIMIC data records
motion_artifact_removal.py | Contains the Butterworth low-pass filter and the autocorrelation-based filtering
feature_extraction.py | Extracts 2 morphological (systolic upstroke time, diastolic time) and 20 spectral features (power bands)
ann_model.py | Creates and defines the MLP architecture
signal_acquisition.py | Connects to the Polar Verity Sense with BLE and handles data streaming of PPG and ACC signals
aligning_signals.py | Aligns PPG and ACC signals through linear interpolation and assigns a unique timestamp for each alignment
measure_bp.py | Predicts SBP and DBP by feeding PPG signals obtained from the Polar Verity Sense into the trained ANN model
bp_evaluation.py | Compares blood pressure predictions against reference values from a cuff
visualization.py | Creates scatter plots, histograms, and signal visualizations

## Results
Metric | ANN model without motion artifact removal | ANN model with motion artifact removal
--- | --- | ---
MAE of SBP | 4.44 mmHg | 3.57 mmHg
SD of SBP | 6.52 mmHg | 4.42 mmHg
MAE of DBP | 2.92 mmHg | 2.41 mmHg
SD of DBP | 4.75 mmHg | 3.09 mmHg

Metric | Prototype without motion artifact removal | Prototype with motion artifact removal
--- | --- | ---
MAE of SBP | 53.58 mmHg | 59.86 mmHg
SD of SBP | 3.66 mmHg | 0.40 mmHg
MAE of DBP | 23.62 mmHg | 24.53 mmHg
SD of DBP | 3.27 mmHg | 4.60 mmHg

Possible reasons for the high error rate of the prototype:
- **Insufficient data**: Only data record 041 was used to train the ANN model
- **Inappropriate database**: MIMIC data was recorded at the fingertip from intensive care unit patients in static positions, whereas Polar Verity Sense data was collected from the upper arm while walking

## Installation & Usage

**Prerequisites**
- Python version 3.11.10
- Polar Verity Sense (for data streaming)
- Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, bleak, bleakheart, joblib, wfdb

**Installation**
```console
pip install numpy pandas scipy scikit-learn matplotlib bleak "bleakheart" joblib wfdb
```

**Quick Start**
1. **Load Data**: Download data records from the MIMIC database
2. **Store Data**: Store the data records in a separate folder within the Blood Pressure Measurement folder
3. **Data Path**: Add the directory path that leads to the MIMIC data within src/main.py (line 9)
4. **Model Training**: Run src/main.py to train the ANN model or skip this step and use the already existing models/ann_bp_model.pkl
   - Note: Step 4 shouldn't be skipped because the prototype was only trained on data record 041
   - Download as many data records as possible from the MIMIC database
6. **Data Collection**: Run src/signal_acquisition.py while wearing the turned on Polar Verity Sense on the arm to record PPG/ACC data to a CSV
7. **Signal Alignment**: Run src/aligning_signals.py to synchronize PPG and ACC signals
8. **Blood Pressure Estimation**: Run src/measure_bp.py to predict SBP and DBP from the collected PPG signals
9. **Reference Values**: Measure reference blood pressure values with a cuff and add these values within src/bp_evaluation.py (line 41 and 42)
10. **Evaluation**: Run src/bp_evaluation.py to compare your blood pressure predictions against the reference values

## Future Work
- **Database Expansion**: Incorporating all data records from the MIMIC database to improve generalization and reduce overfitting
- **Selection of the database**: Testing an alternative database capturing upper-arm data during movement-intensive scenarios
- **Motion artifact removal**:
  - Using the acquired ACC signals to further reduce the impact of motion-induced noise
  - Identifying and applying more methods for motion artifact removal to improve measurement accuracy
- **Sensor Diversity**: Testing the prototype with other wearable sensors beyond the Polar Verity Sense 

## References
- MIMIC database: https://archive.physionet.org/physiobank/database/mimicdb/
- Wang, L., Zhou, W., Xing, Y., & Zhou, X. (2018). A novel neural network model for blood pressure estimation using PPG. Journal of Healthcare Engineering. https://doi.org/10.1155/2018/7804243
- Leitner, J., Chiang, P. H., & Dey, S. (2022). Personalized blood pressure estimation using photoplethysmography: A transfer learning approach. IEEE Journal of Biomedical and Health Informatics, 26(1), 218–228. https://doi.org/10.1109/JBHI.2021.3085526  

**Disclaimer**: This software is for research purposes only and is not a certified medical device
