
# TinyML Lab — Human Activity Recognition

## Overview
A TinyML project that classifies human activities using smartphone sensor data.
The model is trained, quantized to INT8, and prepared for deployment on an ESP32 microcontroller.

## Dataset
- **Name:** Human Activity Recognition Using Smartphones (UCI HAR)
- **Source:** UCI Machine Learning Repository
- **Participants:** 30 people
- **Sensors:** 3-axis accelerometer + 3-axis gyroscope
- **Sampling Rate:** 50 Hz
- **Features:** 561 per sample
- **Classes:** 6 activities
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying

## Project Structure

```
├── README.md
├── activity_labels.txt          # Maps label IDs to activity names
├── features.txt                 # Names of the 561 features
├── final_X_train.txt            # Training features
├── final_X_test.txt             # Testing features
├── final_y_train.txt            # Training labels
├── final_y_test.txt             # Testing labels
│
└── sample_data/
    ├── best_model.keras         # Best model checkpoint
    ├── float_model.keras        # Trained float32 model
    ├── model_quantized.tflite   # INT8 quantized model
    ├── model_data.h             # C header file for ESP32
    └── preprocessing.json       # Scaler params + label mapping
```

## Results

| Metric              | Float Model | Quantized Model |
|---------------------|-------------|-----------------|
| Accuracy            | 81.43%      | 81.23%          |
| Size                | 973 KB      | 86 KB           |
| Compression Ratio   | —           | 11.3x           |
| Accuracy Drop       | —           | 0.20%           |

## Pipeline

1. **Data Loading** — Load and verify train/test datasets
2. **Data Inspection** — Check ranges, types, duplicates
3. **Preprocessing** — Remove duplicates, encode labels (0–5), apply StandardScaler
4. **Model Design** — Dense(128) → Dropout → Dense(64) → Dropout → Dense(6)
5. **Training** — Adam optimizer, early stopping, model checkpointing
6. **Evaluation** — Confusion matrix, classification report
7. **Quantization** — Full INT8 quantization using TFLite
8. **Quantized Evaluation** — Verify accuracy after quantization
9. **ESP32 Preparation** — Convert model to .h file, save preprocessing metadata
10. **Deployment Logic** — Sensor → Preprocess → Inference → Activity Label

## Deployment on ESP32

**Files needed:**
- `model_data.h` — Model as C byte array
- `preprocessing.json` — Scaler mean/std and label mapping

**Steps:**
1. Read accelerometer and gyroscope data (50 Hz, 2.56s window)
2. Extract 561 features
3. Scale features using saved mean and std
4. Quantize input to INT8
5. Run inference using TFLite Micro
6. Map predicted index to activity label

## Tools Used
- Python 3.12
- TensorFlow / Keras
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
```