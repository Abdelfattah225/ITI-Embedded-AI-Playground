
# Lab: TinyML Gas Sensor Classification

## Lab Title
Designing and Optimizing a TinyML Model Using Real Sensor Data

## Objective
Build a lightweight neural network to classify gas types from sensor 
readings, then optimize it for deployment on resource-constrained 
embedded devices using TensorFlow Lite and quantization.

## Dataset
- **Name**: UCI Gas Sensor Array Drift Dataset
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset)
- **Samples**: ~13,910
- **Features**: 128 (16 sensors × 8 measurements each)
- **Classes**: 6 gas types
  - Ethanol
  - Ethylene
  - Ammonia
  - Acetaldehyde
  - Acetone
  - Toluene


## Pipeline

### Task 1: Dataset Exploration
- Download and extract UCI Gas Sensor dataset (10 batch files)
- Parse custom format: `class;concentration feature_id:value ...`
- Inspect samples, features, and class distribution

### Task 2: Data Preprocessing
- Check for NaN and Inf values
- Handle missing data with mean imputation
- Normalize using StandardScaler (mean=0, std=1)
- Split: 80% train / 20% test (stratified)

### Task 3: Feature Selection
- Applied **SelectKBest with ANOVA F-score**
- Reduced features: **128 → 20** (84.4% reduction)
- Keeps only the most discriminative features for gas classification

### Task 4: Model Design
Architecture designed for TinyML constraints:
```
Input(20) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(6, Softmax)
```
- **Total Parameters**: 710
- **Max 2 hidden layers** (constraint met)
- **Max 16 neurons per layer** (constraint met)

### Task 5: Model Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 50
- Batch Size: 32

### Task 6: TFLite Conversion
- Converted Keras model to TensorFlow Lite format
- Saved as `.tflite` binary file

### Task 7: Quantization
Three model versions created:
| Model | Description |
|---|---|
| Float32 | No optimization — baseline |
| Default Quantized | `tf.lite.Optimize.DEFAULT` — dynamic range |
| Full Int8 | Full integer quantization with representative dataset |

### Task 8: Size Comparison
| Model Type | Size (KB) | Parameters |
|---|---|---|
| Float32 | ~4.84 | 710 |
| Default Quantized | ~4.84 | 710 |
| Full Int8 | ~4.XX | 710 |

> **Note**: At 710 parameters, file metadata overhead dominates.
> Quantization benefits become more visible with larger models (1000+ params).

### Task 9: Inference Comparison
- Ran inference on all model variants using TFLite Interpreter
- Compared prediction accuracy across Float32, Quantized, and Int8
- Verified deployment feasibility (all models < 50KB target)

## Key Results
```
Feature Reduction:  128 → 20 features (84.4% reduction)
Model Size:         ~5 KB (well under 50KB target)
Total Parameters:   710
Deployment Ready:   ✓ All variants fit embedded constraints
```

## Key Findings

### Why Feature Selection Matters
- Reduces model input size → smaller first layer → fewer total parameters
- Removes noisy/redundant sensor readings
- Speeds up inference on microcontrollers

### Why Quantization Matters
- Converts 32-bit floats to 8-bit integers (~4x compression)
- Minimal accuracy degradation for most models
- Essential for models with thousands of parameters
- Less impactful for very tiny models where file overhead dominates

### Trade-offs Observed
| Factor | More Features / Bigger Model | Fewer Features / Smaller Model |
|---|---|---|
| Accuracy | Higher | Slightly Lower |
| Model Size | Larger | Smaller |
| Inference Speed | Slower | Faster |
| Power Usage | More | Less |
| Deployability | Harder | Easier |
