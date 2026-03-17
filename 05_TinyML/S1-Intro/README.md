# TinyML Session - Introduction to TinyML Concepts

## Session Overview
This session introduces the fundamentals of TinyML (Tiny Machine Learning) — 
the practice of running machine learning models on resource-constrained 
embedded devices such as microcontrollers.

## What is TinyML?
TinyML refers to deploying machine learning models on devices with:
- **Flash Memory**: 256KB - 2MB (where the model is stored)
- **RAM**: 64KB - 512KB (where computations happen)
- **Power**: Battery or energy harvesting
- **No internet**: Models run locally on the device

## Session Examples

### Example 1: Human Activity Recognition (HAR)
**Goal**: Classify human activities (walking, sitting, standing, etc.) 
using smartphone sensor data — a classic TinyML use case.

**Pipeline Demonstrated**:
1. Download UCI HAR Dataset (accelerometer/gyroscope readings)
2. Preprocess data using StandardScaler
3. Reduce features from 561 → 20 (for embedded constraints)
4. Build a small neural network (Dense 64 → Dense 32 → Dense 6)
5. Train and evaluate the model
6. Convert to TensorFlow Lite format (.tflite)
7. Apply quantization to shrink the model
8. Compare model sizes (Float32 vs Quantized)
9. Run inference using TFLite Interpreter

**Key Concepts Covered**:
| Concept | What It Does |
|---|---|
| StandardScaler | Normalizes features to mean=0, std=1 |
| Feature Reduction | Reduces input size for smaller models |
| TFLite Conversion | Converts Keras model to embedded-friendly format |
| Quantization | Shrinks weights from float32 (4 bytes) to int8 (1 byte) |
| TFLite Interpreter | Simulates running the model on an embedded device |

### Example 2: YOLOv8 Nano Object Detection
**Goal**: Demonstrate that even computer vision models can be made 
tiny for edge deployment.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # nano = smallest version
results = model("bus.jpg")
results[0].show()
```

**Key Takeaway**: YOLOv8-nano (3.2M parameters) is designed for 
edge devices like phones, drones, and cameras — same TinyML philosophy 
of making models small enough to deploy on constrained hardware.

## Core TinyML Pipeline
```
Raw Data → Preprocessing → Feature Selection → Small Model → Train
    → Convert to TFLite → Quantize → Evaluate Size/Accuracy → Deploy
```

## Key Takeaways
1. **Smaller is better** — Embedded devices have severe memory limits
2. **Feature selection** — Fewer inputs = smaller model = faster inference
3. **Quantization** — Reduces model size ~4x with minimal accuracy loss
4. **TensorFlow Lite** — Bridge between training (PC) and deployment (device)
5. **Trade-offs** — Always balance accuracy vs size vs speed vs power

## Tools Used
- Python 3.x
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Ultralytics (YOLOv8)

