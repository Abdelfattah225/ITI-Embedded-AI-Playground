# YOLOv8 Training and Inference Exercise

This directory contains scripts and exercises demonstrating how to train, evaluate, and run inference using the **Ultralytics YOLOv8** object detection model. 

The code is split into two exercises: a baseline "Session" using a general dataset (COCO128), and a "Lab" applying the same workflow to a custom dataset (Construction PPE).

## 📝 What We Did (Step-by-Step)

### Step 1: Environment Setup
We started by installing the Ultralytics library, which provides the YOLOv8 architecture and training pipelines.
```python
!pip install ultralytics
from ultralytics import YOLO
```

### Step 2: Initializing the Base Model
We loaded the **YOLOv8 Nano** (`yolov8n.pt`) model. We chose the Nano version because it is the smallest and most lightweight model, allowing for fast training and testing without requiring massive computational power.

### Step 3: Training (Fine-Tuning)
We trained the model on specific datasets using the `.train()` method. 
* **In the Session:** We trained on `coco128.yaml` (general everyday objects).
* **In the Lab:** We trained on `construction-ppe.yaml` (hard hats, safety vests, etc.).

We configured the training with the following parameters:
* `epochs=10`: The model iterated through the dataset 10 times.
* `imgsz=640`: All images were resized to 640x640 for consistency.
* `batch=16`: Processed 16 images at a time.
* `patience=5` *(Session only)*: An early-stopping mechanism to halt training if no improvement was seen after 5 epochs.

### Step 4: Model Validation
During training, YOLO automatically saves the best-performing model weights to a `runs/` directory. We loaded these specific weights (`best.pt`) and ran the `.val()` method to evaluate our newly trained model's accuracy on a validation set.

### Step 5: Inference (Making Predictions)
We took our trained model and tested it on a brand-new image from the internet using the `.predict()` method. 
```python
pred = best.predict(source='...', save=True, save_txt=True, conf=0.5)
```
* `save=True`: Generated a new image with bounding boxes drawn over detected objects.
* `save_txt=True`: Saved the exact coordinates of those bounding boxes to a text file.
* `conf=0.5`: Ignored any predictions the model was less than 50% confident about.

### Step 6: Visualizing the Output
Finally, we displayed the resulting image. We explored two different ways to do this:
1. **The Manual Way (Session):** We used Python's `pathlib` to locate the saved image, `cv2` (OpenCV) to read and convert the image colors, and `matplotlib` to plot it on the screen.
2. **The Built-in Way (Lab):** We used Ultralytics' native `pred[0].show()` method, which handles the visualization in a single line of code.

---

## 📂 Expected Output Directory
Running these scripts will automatically generate a `runs/` folder in this directory. 
* `runs/detect/train.../`: Contains your model weights (`best.pt`), confusion matrices, and training graphs.
* `runs/detect/predict.../`: Contains your output images with bounding boxes drawn on them.
