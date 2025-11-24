# 🚀 Kaggle Training Guide for YOLOv8-CBAM

Complete guide to train your enhanced YOLOv8-CBAM model on Kaggle.

---

## 📋 Prerequisites

1. ✅ Commit your enhanced ultralytics code to GitHub
2. ✅ Upload your tomato disease dataset to Kaggle Datasets
3. ✅ Create a new Kaggle Notebook with GPU enabled

---

## 🔧 Step-by-Step Setup

### Step 1: Install Dependencies

```python
# Cell 1: Install packages
!pip install -q ultralytics==8.0.196
!pip install -q opencv-python-headless PyYAML

import os
import shutil
import yaml
from pathlib import Path
import torch

print("✓ Dependencies installed")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Step 2: Clone Your Enhanced Repository

```python
# Cell 2: Clone repository
# Replace with YOUR GitHub repository URL
!git clone https://github.com/YOUR_USERNAME/ultralytics-main.git

os.chdir('ultralytics-main')
print(f"✓ Current directory: {os.getcwd()}")
```

### Step 3: Copy Dataset from Readonly Input

```python
# Cell 3: Copy dataset to working directory
# Kaggle datasets are in readonly /kaggle/input/
# We need to copy to /kaggle/working for training

# UPDATE THIS with your actual Kaggle dataset name!
KAGGLE_INPUT = "/kaggle/input/your-dataset-name"
WORKING_DIR = "/kaggle/working"
DATASET_DIR = os.path.join(WORKING_DIR, "dataset")

print(f"Copying dataset from {KAGGLE_INPUT} to {DATASET_DIR}...")

# Copy the entire dataset
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
shutil.copytree(KAGGLE_INPUT, DATASET_DIR)

print("✓ Dataset copied successfully")
```

### Step 4: Create Data Configuration

```python
# Cell 4: Create data.yaml
data_yaml = {
    "path": DATASET_DIR,
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "nc": 9,
    "names": [
        "Early Blight",
        "Healthy",
        "Late Blight",
        "Leaf Miner",
        "Leaf Mold",
        "Mosaic Virus",
        "Septoria",
        "Spider Mites",
        "Yellow Leaf Curl Virus",
    ],
}

# Save configuration
data_yaml_path = os.path.join(WORKING_DIR, "tomato_data.yaml")
with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✓ Data configuration saved: {data_yaml_path}")
```

### Step 5: Verify Model Configuration

```python
# Cell 5: Check enhanced model
model_config = "ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml"

if os.path.exists(model_config):
    print(f"✓ Enhanced model found: {model_config}")

    # Test load
    from ultralytics import YOLO

    model = YOLO(model_config)

    # Check CBAM modules
    cbam_count = sum(1 for m in model.model.model if "CBAM" in m.__class__.__name__)
    print(f"✓ Found {cbam_count} CBAM modules")

    # Check detection heads
    detect = [m for m in model.model.model if "Detect" in m.__class__.__name__]
    if detect and hasattr(detect[0], "stride"):
        print(f"✓ Detection heads: {len(detect[0].stride)}")
else:
    print("❌ Model config not found!")
```

### Step 6: Train the Model

```python
# Cell 6: Training
from ultralytics import YOLO

# Training configuration
EPOCHS = 200
BATCH_SIZE = 16  # Reduce if out of memory
IMAGE_SIZE = 640
MODEL_SCALE = 's'  # Options: n, s, m, l, x

# Load model
model = YOLO(model_config)

# Start training
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,  # Use GPU
    scale=MODEL_SCALE,

    # Project settings
    project='tomato_disease_detection',
    name='yolov8-cbam',

    # Optimization
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,

    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,

    # Settings
    patience=50,
    save=True,
    save_period=10,
    workers=8,
    amp=True,
    val=True,
    plots=True
)

print("✅ Training complete!")
```

### Step 7: Validate Performance

```python
# Cell 7: Validation
best_model_path = "tomato_disease_detection/yolov8-cbam/weights/best.pt"
best_model = YOLO(best_model_path)

# Validate
val_results = best_model.val(data=data_yaml_path)

# Print metrics
print("=" * 80)
print("📊 Validation Metrics:")
print("=" * 80)
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"mAP50:    {val_results.box.map50:.4f}")
print(f"mAP75:    {val_results.box.map75:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall:    {val_results.box.mr:.4f}")
print("=" * 80)
```

### Step 8: Visualize Results

```python
# Cell 8: Display results
from IPython.display import Image, display

results_dir = "tomato_disease_detection/yolov8-cbam"

# Training curves
display(Image(filename=f"{results_dir}/results.png", width=800))

# Confusion matrix
display(Image(filename=f"{results_dir}/confusion_matrix.png", width=600))

# F1 curve
display(Image(filename=f"{results_dir}/F1_curve.png", width=600))

# PR curve
display(Image(filename=f"{results_dir}/PR_curve.png", width=600))
```

### Step 9: Test Predictions

```python
# Cell 9: Run predictions
test_dir = os.path.join(DATASET_DIR, "images", "test")

if os.path.exists(test_dir):
    predictions = best_model.predict(
        source=test_dir, save=True, conf=0.25, project="tomato_disease_detection", name="predictions"
    )
    print("✓ Predictions saved!")
```

### Step 10: Export Model

```python
# Cell 10: Export for deployment
# ONNX format
onnx_path = best_model.export(format="onnx", imgsz=640)
print(f"✓ ONNX exported: {onnx_path}")

# TorchScript format
ts_path = best_model.export(format="torchscript", imgsz=640)
print(f"✓ TorchScript exported: {ts_path}")
```

---

## 📊 Expected Performance

Based on CBAM and P2 feature enhancements:

- **+3-8% mAP** improvement over standard YOLOv8
- Better detection of **small lesions** (P2 feature)
- Improved performance in **complex backgrounds** (CBAM)
- **No inference speed penalty**

---

## 💾 Download Your Model

1. After training, go to Kaggle **Output** tab
2. Download the entire output folder
3. Your weights are in: `tomato_disease_detection/yolov8-cbam/weights/`
   - `best.pt` - Best performing weights
   - `last.pt` - Latest epoch weights

---

## 🚀 Using Your Trained Model

```python
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Predict on image
results = model.predict("tomato_leaf.jpg", conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls]} ({conf:.2f})")
```

---

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
BATCH_SIZE = 8  # or even 4
```

### Model Not Found Error

```python
# Make sure you committed the enhanced model to your GitHub repo
# Check: ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml
```

### Dataset Not Found

```python
# Verify Kaggle dataset name
# Check: /kaggle/input/your-dataset-name
print(os.listdir("/kaggle/input"))
```

### Slow Training

```python
# Enable AMP (Automatic Mixed Precision)
amp = True

# Reduce image size
imgsz = 512

# Use smaller model
scale = "n"  # nano model
```

---

## 📈 Monitor Training

### In Notebook:

- Check console output for real-time metrics
- View plots in output folder after training

### After Training:

- `results.png` - Training curves
- `confusion_matrix.png` - Class confusion
- `F1_curve.png` - F1 score vs confidence
- `PR_curve.png` - Precision-Recall curve

---

## 🎓 Tips for Better Results

1. **Data Quality**
   - Ensure balanced classes (similar number of images per class)
   - Use high-quality, clear images
   - Diverse lighting and backgrounds

2. **Augmentation**
   - Use strong augmentation for small datasets
   - `mosaic=1.0, mixup=0.5` for robustness

3. **Training Time**
   - Start with 100 epochs for testing
   - Use 200-300 epochs for final model
   - Enable `patience=50` for early stopping

4. **Model Scale**
   - `n`: Fastest, lowest accuracy
   - `s`: **Recommended** (balanced)
   - `m`: Higher accuracy, slower
   - `l`, `x`: Maximum accuracy, much slower

5. **Hyperparameters**
   - Lower `lr0` if loss explodes
   - Increase `patience` if improving slowly
   - Adjust `conf` threshold during inference

---

## 📞 Need Help?

- Check Kaggle notebook output for errors
- Review training curves for issues
- Compare with baseline YOLOv8 performance
- Ensure dataset format is correct

---

## ✅ Final Checklist

Before training:

- [ ] GitHub repo cloned successfully
- [ ] Dataset copied to working directory
- [ ] `tomato_data.yaml` created
- [ ] Enhanced model config found
- [ ] GPU enabled in Kaggle notebook
- [ ] All paths updated correctly

After training:

- [ ] Training completed without errors
- [ ] Validation metrics look reasonable
- [ ] Weights saved successfully
- [ ] Predictions tested on sample images
- [ ] Model exported for deployment

---

**🍅 Good luck training your enhanced YOLOv8-CBAM model for tomato disease detection! 🚀**
