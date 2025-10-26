"""
YOLOv8-CBAM Training on Kaggle - Copy each cell to Kaggle notebook
Complete training pipeline for tomato disease detection
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================
!pip install -q ultralytics==8.0.196
!pip install -q opencv-python-headless PyYAML

import os
import shutil
import yaml
from pathlib import Path
import torch

print("✓ Dependencies installed")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# =============================================================================
# CELL 2: Clone Your Enhanced YOLOv8-CBAM Repository
# =============================================================================
# ⚠️ UPDATE THIS with your GitHub repository URL
GITHUB_REPO = "https://github.com/YOUR_USERNAME/ultralytics-main.git"

!git clone {GITHUB_REPO}
os.chdir('ultralytics-main')
print(f"✓ Repository cloned")
print(f"Current directory: {os.getcwd()}")


# =============================================================================
# CELL 3: Copy Dataset from Readonly Input to Working Directory
# =============================================================================
# ⚠️ UPDATE THIS with your actual Kaggle dataset name
KAGGLE_INPUT = '/kaggle/input/your-dataset-name'  # CHANGE THIS!
WORKING_DIR = '/kaggle/working'
DATASET_DIR = os.path.join(WORKING_DIR, 'dataset')

print(f"Input: {KAGGLE_INPUT}")
print(f"Output: {DATASET_DIR}")

# Verify input exists
if not os.path.exists(KAGGLE_INPUT):
    print(f"❌ ERROR: Dataset not found at {KAGGLE_INPUT}")
    print(f"Available datasets:")
    print(os.listdir('/kaggle/input'))
else:
    # Copy dataset
    print("Copying dataset...")
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    shutil.copytree(KAGGLE_INPUT, DATASET_DIR)
    print("✓ Dataset copied")
    
    # Count images
    def count_images(directory):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return sum(1 for f in os.listdir(directory) 
                   if f.lower().endswith(exts)) if os.path.exists(directory) else 0
    
    train_count = count_images(os.path.join(DATASET_DIR, 'images', 'train'))
    val_count = count_images(os.path.join(DATASET_DIR, 'images', 'valid'))
    test_count = count_images(os.path.join(DATASET_DIR, 'images', 'test'))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train: {train_count} images")
    print(f"   Valid: {val_count} images")
    print(f"   Test: {test_count} images")
    print(f"   Total: {train_count + val_count + test_count} images")


# =============================================================================
# CELL 4: Create Data Configuration File
# =============================================================================
# Create tomato_data.yaml
data_yaml = {
    'path': DATASET_DIR,
    'train': 'images/train',
    'val': 'images/valid',
    'test': 'images/test',
    
    'nc': 9,  # Number of classes
    
    'names': [
        'Early Blight',
        'Healthy',
        'Late Blight',
        'Leaf Miner',
        'Leaf Mold',
        'Mosaic Virus',
        'Septoria',
        'Spider Mites',
        'Yellow Leaf Curl Virus'
    ]
}

# Save configuration
data_yaml_path = os.path.join(WORKING_DIR, 'tomato_data.yaml')
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✓ Data config saved: {data_yaml_path}")
print("\nConfiguration:")
with open(data_yaml_path, 'r') as f:
    print(f.read())


# =============================================================================
# CELL 5: Verify Enhanced Model Configuration
# =============================================================================
from ultralytics import YOLO

model_config = 'ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml'

print("Checking enhanced model...")

if os.path.exists(model_config):
    print(f"✓ Model config found: {model_config}")
    
    # Test load
    model = YOLO(model_config)
    print("✓ Model loaded successfully")
    
    # Check CBAM modules
    cbam_count = sum(1 for m in model.model.model if 'CBAM' in m.__class__.__name__)
    print(f"✓ Found {cbam_count} CBAM modules in backbone")
    
    # Check detection heads
    detect_layers = [m for m in model.model.model if 'Detect' in m.__class__.__name__]
    if detect_layers and hasattr(detect_layers[0], 'stride'):
        num_heads = len(detect_layers[0].stride)
        print(f"✓ Detection heads: {num_heads}")
        if num_heads == 4:
            print("✓ P2 support enabled (4-level detection)")
    
    print("\n✅ Model validation passed!")
else:
    print(f"❌ ERROR: Model config not found: {model_config}")
    print("Make sure you committed the enhanced model to your GitHub repo!")


# =============================================================================
# CELL 6: Train the Enhanced YOLOv8-CBAM Model
# =============================================================================
from ultralytics import YOLO

# ⚠️ ADJUST THESE PARAMETERS based on your needs
EPOCHS = 200
BATCH_SIZE = 16  # Reduce to 8 or 4 if out of memory
IMAGE_SIZE = 640
MODEL_SCALE = 's'  # Options: n (fastest), s (recommended), m, l, x (most accurate)

print("="*80)
print("🚀 Starting Training")
print("="*80)
print(f"Model: YOLOv8-CBAM ({MODEL_SCALE})")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Classes: 9 tomato diseases")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("="*80)

# Load model
model = YOLO(model_config)

# Train
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0 if torch.cuda.is_available() else 'cpu',
    
    # Model scale
    scale=MODEL_SCALE,
    
    # Project settings
    project='tomato_disease_detection',
    name='yolov8-cbam',
    exist_ok=False,
    
    # Optimization
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.0,
    
    # Training settings
    patience=50,
    save=True,
    save_period=10,
    cache=False,
    workers=8,
    verbose=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=10,
    amp=True,
    fraction=1.0,
    val=True,
    plots=True
)

print("\n" + "="*80)
print("✅ Training Complete!")
print("="*80)


# =============================================================================
# CELL 7: Evaluate Model Performance
# =============================================================================
from ultralytics import YOLO

# Load best model
best_model_path = 'tomato_disease_detection/yolov8-cbam/weights/best.pt'
print(f"Loading best model: {best_model_path}")

best_model = YOLO(best_model_path)

# Validate
print("\nRunning validation...")
val_results = best_model.val(data=data_yaml_path)

# Print metrics
print("\n" + "="*80)
print("📊 Validation Metrics")
print("="*80)
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"mAP50:    {val_results.box.map50:.4f}")
print(f"mAP75:    {val_results.box.map75:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall:    {val_results.box.mr:.4f}")
print("="*80)


# =============================================================================
# CELL 8: Visualize Training Results
# =============================================================================
from IPython.display import Image, display

results_dir = 'tomato_disease_detection/yolov8-cbam'

print("📈 Training Results")
print("="*80)

# Training curves
print("\nTraining Curves:")
display(Image(filename=f'{results_dir}/results.png', width=800))

# Confusion matrix
print("\nConfusion Matrix:")
display(Image(filename=f'{results_dir}/confusion_matrix.png', width=600))

# F1 curve
print("\nF1 Curve:")
display(Image(filename=f'{results_dir}/F1_curve.png', width=600))

# PR curve
print("\nPrecision-Recall Curve:")
display(Image(filename=f'{results_dir}/PR_curve.png', width=600))


# =============================================================================
# CELL 9: Test Predictions on Sample Images
# =============================================================================
import glob
from IPython.display import Image, display

test_dir = os.path.join(DATASET_DIR, 'images', 'test')

if os.path.exists(test_dir):
    print(f"Running predictions on: {test_dir}")
    
    # Predict
    predictions = best_model.predict(
        source=test_dir,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        iou=0.45,
        project='tomato_disease_detection',
        name='predictions',
        exist_ok=True
    )
    
    print(f"✓ Predictions saved to: tomato_disease_detection/predictions/")
    
    # Display first 5 predictions
    pred_dir = 'tomato_disease_detection/predictions'
    pred_images = sorted(glob.glob(f'{pred_dir}/*.jpg'))[:5]
    
    print(f"\n🖼️ Sample Predictions (first 5):")
    for img_path in pred_images:
        print(f"\n{os.path.basename(img_path)}:")
        display(Image(filename=img_path, width=600))
else:
    print(f"Test directory not found: {test_dir}")


# =============================================================================
# CELL 10: Export Model for Deployment
# =============================================================================
print("📦 Exporting model...")

# ONNX export
try:
    onnx_path = best_model.export(format='onnx', imgsz=IMAGE_SIZE)
    print(f"✓ ONNX exported: {onnx_path}")
except Exception as e:
    print(f"⚠ ONNX export failed: {e}")

# TorchScript export
try:
    ts_path = best_model.export(format='torchscript', imgsz=IMAGE_SIZE)
    print(f"✓ TorchScript exported: {ts_path}")
except Exception as e:
    print(f"⚠ TorchScript export failed: {e}")


# =============================================================================
# CELL 11: Final Summary
# =============================================================================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE - YOLOv8-CBAM")
print("="*80)

print("\n📁 Output Files:")
print(f"   Best weights: {best_model_path}")
print(f"   Last weights: tomato_disease_detection/yolov8-cbam/weights/last.pt")
print(f"   Results: tomato_disease_detection/yolov8-cbam/")

print("\n📊 Final Metrics:")
print(f"   mAP50-95: {val_results.box.map:.4f}")
print(f"   mAP50:    {val_results.box.map50:.4f}")
print(f"   Precision: {val_results.box.mp:.4f}")
print(f"   Recall:    {val_results.box.mr:.4f}")

print("\n✨ Enhanced Features:")
print(f"   ✓ {cbam_count} CBAM attention modules")
print("   ✓ P2 feature support for small objects")
print("   ✓ Bi-directional feature pyramid")
print("   ✓ 4-level detection (P2, P3, P4, P5)")

print("\n💾 To download your model:")
print("   1. Go to Kaggle Output tab")
print("   2. Download the entire output folder")
print("   3. Your weights: tomato_disease_detection/yolov8-cbam/weights/best.pt")

print("\n🚀 To use your trained model:")
print("   from ultralytics import YOLO")
print("   model = YOLO('best.pt')")
print("   results = model.predict('image.jpg')")

print("\n" + "="*80)
print("🍅 Happy detecting! Your tomato plants will thank you! 🌱")
print("="*80)

