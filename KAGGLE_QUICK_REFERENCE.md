# 🚀 Kaggle Training - Quick Reference Card

## ⚡ Quick Start (5 Steps)

### 1️⃣ Commit to GitHub
```bash
cd "C:\Users\seinp\Documents\yolo training\ultralytics-main"
git add .
git commit -m "Add YOLOv8-CBAM enhanced model"
git push origin main
```

### 2️⃣ Kaggle Setup
- Create new notebook on Kaggle
- Enable **GPU** (Settings → Accelerator → GPU T4 x2)
- Add your tomato dataset as **Input**
- Copy cells from `kaggle_train_cells.py`

### 3️⃣ Update Paths (2 places to change!)

**Cell 2 - Your GitHub URL:**
```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/ultralytics-main.git"
```

**Cell 3 - Your Kaggle dataset:**
```python
KAGGLE_INPUT = '/kaggle/input/your-dataset-name'  # CHANGE THIS!
```

### 4️⃣ Run All Cells
- Click "Run All" or run cells sequentially
- Training takes ~2-4 hours for 200 epochs

### 5️⃣ Download Results
- Go to **Output** tab in Kaggle
- Download: `tomato_disease_detection/yolov8-cbam/weights/best.pt`

---

## 📋 Complete Cell-by-Cell Checklist

| Cell | Action | Status |
|------|--------|--------|
| 1 | Install dependencies | ⬜ |
| 2 | Clone your GitHub repo (UPDATE URL!) | ⬜ |
| 3 | Copy dataset to working dir (UPDATE PATH!) | ⬜ |
| 4 | Create data.yaml | ⬜ |
| 5 | Verify enhanced model | ⬜ |
| 6 | Train model (2-4 hours) | ⬜ |
| 7 | Evaluate performance | ⬜ |
| 8 | View training curves | ⬜ |
| 9 | Test predictions | ⬜ |
| 10 | Export ONNX/TorchScript | ⬜ |
| 11 | View summary | ⬜ |

---

## 🎯 Training Parameters (Adjust in Cell 6)

```python
EPOCHS = 200          # Training epochs
BATCH_SIZE = 16       # GPU memory (reduce if OOM)
IMAGE_SIZE = 640      # Input image size
MODEL_SCALE = 's'     # Model variant (n/s/m/l/x)
```

**Model Scales:**
- `n` - Nano: Fastest, lowest accuracy
- `s` - Small: **RECOMMENDED** ⭐
- `m` - Medium: Higher accuracy
- `l` - Large: Even higher accuracy
- `x` - XLarge: Maximum accuracy

---

## 📊 Expected Training Time

| Model Scale | Batch Size | Time per Epoch | Total (200 epochs) |
|-------------|------------|----------------|-------------------|
| n | 32 | ~30 sec | ~2 hours |
| s | 16 | ~45 sec | ~3 hours |
| m | 8 | ~90 sec | ~5 hours |
| l | 4 | ~2 min | ~7 hours |

*Times are approximate for Kaggle P100 GPU*

---

## ⚠️ Common Issues & Quick Fixes

### Issue 1: "Dataset not found"
```python
# Check available datasets
print(os.listdir('/kaggle/input'))

# Update path in Cell 3
KAGGLE_INPUT = '/kaggle/input/CORRECT-NAME-HERE'
```

### Issue 2: "Out of memory"
```python
# In Cell 6, reduce batch size
BATCH_SIZE = 8  # or even 4
```

### Issue 3: "Model config not found"
```python
# Make sure you committed the enhanced model files:
# ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml
# ultralytics/nn/modules/block.py
```

### Issue 4: "Import error"
```python
# Re-run Cell 1 to reinstall dependencies
!pip install --upgrade ultralytics opencv-python-headless
```

---

## 📁 Your Dataset Structure

```
/kaggle/input/your-dataset-name/
├── images/
│   ├── train/      # Training images
│   ├── valid/      # Validation images
│   └── test/       # Test images
└── labels/
    ├── train/      # Training labels (.txt)
    ├── valid/      # Validation labels (.txt)
    └── test/       # Test labels (.txt)
```

---

## 🎓 Your 9 Classes

1. Early Blight
2. Healthy
3. Late Blight
4. Leaf Miner
5. Leaf Mold
6. Mosaic Virus
7. Septoria
8. Spider Mites
9. Yellow Leaf Curl Virus

---

## 📈 Monitoring Training

### Real-time Metrics (Console)
- `box_loss` - Bounding box loss (lower is better)
- `cls_loss` - Classification loss (lower is better)
- `dfl_loss` - Distribution focal loss (lower is better)
- `mAP50` - Mean Average Precision at IoU 0.5
- `mAP50-95` - mAP averaged over IoU 0.5-0.95

### Target Metrics
- **mAP50-95**: > 0.60 (good), > 0.70 (excellent)
- **mAP50**: > 0.80 (good), > 0.90 (excellent)
- **Precision**: > 0.80
- **Recall**: > 0.75

---

## 💾 Output Files

After training, you'll have:

```
tomato_disease_detection/yolov8-cbam/
├── weights/
│   ├── best.pt        # ⭐ Best performing model
│   └── last.pt        # Latest epoch model
├── results.png        # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── predictions/       # Test predictions
```

---

## 🚀 Using Your Trained Model

### Inference Code
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Single image
results = model.predict('tomato_leaf.jpg', conf=0.25)

# Batch prediction
results = model.predict('images/', save=True)

# Real-time webcam
results = model.predict(source=0, stream=True)

# Get predictions
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        disease = model.names[cls]
        print(f"{disease}: {conf:.2f}")
```

---

## 🎯 Performance Tips

### For Better Accuracy:
1. Use `scale='m'` or `scale='l'`
2. Increase `EPOCHS` to 300
3. Enable stronger augmentation
4. Ensure balanced dataset

### For Faster Training:
1. Use `scale='n'` or `scale='s'`
2. Reduce `IMAGE_SIZE` to 512
3. Increase `BATCH_SIZE` if GPU allows
4. Set `cache=True` (uses more RAM)

### For Better Generalization:
1. Use strong data augmentation
2. Enable `mosaic=1.0, mixup=0.5`
3. Increase `patience` to 100
4. Use dropout if overfitting

---

## 📞 Need Help?

1. **Check console output** for error messages
2. **View training curves** - look for convergence
3. **Compare metrics** with baseline YOLOv8
4. **Verify dataset** - check labels format
5. **Review Kaggle logs** - check GPU usage

---

## ✅ Success Checklist

Before starting training:
- [ ] GitHub repo has all enhanced files
- [ ] Kaggle GPU is enabled
- [ ] Dataset uploaded to Kaggle
- [ ] Both paths updated (GitHub + Dataset)
- [ ] Dataset structure is correct
- [ ] Labels are in YOLO format

After training:
- [ ] Training completed without errors
- [ ] Validation metrics are reasonable
- [ ] Confusion matrix looks good
- [ ] Weights downloaded from Kaggle
- [ ] Test predictions are accurate

---

## 🎉 Expected Results

With YOLOv8-CBAM enhancements:
- ✨ **+3-8% mAP** over standard YOLOv8
- ✨ **Better small lesion detection** (P2 feature)
- ✨ **Improved complex backgrounds** (CBAM)
- ✨ **4-level detection** (P2, P3, P4, P5)

---

**🍅 Happy training! Your enhanced model is ready to detect tomato diseases! 🚀**

---

**Files to reference:**
- `KAGGLE_TRAINING_GUIDE.md` - Complete detailed guide
- `kaggle_train_cells.py` - All code cells
- `KAGGLE_QUICK_REFERENCE.md` - This file

