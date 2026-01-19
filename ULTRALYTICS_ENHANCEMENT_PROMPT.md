# 🎯 Ultralytics YOLO Enhancement Prompt for Cursor AI

## 📋 Copy-Paste This to Cursor AI:

---

I need to enhance the Ultralytics YOLOv8 architecture for tomato leaf disease detection by adding:

1. **CBAM (Convolutional Block Attention Module)** to the backbone
2. **BiRepGFPN with P2 feature fusion** to replace the standard neck

## 🏗️ Architecture Modifications Required:

### **1. CBAM Integration (Backbone Enhancement)**

**Location**: Insert CBAM modules after backbone CSP layers

**CBAM Specifications:**

- **Channel Attention Module**:
  - Apply global average pooling and global max pooling to input features F
  - Feed pooled features into a 3-layer MLP with reduction ratio of 16
  - Element-wise sum the MLP outputs
  - Apply Sigmoid activation to get channel attention map M_c(F)
  - Formula: `M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))`

- **Spatial Attention Module**:
  - Input: Feature map F' from channel attention
  - Apply max pooling and average pooling across channels
  - Concatenate pooled maps along channel dimension
  - Apply 7×7 convolution
  - Apply Sigmoid activation to get spatial attention map M_s(F')
  - Formula: `M_s(F') = σ(Conv7×7(Concat(AvgPool(F'), MaxPool(F'))))`

**Implementation:**

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
```

**Where to Insert:**

- After C3/C2f blocks in backbone layers
- Specifically after the 3rd, 4th, and 5th backbone stages
- Insert CBAM before features are passed to the neck

---

### **2. BiRepGFPN with P2 Feature Fusion (Neck Replacement)**

**Current Ultralytics**: Uses PANet (PAFPN) with P3, P4, P5 features

**Required Change**:

- Add P2 feature map (downsampled by factor of 4 from input image)
- Implement BiRepGFPN that fuses P2, P3, P4, P5 features
- P2 should be downsampled and fused with P3 feature map

**BiRepGFPN Specifications:**

- Base: Reparameterized Generalized Feature Pyramid Network (RepGFPN)
- Add P2 output from backbone (stride 4)
- Multi-scale feature fusion with P2→P3 connection
- Bi-directional feature flow (top-down + bottom-up)
- Enable aggregation of shallow spatial information (P2) with deep semantic information

**Implementation Approach:**

```python
class BiRepGFPN(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], out_channel=256):
        """
        Args:
            in_channels: [P2, P3, P4, P5] channel counts
            out_channel: Output channel count for all pyramid levels.
        """
        super().__init__()
        # P2, P3, P4, P5 inputs

        # Top-down pathway
        self.lateral_p5 = nn.Conv2d(in_channels[3], out_channel, 1)
        self.lateral_p4 = nn.Conv2d(in_channels[2], out_channel, 1)
        self.lateral_p3 = nn.Conv2d(in_channels[1], out_channel, 1)
        self.lateral_p2 = nn.Conv2d(in_channels[0], out_channel, 1)

        # Reparameterizable blocks for fusion
        self.rep_p4 = RepBlock(out_channel)
        self.rep_p3 = RepBlock(out_channel)
        self.rep_p2 = RepBlock(out_channel)

        # Bottom-up pathway
        self.downsample_p2 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        self.downsample_p3 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        self.downsample_p4 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

    def forward(self, features):
        # features = [P2, P3, P4, P5]
        # Implement bi-directional feature fusion
        pass
```

**Key Requirements:**

- Output P2 from backbone (modify backbone to include early feature map)
- Downsample P2 by 2x and fuse with P3
- Maintain feature pyramid with 4 levels: P2, P3, P4, P5
- Use reparameterizable convolutions (3x3 + 1x1 branches that merge during inference)

---

## 🎯 Specific Tasks:

1. **Create CBAM Module**:
   - File: `ultralytics/nn/modules/block.py`
   - Add ChannelAttention, SpatialAttention, and CBAM classes

2. **Modify Backbone**:
   - File: `ultralytics/nn/modules/head.py` or model config YAML
   - Insert CBAM after C2f/C3 blocks at stages 3, 4, 5
   - Ensure P2 feature map is output (stride 4)

3. **Create BiRepGFPN**:
   - File: `ultralytics/nn/modules/block.py`
   - Implement BiRepGFPN class with P2 support
   - Add RepBlock for reparameterization

4. **Update Model Config**:
   - File: `ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml`
   - Define new architecture with CBAM and BiRepGFPN
   - Specify P2, P3, P4, P5 feature extraction

5. **Register New Modules**:
   - File: `ultralytics/nn/tasks.py`
   - Register CBAM and BiRepGFPN in module registry

---

## 📝 Configuration YAML Example:

```yaml
# YOLOv8 with CBAM + BiRepGFPN

# Parameters
nc: 9 # number of classes (tomato diseases)
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]

# Backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4  ⭐ Output this
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, CBAM, [128]] # ⭐ CBAM after C2f
  - [-1, 1, Conv, [256, 3, 2]] # 4-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, CBAM, [256]] # ⭐ CBAM
  - [-1, 1, Conv, [512, 3, 2]] # 7-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, CBAM, [512]] # ⭐ CBAM
  - [-1, 1, Conv, [1024, 3, 2]] # 10-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 12

# Head
head:
  - [[2, 5, 8, 12], 1, BiRepGFPN, [256]] # ⭐ P2, P3, P4, P5 → BiRepGFPN
  -  # Detection layers follow
```

---

## 🎯 Training Command After Implementation:

```bash
yolo train model=yolov8-cbam-birepgfpn.yaml data=tomato.yaml epochs=200 imgsz=640 batch=32
```

---

## 📋 Validation Steps:

1. Model loads without errors
2. CBAM modules are in backbone (check model.model)
3. BiRepGFPN receives P2, P3, P4, P5 features
4. Output has correct number of detection heads
5. Can train on tomato disease dataset

---

## 🔍 Reference Implementation:

- CBAM Paper: https://arxiv.org/abs/1807.06521
- RepVGG (for reparameterization): https://arxiv.org/abs/2101.03697
- YOLOv6 (has similar modifications): https://github.com/meituan/YOLOv6

---

## 💡 Expected Benefits:

1. **CBAM**: Better feature selection, improved detection in complex backgrounds
2. **P2 Feature**: Better small object detection (small lesions)
3. **BiRepGFPN**: Enhanced multi-scale feature fusion

---

## ✅ Final Checklist:

- [ ] CBAM module created in `ultralytics/nn/modules/block.py`
- [ ] CBAM inserted after 3 backbone stages
- [ ] BiRepGFPN created with P2 support
- [ ] Model config YAML created
- [ ] Modules registered in `ultralytics/nn/tasks.py`
- [ ] Model can be instantiated and trained
- [ ] Forward pass works with dummy input

---

Please implement these enhancements to Ultralytics YOLOv8, ensuring the architecture matches the specifications above. Create all necessary files and provide the final training command.
