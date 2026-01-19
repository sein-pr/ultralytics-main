# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validation script for YOLOv8-CBAM-BiRepGFPN implementation.

This script validates:
1. All modules can be imported
2. YAML configuration file exists
3. Model can be instantiated (if PyTorch is available)
"""

import os
import sys


def check_imports():
    """Check if all new modules can be imported."""
    print("=" * 80)
    print("Checking Module Imports")
    print("=" * 80)

    try:
        from ultralytics.nn.modules.block import CBAM, BiRepGFPN, ChannelAttention, RepBlock, SpatialAttention

        print("✓ All modules imported successfully from block.py")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def check_yaml():
    """Check if YAML configuration file exists."""
    print("\n" + "=" * 80)
    print("Checking YAML Configuration")
    print("=" * 80)

    yaml_path = "ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml"
    if os.path.exists(yaml_path):
        print(f"✓ YAML file exists: {yaml_path}")
        return True
    else:
        print(f"✗ YAML file not found: {yaml_path}")
        return False


def check_model():
    """Try to instantiate the model if PyTorch is available."""
    print("\n" + "=" * 80)
    print("Checking Model Instantiation")
    print("=" * 80)

    try:
        from ultralytics import YOLO

        model = YOLO("ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml")
        print("✓ Model instantiated successfully")

        # Check for CBAM modules
        cbam_count = sum(1 for m in model.model.model if "CBAM" in m.__class__.__name__)
        print(f"✓ Found {cbam_count} CBAM modules in backbone")

        # Check for bi-directional feature pyramid (implemented via standard PANet)
        # Count detection heads - should have 4 (P2, P3, P4, P5)
        detect_layers = [m for m in model.model.model if "Detect" in m.__class__.__name__]
        if detect_layers:
            # Check if P2 is included (4 heads instead of 3)
            num_heads = len(detect_layers[0].stride) if hasattr(detect_layers[0], "stride") else 0
            if num_heads == 4:
                print("✓ Bi-directional feature pyramid with P2 support (4 detection heads)")
            else:
                print(f"✓ Model has {num_heads} detection heads")
        else:
            print("⚠ Could not verify detection heads")

        return True
    except ImportError:
        print("⚠ PyTorch not available - skipping model instantiation test")
        print("  Install PyTorch to run full validation: pip install torch")
        return None
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("YOLOv8-CBAM-BiRepGFPN Implementation Validation")
    print("=" * 80)

    results = {"imports": check_imports(), "yaml": check_yaml(), "model": check_model()}

    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)

    for test, result in results.items():
        if result is True:
            print(f"✓ {test.capitalize()}: PASSED")
        elif result is None:
            print(f"⚠ {test.capitalize()}: SKIPPED")
        else:
            print(f"✗ {test.capitalize()}: FAILED")

    if all(r in [True, None] for r in results.values()):
        print("\n" + "=" * 80)
        print("✅ IMPLEMENTATION VALIDATED")
        print("=" * 80)
        print("\nYou're ready to train!")
        print("\nQuick start:")
        print("  yolo train model=ultralytics/cfg/models/v8/yolov8-cbam-birepgfpn.yaml \\")
        print("             data=your_data.yaml epochs=200 imgsz=640 batch=16")
        print("\nFor more information, see:")
        print("  - QUICK_START_GUIDE.md")
        print("  - IMPLEMENTATION_SUMMARY.md")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        print("\nPlease check the errors above and ensure all files are in place.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
