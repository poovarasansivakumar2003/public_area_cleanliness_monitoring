#!/usr/bin/env python3
"""
Model Export & Deployment Preparation for Cleanliness Monitoring.

Exports trained YOLO weights to production formats:
1. ONNX (with dynamic batching and simplified computational graph)
2. TorchScript (for PyTorch C++ / libtorch deployment)
3. Generates deployment_config.json containing runtime parameters and class definitions
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def export_all_formats(model_path: str, imgsz: int = 640, out_dir: str = "models") -> dict:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Exporting Model for Deployment: {model_file.resolve()} ---")
    model = YOLO(str(model_file))

    # Extract class dictionary
    names = model.names
    class_map = {int(k): str(v) for k, v in names.items()}
    print(f"Model has {len(class_map)} classes: {list(class_map.values())[:10]}")

    exported_files = {"weights_pt": str(model_file.resolve())}

    # 1. Export to ONNX
    try:
        print("\nExporting to ONNX format...")
        onnx_file = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            simplify=True,
            opset=12,
        )
        print(f"ONNX exported successfully: {onnx_file}")
        exported_files["onnx"] = str(onnx_file)
    except Exception as e:
        print(f"ONNX export warning: {e}")

    # 2. Export to TorchScript
    try:
        print("\nExporting to TorchScript format...")
        ts_file = model.export(
            format="torchscript",
            imgsz=imgsz,
        )
        print(f"TorchScript exported successfully: {ts_file}")
        exported_files["torchscript"] = str(ts_file)
    except Exception as e:
        print(f"TorchScript export warning: {e}")

    # 3. Generate deployment metadata config
    config_file = out_path / "deployment_config.json"
    deployment_info = {
        "model_name": "cleanliness_litter_detector",
        "export_date": datetime.now().isoformat(),
        "input_resolution": [imgsz, imgsz],
        "input_channels": 3,
        "input_color_space": "RGB",
        "normalize": "divide_by_255",
        "default_conf_threshold": 0.35,
        "default_iou_threshold": 0.45,
        "classes": class_map,
        "exported_files": exported_files,
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"\nDeployment configuration written to: {config_file.resolve()}")
    return deployment_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained YOLO model for deployment.")
    parser.add_argument("--model", default="models/litter_detector_best.pt", help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--out-dir", default="models", help="Output directory for deployment files")
    args = parser.parse_args()

    export_all_formats(model_path=args.model, imgsz=args.imgsz, out_dir=args.out_dir)

