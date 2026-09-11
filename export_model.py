#!/usr/bin/env python3
"""
Model Export for Production Deployment — Cleanliness Monitoring.

Exports trained YOLO weights to:
1. ONNX  (dynamic batching, opset 17, simplified graph)
2. TorchScript (PyTorch C++ / LibTorch)
3. deployment_config.json  — runtime parameters and class map

Tested with Ultralytics 8.x and ONNX opset 17.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def export_all_formats(
    model_path: str,
    imgsz: int = 1280,
    out_dir: str = "models",
) -> dict:
    """
    Load a trained .pt model and export to ONNX and TorchScript.
    Writes deployment_config.json alongside the exported files.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLOv8 .pt weights.
    imgsz : int
        Inference image size used during training.  Must match training imgsz
        so that output strides are correct.
    out_dir : str
        Directory where deployment artefacts are written.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Exporting model for deployment: {model_file.resolve()} ---")
    model = YOLO(str(model_file))

    class_map = {int(k): str(v) for k, v in model.names.items()}
    print(f"Classes ({len(class_map)}): {list(class_map.values())[:10]}")

    exported_files = {"weights_pt": str(model_file.resolve())}

    # ------------------------------------------------------------------ #
    # 1. ONNX — preferred for cross-platform / edge deployment
    # ------------------------------------------------------------------ #
    try:
        print("\nExporting to ONNX (opset 17, dynamic batching)…")
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,       # variable batch size and spatial dims
            simplify=True,      # onnxslim / onnxsim
            opset=17,           # opset 17 works with ORT ≥ 1.15 and TensorRT ≥ 8.6
        )
        print(f"  ONNX exported: {onnx_path}")
        exported_files["onnx"] = str(onnx_path)
    except Exception as exc:
        print(f"  ONNX export failed: {exc}")

    # ------------------------------------------------------------------ #
    # 2. TorchScript — for LibTorch / C++ inference
    # ------------------------------------------------------------------ #
    try:
        print("\nExporting to TorchScript…")
        ts_path = model.export(
            format="torchscript",
            imgsz=imgsz,
        )
        print(f"  TorchScript exported: {ts_path}")
        exported_files["torchscript"] = str(ts_path)
    except Exception as exc:
        print(f"  TorchScript export failed: {exc}")

    # ------------------------------------------------------------------ #
    # 3. deployment_config.json
    # ------------------------------------------------------------------ #
    config = {
        "model_name": "cleanliness_litter_detector",
        "export_date": datetime.now().isoformat(),
        "input_resolution": [imgsz, imgsz],
        "input_channels": 3,
        "input_color_space": "RGB",
        "normalisation": "divide_by_255",
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "classes": class_map,
        "exported_files": exported_files,
    }
    config_path = out_path / "deployment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nDeployment config: {config_path.resolve()}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained YOLO model for deployment.")
    parser.add_argument("--model",   default="models/litter_detector_best.pt")
    parser.add_argument("--imgsz",   type=int, default=1280,
                        help="Must match the imgsz used during training.")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    export_all_formats(model_path=args.model, imgsz=args.imgsz, out_dir=args.out_dir)