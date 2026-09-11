#!/usr/bin/env python3
"""
Train YOLOv8 on real litter dataset (TACO) for public area cleanliness monitoring.

Features:
- Configurable base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)
- Tuned augmentations for floor camera perspectives (flipud, fliplr, rotation, mosaic, mixup)
- Automatic device detection (CUDA GPU / CPU)
- Automatic dataset conversion if not already converted
- Saves best weights to models/litter_detector_best.pt
- Automatically exports to ONNX and writes deployment metadata
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

from convert_dataset import convert_coco_to_yolo


def train(
    data_yaml: str = "litter_yolo_dataset/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    workers: int = 4,
    patience: int = 15,
    project_dir: str = "runs_litter",
    exp_name: str = "taco_yolov8",
    export_deployment: bool = True,
    class_mode: str = "single",
):
    print("=" * 60)
    print("   PUBLIC AREA CLEANLINESS MONITORING — YOLOv8 TRAINING   ")
    print("=" * 60)

    # 1. Ensure dataset exists
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        print(f"Dataset YAML not found at '{data_yaml}'. Running dataset conversion...")
        data_yaml = convert_coco_to_yolo(
            coco_ann_path="TACO/data/annotations.json",
            images_root="TACO/data",
            out_dir=str(data_yaml_path.parent),
            class_mode=class_mode,
        )
        data_yaml_path = Path(data_yaml)

    # 2. Determine device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 3. Initialize model
    print(f"Initializing base model: {base_model}")
    model = YOLO(base_model)

    # 4. Train model with augmentations optimized for floor litter
    print(f"Starting training for {epochs} epochs, imgsz={imgsz}, batch={batch}...")
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=project_dir,
        name=exp_name,
        # Augmentations suitable for floor/ground trash:
        flipud=0.5,      # top-down floor objects have no preferred vertical orientation
        fliplr=0.5,
        degrees=10.0,    # rotation invariance
        mosaic=1.0,      # multi-scale composition
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
    )

    # 5. Validate model
    print("\n--- Running Final Validation ---")
    val_metrics = model.val()
    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    mp = float(val_metrics.box.mp)
    mr = float(val_metrics.box.mr)

    print(f"Validation mAP@0.50     : {map50:.4f}")
    print(f"Validation mAP@0.50:0.95: {map50_95:.4f}")
    print(f"Validation Mean Precision: {mp:.4f}")
    print(f"Validation Mean Recall   : {mr:.4f}")

    # 6. Save best weights to models/
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_pt = models_dir / "litter_detector_best.pt"

    trained_best = Path(project_dir) / exp_name / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, best_pt)
        print(f"Best model weights saved to: {best_pt.resolve()}")
    else:
        model.save(str(best_pt))
        print(f"Model saved to: {best_pt.resolve()}")

    # 7. Export for deployment if requested
    if export_deployment:
        from export_model import export_all_formats
        export_all_formats(str(best_pt), imgsz=imgsz, out_dir=str(models_dir))

    print("\nTraining completed successfully!")
    return str(best_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Litter/Cleanliness Monitoring.")
    parser.add_argument("--data", default="litter_yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights: yolov8n.pt, yolov8s.pt, etc.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto-batch)")
    parser.add_argument("--device", default="", help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker count")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", default="runs_litter", help="Output project directory")
    parser.add_argument("--name", default="taco_yolov8", help="Experiment name")
    parser.add_argument("--class-mode", choices=["single", "supercategory", "all"], default="single",
                        help="Dataset class mode: single, supercategory, all")
    parser.add_argument("--no-export", action="store_true", help="Skip deployment format export")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project_dir=args.project,
        exp_name=args.name,
        export_deployment=not args.no_export,
        class_mode=args.class_mode,
    )

