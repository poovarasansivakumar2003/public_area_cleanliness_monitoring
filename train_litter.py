#!/usr/bin/env python3
"""
Train YOLOv8 on the real TACO litter dataset for public-area cleanliness monitoring.

Key design decisions that make the model predict well:
------------------------------------------------------
1. Model size: yolov8s (small) by default instead of nano — more capacity for diverse litter.
2. imgsz=1280: TACO has many tiny objects (30% < 2% of frame). Higher resolution is critical.
3. batch=8: 1280px tiles need ~3x more memory than 640px.
4. Epochs=100 with cos_lr=True and warmup: gives the scheduler room to converge.
5. Augmentation tuned for floor/ground cameras:
   - flipud=0.5: overhead CCTV; litter has no upright orientation
   - degrees=15: camera tilt and object rotation invariance
   - scale=0.6: objects at varying distances from camera
   - mosaic=1.0 + close_mosaic=20: helps small-object detection significantly
   - copy_paste=0.3: places extra litter instances on clean backgrounds
6. Proper device detection: tries CUDA env var before torch.cuda.is_available().
7. Saves best.pt to models/ and exports ONNX + TorchScript automatically.
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


def get_device() -> str:
    """
    Robust device selection.
    Tries CUDA_VISIBLE_DEVICES, then torch.cuda.is_available(), then falls back to cpu.
    """
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env and cuda_env != "-1":
        return "0"
    if torch.cuda.is_available():
        return "0"
    # torch.version.cuda is set when torch was built with CUDA support;
    # if so the GPU may still work even if is_available() returns False in a sandbox.
    if torch.version.cuda:
        try:
            torch.zeros(1, device="cuda")
            return "0"
        except Exception:
            pass
    return "cpu"


def train(
    data_yaml: str = "litter_yolo_dataset/data.yaml",
    base_model: str = "yolov8s.pt",    # 's' gives much better accuracy than 'n' for litter
    epochs: int = 100,
    imgsz: int = 1280,                 # critical for small objects (bottle caps, cigarettes…)
    batch: int = 8,                    # reduce if GPU VRAM < 8 GB; use -1 for auto-batch
    device: str = "",
    workers: int = 4,
    patience: int = 30,                # allow more time before early stopping
    project_dir: str = "runs_litter",
    exp_name: str = "taco_litter",
    class_mode: str = "single",
    export_deployment: bool = True,
):
    print("=" * 60)
    print("   PUBLIC AREA CLEANLINESS — YOLOv8 TRAINING ON TACO   ")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Ensure the converted dataset exists; run conversion if it does not
    # ------------------------------------------------------------------ #
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        print(f"\n[INFO] data.yaml not found at '{data_yaml}'. Running dataset conversion…")
        data_yaml = convert_coco_to_yolo(
            coco_ann_path="TACO/data/annotations.json",
            images_root="TACO/data",
            out_dir=str(data_yaml_path.parent),
            class_mode=class_mode,
        )
        data_yaml_path = Path(data_yaml)
    else:
        # Refresh absolute path inside the yaml (in case the directory was moved)
        text = data_yaml_path.read_text(encoding="utf-8")
        out_abs = str(data_yaml_path.parent.resolve())
        if "path:" not in text or out_abs not in text:
            lines = []
            for line in text.splitlines():
                if line.startswith("path:"):
                    lines.append(f"path: {out_abs}")
                else:
                    lines.append(line)
            data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[INFO] Updated data.yaml path to: {out_abs}")

    # ------------------------------------------------------------------ #
    # 2. Device selection
    # ------------------------------------------------------------------ #
    if not device:
        device = get_device()
    print(f"\n[INFO] Using device: '{device}'")
    if device != "cpu":
        try:
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 3. Initialise model
    # ------------------------------------------------------------------ #
    print(f"\n[INFO] Initialising base model: {base_model}")
    model = YOLO(base_model)

    # ------------------------------------------------------------------ #
    # 4. Train
    # ------------------------------------------------------------------ #
    print(f"\n[INFO] Starting training — {epochs} epochs, imgsz={imgsz}, batch={batch}")
    model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=project_dir,
        name=exp_name,
        exist_ok=True,          # resume / overwrite run directory
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,               # final lr = lr0 * lrf
        cos_lr=True,            # cosine annealing → smoother convergence
        warmup_epochs=5,
        weight_decay=0.0005,
        # Augmentation — tuned for floor/ground cameras
        flipud=0.5,             # overhead CCTV: no gravity cue
        fliplr=0.5,
        degrees=15.0,           # camera tilt + rotated bottles / bags
        translate=0.1,
        scale=0.6,              # objects at varying distances
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,             # multi-image tiling — greatly improves small objects
        mixup=0.15,
        copy_paste=0.3,         # paste litter instances on clean backgrounds
        close_mosaic=20,        # disable mosaic in last 20 epochs for stability
        # Misc
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
        # Loss weights — upweight box regression slightly for small objects
        box=8.0,
        cls=0.5,
        dfl=1.5,
    )

    # ------------------------------------------------------------------ #
    # 5. Validate
    # ------------------------------------------------------------------ #
    print("\n[INFO] Running final validation…")
    val_metrics = model.val(imgsz=imgsz, device=device)
    map50     = float(val_metrics.box.map50)
    map50_95  = float(val_metrics.box.map)
    precision = float(val_metrics.box.mp)
    recall    = float(val_metrics.box.mr)

    print(f"  Validation mAP@0.50     : {map50:.4f}")
    print(f"  Validation mAP@0.50:0.95: {map50_95:.4f}")
    print(f"  Mean Precision          : {precision:.4f}")
    print(f"  Mean Recall             : {recall:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Save best weights to models/
    # ------------------------------------------------------------------ #
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_pt = models_dir / "litter_detector_best.pt"

    trained_best = Path(project_dir) / exp_name / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, best_pt)
        print(f"\n[INFO] Best weights saved to: {best_pt.resolve()}")
    else:
        model.save(str(best_pt))
        print(f"\n[INFO] Model saved to: {best_pt.resolve()}")

    # ------------------------------------------------------------------ #
    # 7. Save training metadata
    # ------------------------------------------------------------------ #
    meta = {
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "class_mode": class_mode,
        "device": device,
        "metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
        "weights": str(best_pt.resolve()),
    }
    meta_path = models_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Training metadata written to: {meta_path.resolve()}")

    # ------------------------------------------------------------------ #
    # 8. Export for deployment
    # ------------------------------------------------------------------ #
    if export_deployment:
        from export_model import export_all_formats
        export_all_formats(str(best_pt), imgsz=imgsz, out_dir=str(models_dir))

    print("\n[DONE] Training and export complete.")
    return str(best_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Litter/Cleanliness Monitoring.")
    parser.add_argument("--data",       default="litter_yolo_dataset/data.yaml")
    parser.add_argument("--model",      default="yolov8s.pt",
                        help="Base weights: yolov8n.pt (fast), yolov8s.pt (recommended), yolov8m.pt (best accuracy)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--imgsz",      type=int,   default=1280,
                        help="Use 1280 for small objects (recommended). Use 640 if VRAM < 4 GB.")
    parser.add_argument("--batch",      type=int,   default=8,
                        help="Batch size. Use 4 for 1280px on 4-6 GB GPU, -1 for auto.")
    parser.add_argument("--device",     default="",
                        help="'0' for first GPU, 'cpu' for CPU, '' to auto-detect.")
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--patience",   type=int,   default=30)
    parser.add_argument("--project",    default="runs_litter")
    parser.add_argument("--name",       default="taco_litter")
    parser.add_argument("--class-mode", choices=["single", "supercategory", "all"], default="single")
    parser.add_argument("--no-export",  action="store_true", help="Skip deployment export")

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
        class_mode=args.class_mode,
        export_deployment=not args.no_export,
    )