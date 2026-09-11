#!/usr/bin/env python3
"""
Convert TACO (or custom COCO-format dataset) to YOLO format for YOLOv8 training.

Fixes:
1. Filename collision bug across batches (e.g., batch_1/000006.jpg vs batch_2/000006.jpg).
2. Clamping and validation of bounding box coordinates (prevents corrupted or negative coordinates).
3. Flexible class mapping:
   - 'single': All trash items mapped to 1 class 'litter' (Recommended for floor cleanliness monitoring).
   - 'supercategory': 28 supercategories (e.g. 'Bottle', 'Plastic bag & wrapper', 'Can', 'Cigarette').
   - 'all': Original 60 TACO categories.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_coco_annotations(ann_path: str) -> dict:
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_mapping(coco: dict, mode: str = "single") -> Tuple[Dict[int, int], List[str]]:
    """
    Builds mapping from original COCO category_id to YOLO class_id (0-indexed).
    Returns (id_map, class_names).
    """
    cats = coco["categories"]

    if mode == "single":
        id_map = {c["id"]: 0 for c in cats}
        class_names = ["litter"]
        return id_map, class_names

    elif mode == "supercategory":
        supercats = sorted(list(set(c["supercategory"] for c in cats if c.get("supercategory"))))
        supercat_to_id = {sc: i for i, sc in enumerate(supercats)}
        id_map = {c["id"]: supercat_to_id[c["supercategory"]] for c in cats if c.get("supercategory") in supercat_to_id}
        return id_map, supercats

    elif mode == "all":
        # Sort by ID to ensure consistent ordering
        sorted_cats = sorted(cats, key=lambda c: c["id"])
        id_map = {c["id"]: i for i, c in enumerate(sorted_cats)}
        class_names = [c["name"] for c in sorted_cats]
        return id_map, class_names

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'single', 'supercategory', or 'all'.")


def convert_coco_to_yolo(
    coco_ann_path: str = "TACO/data/annotations.json",
    images_root: str = "TACO/data",
    out_dir: str = "litter_yolo_dataset",
    val_split: float = 0.15,
    class_mode: str = "single",
    seed: int = 42,
    copy_files: bool = False,
) -> str:
    """
    Converts COCO dataset to YOLOv8 dataset format.
    """
    coco_path = Path(coco_ann_path)
    if not coco_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {coco_ann_path}")

    print(f"Loading annotations from {coco_ann_path}...")
    coco = load_coco_annotations(str(coco_path))

    cat_map, class_names = build_class_mapping(coco, mode=class_mode)
    print(f"Class mapping mode: '{class_mode}' ({len(class_names)} classes: {class_names[:5]}{'...' if len(class_names)>5 else ''})")

    img_info = {im["id"]: im for im in coco["images"]}
    anns_by_img: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out = Path(out_dir)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_ids = list(img_info.keys())
    np.random.seed(seed)
    np.random.shuffle(img_ids)
    n_val = int(len(img_ids) * val_split)
    val_ids = set(img_ids[:n_val])

    processed_count = 0
    skipped_missing = 0
    total_boxes = 0

    for img_id in img_ids:
        info = img_info[img_id]
        split = "val" if img_id in val_ids else "train"
        src_path = Path(images_root) / info["file_name"]

        if not src_path.exists():
            skipped_missing += 1
            continue

        # Prevent collision: batch_1/000006.jpg -> batch_1_000006.jpg
        safe_stem = info["file_name"].replace("/", "_").replace("\\", "_")
        safe_stem_without_ext = Path(safe_stem).stem
        ext = src_path.suffix.lower() if src_path.suffix else ".jpg"
        dst_img = out / "images" / split / f"{safe_stem_without_ext}{ext}"

        if not dst_img.exists():
            if copy_files or os.name == "nt":
                shutil.copy2(src_path, dst_img)
            else:
                try:
                    os.link(src_path, dst_img)
                except OSError:
                    shutil.copy2(src_path, dst_img)

        w, h = float(info["width"]), float(info["height"])
        label_lines = []

        for ann in anns_by_img.get(img_id, []):
            cat_id = ann.get("category_id")
            if cat_id not in cat_map:
                continue

            x, y, bw, bh = ann["bbox"]
            # Validate box dimensions
            if bw <= 0 or bh <= 0:
                continue

            # Convert COCO (top-left x, y, width, height) to YOLO (center_x, center_y, width, height) normalized
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h

            # Clamp coordinates to [0, 1]
            cx = min(max(0.0, cx), 1.0)
            cy = min(max(0.0, cy), 1.0)
            nw = min(max(0.0001, nw), 1.0)
            nh = min(max(0.0001, nh), 1.0)

            cls_id = cat_map[cat_id]
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        label_path = out / "labels" / split / f"{safe_stem_without_ext}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        processed_count += 1

    # Write data.yaml for YOLOv8
    yaml_path = out / "data.yaml"
    clean_names = [name.replace("'", "") for name in class_names]
    names_str = ", ".join([f"'{name}'" for name in clean_names])
    yaml_content = f"""# YOLOv8 Dataset Configuration - Public Area Cleanliness Monitoring
path: {out.resolve()}
train: images/train
val: images/val
nc: {len(class_names)}
names: [{names_str}]
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\n--- Dataset Conversion Summary ---")
    print(f"Output directory : {out.resolve()}")
    print(f"Total images     : {processed_count} processed ({skipped_missing} missing)")
    print(f"Total boxes      : {total_boxes}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Config YAML      : {yaml_path.resolve()}")
    return str(yaml_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TACO COCO dataset to YOLOv8 format.")
    parser.add_argument("--coco-ann", default="TACO/data/annotations.json", help="Path to annotations.json")
    parser.add_argument("--images-root", default="TACO/data", help="Root folder containing images")
    parser.add_argument("--out-dir", default="litter_yolo_dataset", help="Destination YOLO dataset directory")
    parser.add_argument("--class-mode", choices=["single", "supercategory", "all"], default="single",
                        help="single: all trash->'litter'; supercategory: 28 superclasses; all: 60 original classes")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking")
    args = parser.parse_args()

    convert_coco_to_yolo(
        coco_ann_path=args.coco_ann,
        images_root=args.images_root,
        out_dir=args.out_dir,
        val_split=args.val_split,
        class_mode=args.class_mode,
        copy_files=args.copy,
    )

