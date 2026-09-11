#!/usr/bin/env python3
"""
Convert TACO (or custom COCO-format dataset) to YOLO format for YOLOv8 training.

Fixes:
1. Filename collision bug across batches (e.g., batch_1/000006.jpg vs batch_2/000006.jpg).
2. Strict bounding-box validation: drops zero/negative-size boxes and noise-level tiny boxes.
3. Flexible class mapping:
   - 'single': All trash items mapped to 1 class 'litter' (Recommended).
   - 'supercategory': 28 supercategories (Bottle, Plastic bag, Can, Cigarette...).
   - 'all': Original 60 TACO categories.
4. data.yaml written with absolute path so training works from any working directory.
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
        supercats = sorted(list(set(
            c["supercategory"] for c in cats if c.get("supercategory")
        )))
        supercat_to_id = {sc: i for i, sc in enumerate(supercats)}
        id_map = {
            c["id"]: supercat_to_id[c["supercategory"]]
            for c in cats if c.get("supercategory") in supercat_to_id
        }
        return id_map, supercats

    elif mode == "all":
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
    min_box_area: float = 0.0001,   # minimum normalised area (nw*nh); drops noise-level boxes
) -> str:
    """
    Converts a COCO-format dataset to YOLOv8 dataset layout.

    Improvements over a naive converter
    ------------------------------------
    * batch_1/000006.jpg  ->  batch_1_000006.jpg  (zero cross-batch filename collisions)
    * Degenerate (zero/negative-size) boxes are dropped.
    * Boxes are clamped to [0,1] in normalised coordinates.
    * Boxes whose normalised area < min_box_area are dropped (annotation noise).
    * data.yaml uses the absolute path so training works regardless of cwd.
    """
    coco_path = Path(coco_ann_path)
    if not coco_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {coco_ann_path}")

    print(f"Loading annotations from {coco_ann_path}...")
    coco = load_coco_annotations(str(coco_path))

    cat_map, class_names = build_class_mapping(coco, mode=class_mode)
    preview = class_names[:5]
    print(f"Class mode: '{class_mode}' -> {len(class_names)} class(es): {preview}{'...' if len(class_names) > 5 else ''}")

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
    dropped_boxes = 0

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

        img_w = float(info["width"])
        img_h = float(info["height"])
        label_lines = []

        for ann in anns_by_img.get(img_id, []):
            cat_id = ann.get("category_id")
            if cat_id not in cat_map:
                continue

            x, y, bw, bh = ann["bbox"]

            # Drop degenerate boxes
            if bw <= 0 or bh <= 0:
                dropped_boxes += 1
                continue

            # COCO (top-left x,y,w,h) -> YOLO centre-normalised (cx,cy,nw,nh)
            cx = (x + bw / 2.0) / img_w
            cy = (y + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Clamp to [0, 1]
            cx = min(max(0.0, cx), 1.0)
            cy = min(max(0.0, cy), 1.0)
            nw = min(max(1e-4, nw), 1.0)
            nh = min(max(1e-4, nh), 1.0)

            # Drop near-invisible boxes (annotation noise)
            if nw * nh < min_box_area:
                dropped_boxes += 1
                continue

            cls_id = cat_map[cat_id]
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        label_path = out / "labels" / split / f"{safe_stem_without_ext}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        processed_count += 1

    # Write data.yaml with absolute path for maximum reproducibility
    yaml_path = out / "data.yaml"
    clean_names = [n.replace("'", "") for n in class_names]
    names_str = ", ".join([f"'{n}'" for n in clean_names])
    yaml_content = (
        f"# YOLOv8 Dataset - Public Area Cleanliness Monitoring\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_str}]\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print("\n--- Dataset Conversion Summary ---")
    print(f"Output directory : {out.resolve()}")
    print(f"Images processed : {processed_count}  (skipped {skipped_missing} missing)")
    print(f"Boxes written    : {total_boxes}  (dropped {dropped_boxes})")
    print(f"Classes ({len(class_names)})   : {class_names}")
    print(f"data.yaml        : {yaml_path.resolve()}")
    return str(yaml_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TACO COCO dataset to YOLOv8 format.")
    parser.add_argument("--coco-ann", default="TACO/data/annotations.json")
    parser.add_argument("--images-root", default="TACO/data")
    parser.add_argument("--out-dir", default="litter_yolo_dataset")
    parser.add_argument("--class-mode", choices=["single", "supercategory", "all"], default="single")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hardlink")
    args = parser.parse_args()

    convert_coco_to_yolo(
        coco_ann_path=args.coco_ann,
        images_root=args.images_root,
        out_dir=args.out_dir,
        val_split=args.val_split,
        class_mode=args.class_mode,
        copy_files=args.copy,
    )