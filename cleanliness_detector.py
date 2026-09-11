#!/usr/bin/env python3
"""
Full Cleanliness Monitoring & Abandoned-Object Pipeline for Deployment.

Combines:
1. Fine-tuned Litter Detection (YOLOv8 trained on TACO)
2. Person Detection (COCO stock YOLOv8 for owner proximity check)
3. Centroid Tracking for abandoned/stationary litter tracking
4. Classical CV heuristics for wet spills and dirt
5. Configurable area cleanliness scoring (lobby, restroom, corridor, warehouse)
6. Visualization HUD renderer for live feeds / recordings
"""

import os
import cv2
import time
import math
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


@dataclass
class TrackedObject:
    cls: str
    center: Tuple[float, float]
    first_seen: float
    last_seen: float
    still_since: float
    box: Tuple[float, float, float, float]
    flagged: bool = False


class CentroidTracker:
    """Matches new detections to existing tracks by nearest centroid."""

    def __init__(self, match_dist_px: float = 50.0, max_missed_seconds: float = 5.0):
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_id = 0
        self.match_dist_px = match_dist_px
        self.max_missed_seconds = max_missed_seconds

    def update(self, detections: List[dict], timestamp: float, move_tolerance_px: float = 15.0) -> Dict[int, TrackedObject]:
        unmatched = list(range(len(detections)))
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = None, self.match_dist_px
            for j in unmatched:
                d = math.dist(tr.center, detections[j]["center"])
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                det = detections[best_j]
                moved = math.dist(tr.center, det["center"]) > move_tolerance_px
                tr.center = det["center"]
                tr.box = det["box"]
                tr.last_seen = timestamp
                if moved:
                    tr.still_since = timestamp  # reset stationary timer if moved
                unmatched.remove(best_j)
            elif timestamp - tr.last_seen > self.max_missed_seconds:
                del self.tracks[tid]  # object removed / cleaned up

        for j in unmatched:
            det = detections[j]
            self.tracks[self._next_id] = TrackedObject(
                cls=det["cls"],
                center=det["center"],
                first_seen=timestamp,
                last_seen=timestamp,
                still_since=timestamp,
                box=det["box"],
            )
            self._next_id += 1

        return self.tracks


def flag_abandoned_objects(
    tracks: Dict[int, TrackedObject],
    persons: List[dict],
    timestamp: float,
    stationary_seconds: float = 15.0,
    owner_radius_px: float = 150.0,
) -> List[int]:
    """Flag objects that are stationary with no person nearby."""
    flagged_ids = []
    for tid, tr in tracks.items():
        stationary_for = timestamp - tr.still_since
        if stationary_for < stationary_seconds:
            continue
        has_nearby_person = any(math.dist(tr.center, p["center"]) < owner_radius_px for p in persons)
        if not has_nearby_person:
            tr.flagged = True
            flagged_ids.append(tid)
    return flagged_ids


def exclude_boxes_from_mask(mask: np.ndarray, boxes: List[Tuple[float, float, float, float]], pad: int = 12) -> np.ndarray:
    """Zero out (dilated) detection boxes in a mask so the spill/dirt heuristics never analyze
    a person's clothing, bag texture, etc. as if it were floor. This is the single biggest
    source of false positives when those heuristics are run on the raw frame."""
    out = mask.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        out[y1:y2, x1:x2] = 0
    return out


class BaselineFloorModel:
    """Optional but recommended: calibrate against a short clip of the EMPTY floor first, then
    detect spills/dirt as deviations from that specific floor's own baseline brightness/texture
    instead of fixed absolute thresholds. This removes the vast majority of false positives that
    come from a patterned or naturally uneven floor (tiles, grout lines, rugs, wood grain), since
    those patterns are baked into the baseline and no longer read as "anomalies."""

    def __init__(self):
        self.baseline_v: Optional[np.ndarray] = None   # mean HSV-Value of the empty floor
        self.baseline_tex: Optional[np.ndarray] = None  # mean local texture (Laplacian) of the empty floor

    def calibrate(self, frames: List[np.ndarray]):
        vs, texs = [], []
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            vs.append(cv2.blur(hsv[:, :, 2], (41, 41)).astype(np.float32))
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lap = np.uint8(np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 0, 255))
            texs.append(cv2.blur(lap, (25, 25)).astype(np.float32))
        self.baseline_v = np.mean(vs, axis=0)
        self.baseline_tex = np.mean(texs, axis=0)

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_v is not None

    def save(self, path: str):
        np.savez(path, v=self.baseline_v, tex=self.baseline_tex)

    def load(self, path: str):
        data = np.load(path)
        self.baseline_v, self.baseline_tex = data["v"], data["tex"]


def detect_spill_heuristic(frame_bgr: np.ndarray, floor_mask: Optional[np.ndarray] = None, min_area: int = 150,
                            baseline: Optional["BaselineFloorModel"] = None) -> Dict:
    """Detect probable wet spots via specular highlights."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    if floor_mask is None:
        floor_mask = np.ones(v.shape, dtype=np.uint8) * 255

    v_floor = v[floor_mask > 0]
    if v_floor.size == 0:
        return {"spill_score": 0.0, "regions": []}

    if baseline is not None and baseline.is_calibrated:
        # Compare against THIS floor's own calibrated baseline brightness instead of a fixed
        # local-blur estimate — much less sensitive to naturally patterned/uneven floors.
        local_mean = baseline.baseline_v.astype(np.uint8)
        threshold = 25  # a calibrated baseline can use a tighter, more confident threshold
    else:
        local_mean = cv2.blur(v, (41, 41))
        threshold = 35
    brightness_delta = cv2.subtract(v, local_mean)

    highlight_mask = ((brightness_delta > threshold) & (s < 60)).astype(np.uint8) * 255
    highlight_mask = cv2.bitwise_and(highlight_mask, highlight_mask, mask=floor_mask)
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]

    floor_area = max(int((floor_mask > 0).sum()), 1)
    highlight_area = sum(w * h for (_, _, w, h) in regions)
    spill_score = min(1.0, highlight_area / (0.02 * floor_area))
    return {"spill_score": spill_score, "regions": regions}


def detect_dirt_heuristic(frame_bgr: np.ndarray, floor_mask: Optional[np.ndarray] = None, min_area: int = 200,
                           baseline: Optional["BaselineFloorModel"] = None) -> Dict:
    """Detect probable dirt/stains via local texture-variance anomalies."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if floor_mask is None:
        floor_mask = np.ones(gray.shape, dtype=np.uint8) * 255

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))
    local_texture = cv2.blur(lap_abs, (25, 25))

    floor_texture_vals = local_texture[floor_mask > 0]
    if floor_texture_vals.size == 0:
        return {"dirt_score": 0.0, "regions": []}

    if baseline is not None and baseline.is_calibrated:
        # Per-pixel comparison against this floor's own calibrated texture baseline, rather than
        # a single global median — a patterned floor's normal texture no longer reads as "dirt."
        ref_texture = baseline.baseline_tex
        anomaly_mask = ((local_texture.astype(np.float32) - ref_texture) > 20).astype(np.uint8) * 255
    else:
        baseline_val = np.median(floor_texture_vals)
        anomaly_mask = ((local_texture.astype(np.int16) - baseline_val) > 25).astype(np.uint8) * 255
    anomaly_mask = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=floor_mask)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]

    floor_area = max(int((floor_mask > 0).sum()), 1)
    anomaly_area = sum(w * h for (_, _, w, h) in regions)
    dirt_score = min(1.0, anomaly_area / (0.05 * floor_area))
    return {"dirt_score": dirt_score, "regions": regions}


@dataclass
class AreaCleanlinessConfig:
    area_type: str
    max_litter_items: int = 2
    max_abandoned_objects: int = 0
    spill_score_threshold: float = 0.15
    dirt_score_threshold: float = 0.25
    weight_litter: float = 0.35
    weight_spill: float = 0.35
    weight_dirt: float = 0.30


AREA_CONFIGS: Dict[str, AreaCleanlinessConfig] = {
    "lobby": AreaCleanlinessConfig("lobby", max_litter_items=1, spill_score_threshold=0.10, dirt_score_threshold=0.15),
    "restroom": AreaCleanlinessConfig("restroom", max_litter_items=0, spill_score_threshold=0.05, dirt_score_threshold=0.10),
    "corridor": AreaCleanlinessConfig("corridor", max_litter_items=2, spill_score_threshold=0.15, dirt_score_threshold=0.25),
    "warehouse": AreaCleanlinessConfig("warehouse", max_litter_items=5, spill_score_threshold=0.25, dirt_score_threshold=0.40),
}


def score_cleanliness(area_type: str, litter_count: int, abandoned_count: int, spill_score: float, dirt_score: float) -> Dict:
    cfg = AREA_CONFIGS.get(area_type, AREA_CONFIGS["corridor"])
    litter_penalty = min(1.0, litter_count / max(cfg.max_litter_items + 1, 1))
    spill_penalty = min(1.0, spill_score / cfg.spill_score_threshold) if cfg.spill_score_threshold else spill_score
    dirt_penalty = min(1.0, dirt_score / cfg.dirt_score_threshold) if cfg.dirt_score_threshold else dirt_score

    combined_penalty = (
        cfg.weight_litter * litter_penalty +
        cfg.weight_spill * spill_penalty +
        cfg.weight_dirt * dirt_penalty
    )
    clean_score = round(max(0.0, 1.0 - combined_penalty) * 100, 1)

    fails = (
        litter_count > cfg.max_litter_items or
        abandoned_count > cfg.max_abandoned_objects or
        spill_score > cfg.spill_score_threshold or
        dirt_score > cfg.dirt_score_threshold
    )

    return {
        "area_type": area_type,
        "clean_score": clean_score,
        "status": "FAIL" if fails else "PASS",
        "details": {
            "litter_count": litter_count,
            "abandoned_count": abandoned_count,
            "spill_score": round(spill_score, 3),
            "dirt_score": round(dirt_score, 3),
        },
    }


class CleanlinessMonitor:
    """Production cleanliness monitor with dual-model architecture."""

    def __init__(
        self,
        litter_model_path: str = "models/litter_detector_best.pt",
        person_model_path: str = "yolov8n.pt",
        area_type: str = "corridor",
        floor_mask: Optional[np.ndarray] = None,
        conf_thresh: float = 0.35,
        baseline: Optional["BaselineFloorModel"] = None,
    ):
        self.area_type = area_type
        self.floor_mask = floor_mask
        self.conf_thresh = conf_thresh
        self.tracker = CentroidTracker()
        # Calibrate this against a clip of the EMPTY floor (see calibrate_baseline_from_video
        # below) to drastically cut false positives from a patterned/uneven floor surface.
        self.baseline = baseline

        # Load litter model (custom trained) or fallback to stock if custom not yet trained
        if Path(litter_model_path).exists():
            print(f"Loading custom litter model: {litter_model_path}")
            self.litter_model = YOLO(litter_model_path)
            self.is_custom_model = True
        else:
            print(f"Custom model '{litter_model_path}' not found. Falling back to stock {person_model_path}")
            self.litter_model = YOLO(person_model_path)
            self.is_custom_model = False

        # Person detector (stock COCO YOLO detects person at class 0)
        self.person_model = YOLO(person_model_path)

    def detect_litter(self, frame: np.ndarray) -> List[dict]:
        results = self.litter_model.predict(frame, conf=self.conf_thresh, verbose=False)[0]
        names = results.names
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, str(cls_id))

            # If using stock COCO model before custom training:
            if not self.is_custom_model:
                stock_litter = {"bottle", "cup", "backpack", "handbag", "suitcase", "cell phone", "banana", "book"}
                if cls_name not in stock_litter:
                    continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "cls": cls_name,
                "conf": float(box.conf.item()),
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
        return dets

    def detect_persons(self, frame: np.ndarray) -> List[dict]:
        # COCO class 0 is 'person'
        results = self.person_model.predict(frame, classes=[0], conf=0.40, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "cls": "person",
                "conf": float(box.conf.item()),
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
        return dets

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Tuple[Dict, np.ndarray]:
        ts = timestamp if timestamp is not None else time.time()

        litter_dets = self.detect_litter(frame)
        person_dets = self.detect_persons(frame)

        tracks = self.tracker.update(litter_dets, ts)
        abandoned_ids = flag_abandoned_objects(tracks, person_dets, ts, stationary_seconds=15.0)

        # Critical fix: never let the spill/dirt heuristics analyze pixels that belong to a
        # detected person or object — clothing texture and bag patterns are not floor dirt.
        base_mask = self.floor_mask if self.floor_mask is not None else np.ones(frame.shape[:2], dtype=np.uint8) * 255
        foreground_boxes = [d["box"] for d in litter_dets] + [d["box"] for d in person_dets]
        effective_mask = exclude_boxes_from_mask(base_mask, foreground_boxes)

        spill_res = detect_spill_heuristic(frame, effective_mask, baseline=self.baseline)
        dirt_res = detect_dirt_heuristic(frame, effective_mask, baseline=self.baseline)

        report = score_cleanliness(
            area_type=self.area_type,
            litter_count=len(litter_dets),
            abandoned_count=len(abandoned_ids),
            spill_score=spill_res["spill_score"],
            dirt_score=dirt_res["dirt_score"],
        )
        report["timestamp"] = ts
        report["regions"] = {"spill": spill_res["regions"], "dirt": dirt_res["regions"]}

        # Render visual annotations
        annotated_frame = self.render_overlay(frame, litter_dets, person_dets, tracks, abandoned_ids, spill_res, dirt_res, report)
        return report, annotated_frame

    def render_overlay(
        self,
        frame: np.ndarray,
        litter_dets: List[dict],
        person_dets: List[dict],
        tracks: Dict[int, TrackedObject],
        abandoned_ids: List[int],
        spill_res: dict,
        dirt_res: dict,
        report: dict,
    ) -> np.ndarray:
        vis = frame.copy()

        # 1. Draw spill regions (Cyan)
        for (x, y, w, h) in spill_res["regions"]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(vis, "Spill", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 2. Draw dirt regions (Orange/Brown)
        for (x, y, w, h) in dirt_res["regions"]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(vis, "Dirt/Stain", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        # 3. Draw persons (Blue)
        for p in person_dets:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.putText(vis, f"Person {p['conf']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)

        # 4. Draw litter items & abandoned status
        for tid, tr in tracks.items():
            x1, y1, x2, y2 = map(int, tr.box)
            is_abandoned = tid in abandoned_ids
            color = (0, 0, 255) if is_abandoned else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {tr.cls}" + (" [ABANDONED]" if is_abandoned else "")
            cv2.putText(vis, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Top HUD
        status = report["status"]
        score = report["clean_score"]
        hud_color = (0, 180, 0) if status == "PASS" else (0, 0, 220)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (30, 30, 30), -1)
        hud_text = f"Area: {report['area_type'].upper()} | Cleanliness: {score}/100 | Status: {status} | Litter: {len(litter_dets)} | Spill: {report['details']['spill_score']:.2f}"
        cv2.putText(vis, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

        return vis


def calibrate_baseline_from_video(source, n_frames: int = 30) -> BaselineFloorModel:
    """Point this at a short clip (or the first N seconds of a live feed) of the EMPTY floor —
    no people, no litter, no spills — to build a per-scene baseline. Strongly recommended: this
    is what lets the heuristics tell 'this floor's normal pattern' apart from 'something changed.'"""
    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from '{source}' for baseline calibration.")
    model = BaselineFloorModel()
    model.calibrate(frames)
    print(f"Baseline calibrated from {len(frames)} empty-floor frames.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanliness Monitoring Deployment Pipeline")
    parser.add_argument("--litter-model", default="models/litter_detector_best.pt", help="Path to fine-tuned litter weights")
    parser.add_argument("--person-model", default="yolov8n.pt", help="Path to person detector")
    parser.add_argument("--area", default="corridor", choices=["lobby", "restroom", "corridor", "warehouse"])
    parser.add_argument("--source", default="0", help="Video file or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Litter detection confidence threshold")
    parser.add_argument("--calibrate-from", default=None,
                         help="Path to a short clip/image sequence of the EMPTY floor, used to build a "
                              "per-scene baseline for the spill/dirt heuristics (strongly recommended)")
    args = parser.parse_args()

    baseline = None
    if args.calibrate_from:
        baseline = calibrate_baseline_from_video(args.calibrate_from)

    monitor = CleanlinessMonitor(
        litter_model_path=args.litter_model,
        person_model_path=args.person_model,
        area_type=args.area,
        conf_thresh=args.conf,
        baseline=baseline,
    )

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    print(f"Monitoring started for area '{args.area}' on source '{args.source}'. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        report, vis_frame = monitor.process_frame(frame)
        cv2.imshow("Cleanliness Monitor", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()