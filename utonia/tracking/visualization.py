from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr


CLASS_COLORS = {
    "Car": np.array([0, 255, 0], dtype=np.uint8),
    "Pedestrian": np.array([255, 255, 0], dtype=np.uint8),
    "Cyclist": np.array([0, 255, 255], dtype=np.uint8),
    "Van": np.array([255, 128, 0], dtype=np.uint8),
}


def load_xyz(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def label_color(label: str) -> np.ndarray:
    return CLASS_COLORS.get(label, np.array([255, 255, 255], dtype=np.uint8))


def track_color(track) -> np.ndarray:
    if getattr(track, "state", "detected") == "recovered":
        return np.array([255, 0, 255], dtype=np.uint8)
    return label_color(track.label)


def track_label(track) -> str:
    suffix = " [R]" if getattr(track, "state", "detected") == "recovered" else ""
    return f"T{track.track_id} {track.label}{suffix} h={track.hits} m={track.missed}"


def boxes3d(boxes, labels, colors):
    if not boxes:
        return None

    return rr.Boxes3D(
        centers=np.stack([box.center for box in boxes], axis=0),
        sizes=np.stack([box.size for box in boxes], axis=0),
        rotations=[
            rr.RotationAxisAngle(axis=[0.0, 0.0, 1.0], radians=box.yaw)
            for box in boxes
        ],
        colors=colors,
        labels=labels,
        show_labels=True,
    )


def detection_boxes(frame):
    detections = frame.detections
    return boxes3d(
        boxes=[detection.box for detection in detections],
        labels=[
            (
                f"{detection.label}:GT{detection.metadata.get('track_id', '?')}"
                if "track_id" in detection.metadata
                else f"{detection.label}:{detection.score:.2f}"
            )
            for detection in detections
        ],
        colors=np.stack([label_color(detection.label) for detection in detections], axis=0)
        if detections
        else np.zeros((0, 3), dtype=np.uint8),
    )


def track_boxes(tracks):
    return boxes3d(
        boxes=[track.box for track in tracks],
        labels=[track_label(track) for track in tracks],
        colors=np.stack([track_color(track) for track in tracks], axis=0)
        if tracks
        else np.zeros((0, 3), dtype=np.uint8),
    )
