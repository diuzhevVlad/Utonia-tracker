import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

CLASS_NAMES = {
    1: "Car",
    2: "Pedestrian",
    3: "Cyclist",
}

CLASS_COLORS = {
    1: np.array([255, 80, 80], dtype=np.uint8),
    2: np.array([80, 255, 80], dtype=np.uint8),
    3: np.array([80, 180, 255], dtype=np.uint8),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI velodyne sequence dir, e.g. /media/.../trackkitti/training/velodyne/0000",
    )
    parser.add_argument(
        "--detections_root",
        default=str(REPO_ROOT / "data" / "detections" / "pointrcnn" / "npz"),
        help="Root directory containing <sequence>/<frame>.npz detection files.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Hide detections below this score.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame limit for quick inspection.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the rerun viewer automatically.",
    )
    return parser.parse_args()


def load_xyz(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def load_detections(path: Path, score_thresh: float):
    if not path.exists():
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    data = np.load(path)
    boxes = data["pred_boxes"].astype(np.float32)
    scores = data["pred_scores"].astype(np.float32)
    labels = data["pred_labels"].astype(np.int32)
    if score_thresh > 0:
        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    return boxes, scores, labels


def build_center_colors(labels: np.ndarray) -> np.ndarray:
    if len(labels) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.stack(
        [CLASS_COLORS.get(int(label), np.array([255, 255, 255], dtype=np.uint8)) for label in labels]
    )


def build_quaternions(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    half_heading = boxes[:, 6] * 0.5
    quaternions = np.zeros((len(boxes), 4), dtype=np.float32)
    quaternions[:, 2] = np.sin(half_heading)
    quaternions[:, 3] = np.cos(half_heading)
    return quaternions


def main():
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    sequence = sequence_dir.name
    detections_dir = Path(args.detections_root) / sequence
    frame_paths = sorted(sequence_dir.glob("*.bin"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    rr.init(f"kitti_detections_{sequence}", spawn=not args.no_spawn)

    for frame_path in frame_paths:
        frame_id = frame_path.stem
        points = load_xyz(frame_path)
        boxes, scores, labels = load_detections(
            detections_dir / f"{frame_id}.npz",
            args.score_thresh,
        )
        center_colors = build_center_colors(labels)
        quaternions = build_quaternions(boxes)
        class_labels = [
            f"{CLASS_NAMES.get(int(label), str(int(label)))} {score:.2f}"
            for label, score in zip(labels, scores, strict=False)
        ]

        rr.set_time("frame", sequence=int(frame_id))
        rr.log(
            "points",
            rr.Points3D(points, colors=np.full((len(points), 3), 140, dtype=np.uint8)),
        )
        rr.log(
            "detections/centers",
            rr.Points3D(boxes[:, :3], colors=center_colors, labels=class_labels),
        )
        rr.log(
            "detections/boxes",
            rr.Boxes3D(
                centers=boxes[:, :3],
                sizes=boxes[:, 3:6],
                quaternions=quaternions,
                colors=center_colors,
                labels=class_labels,
                show_labels=True,
            ),
        )


if __name__ == "__main__":
    main()
