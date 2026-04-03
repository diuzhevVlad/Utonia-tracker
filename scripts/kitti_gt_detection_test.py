import argparse
from pathlib import Path
import sys

import numpy as np
import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.adapters.kitti_tracking import KittiGtDetectionSource
from utonia.tracking.mot import MOTracker


CLASS_COLORS = {
    "Car": np.array([0, 255, 0], dtype=np.uint8),
    "Pedestrian": np.array([255, 255, 0], dtype=np.uint8),
    "Cyclist": np.array([0, 255, 255], dtype=np.uint8),
    "Van": np.array([255, 128, 0], dtype=np.uint8),
}


def load_xyz(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def boxes3d(boxes, labels, colors):
    if not boxes:
        return None

    centers = np.stack([box.center for box in boxes], axis=0)
    sizes = np.stack([box.size for box in boxes], axis=0)
    rotations = [
        rr.RotationAxisAngle(axis=[0.0, 0.0, 1.0], radians=box.yaw)
        for box in boxes
    ]
    return rr.Boxes3D(
        centers=centers,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        labels=labels,
        show_labels=True,
    )


def detection_boxes(frame):
    return boxes3d(
        boxes=[detection.box for detection in frame.detections],
        labels=[
            f"{detection.label}:{detection.metadata.get('track_id', '?')}"
            for detection in frame.detections
        ],
        colors=np.stack(
            [
                CLASS_COLORS.get(
                    detection.label,
                    np.array([255, 255, 255], dtype=np.uint8),
                )
                for detection in frame.detections
            ],
            axis=0,
        )
        if frame.detections
        else np.zeros((0, 3), dtype=np.uint8),
    )


def track_boxes(tracks):
    return boxes3d(
        boxes=[track.box for track in tracks],
        labels=[f"{track.label}:T{track.track_id}" for track in tracks],
        colors=np.stack(
            [
                CLASS_COLORS.get(track.label, np.array([255, 0, 255], dtype=np.uint8))
                for track in tracks
            ],
            axis=0,
        )
        if tracks
        else np.zeros((0, 3), dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument("--frame", type=int, help="Optional frame id to inspect")
    parser.add_argument("--track", action="store_true", help="Run the basic box tracker")
    args = parser.parse_args()

    source = KittiGtDetectionSource(args.sequence_dir)
    frame_ids = [args.frame] if args.frame is not None else source.frame_ids()
    tracker = MOTracker() if args.track else None

    rr.init("utonia_kitti_gt_detections", spawn=True)
    for frame_id in frame_ids:
        point_path = Path(args.sequence_dir) / f"{frame_id:06d}.bin"
        coord = load_xyz(point_path)
        frame = source.get_frame_detections(frame_id)
        boxes = detection_boxes(frame)
        tracks = tracker.update(frame) if tracker is not None else []
        print(f"frame {frame_id}: {len(frame.detections)} detections")
        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(coord))
        if boxes is not None:
            rr.log("detections/boxes", boxes)
        tracked_boxes = track_boxes(tracks)
        if tracked_boxes is not None:
            rr.log("tracks/boxes", tracked_boxes)


if __name__ == "__main__":
    main()
