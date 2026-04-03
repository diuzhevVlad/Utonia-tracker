import argparse
from pathlib import Path
import sys

import numpy as np
import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.adapters.kitti_tracking import KittiGtDetectionSource
from utonia.tracking.mot import MOTracker, UtoniaMOTracker


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
            f"{detection.label}:GT{detection.metadata.get('track_id', '?')}"
            for detection in detections
        ],
        colors=np.stack(
            [
                CLASS_COLORS.get(
                    detection.label,
                    np.array([255, 255, 255], dtype=np.uint8),
                )
                for detection in detections
            ],
            axis=0,
        )
        if detections
        else np.zeros((0, 3), dtype=np.uint8),
    )


def track_boxes(tracks):
    return boxes3d(
        boxes=[track.box for track in tracks],
        labels=[
            f"T{track.track_id} {track.label} h={track.hits} m={track.missed}"
            for track in tracks
        ],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-match-distance", type=float, default=5.0)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use the basic motion-only tracker instead of Utonia matching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = KittiGtDetectionSource(args.sequence_dir)
    basic_tracker = None
    utonia_tracker = None
    if args.basic:
        basic_tracker = MOTracker(
            max_match_distance=args.max_match_distance,
            max_missed=args.max_missed,
        )
        tracker_name = "basic"
    else:
        utonia_tracker = UtoniaMOTracker(
            max_match_distance=args.max_match_distance,
            max_missed=args.max_missed,
        )
        tracker_name = "utonia"

    frame_ids = [frame_id for frame_id in source.frame_ids() if frame_id >= args.start_frame]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    rr.init("utonia_kitti_mot_demo", spawn=True)
    for frame_id in frame_ids:
        point_path = Path(args.sequence_dir) / f"{frame_id:06d}.bin"
        coord = load_xyz(point_path)
        frame = source.get_frame_detections(frame_id)
        if args.basic:
            assert basic_tracker is not None
            tracks = basic_tracker.update(frame)
        else:
            assert utonia_tracker is not None
            tracks = utonia_tracker.update(coord, frame)

        print(
            f"frame {frame_id}: {len(frame.detections)} detections, {len(tracks)} active tracks ({tracker_name})"
        )

        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(coord))

        gt_boxes = detection_boxes(frame)
        if gt_boxes is not None:
            rr.log("gt/boxes", gt_boxes)

        active_boxes = track_boxes(tracks)
        if active_boxes is not None:
            rr.log("tracks/boxes", active_boxes)


if __name__ == "__main__":
    main()
