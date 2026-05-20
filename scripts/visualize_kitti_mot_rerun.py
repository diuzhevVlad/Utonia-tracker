import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rerun as rr


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402


CLASS_COLORS = {
    "Car": np.array([255, 90, 70], dtype=np.uint8),
    "Pedestrian": np.array([70, 220, 120], dtype=np.uint8),
}
EVENT_COLORS = {
    "new": np.array([80, 170, 255], dtype=np.uint8),
    "match": np.array([80, 220, 120], dtype=np.uint8),
    "low_score_match": np.array([255, 190, 70], dtype=np.uint8),
    "utonia_fallback": np.array([210, 90, 255], dtype=np.uint8),
}
GT_COLOR = np.array([220, 220, 220], dtype=np.uint8)
POINT_COLOR = np.array([120, 120, 120], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with velodyne/, calib/, label_02/.",
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "data" / "mot_kitti" / "official_mot_pointpillar_score07_low03"),
        help="MOT run directory containing data/<sequence>.txt and diagnostics.csv.",
    )
    parser.add_argument("--sequence", default="0019")
    parser.add_argument("--classes", nargs="+", default=["Car", "Pedestrian"])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument(
        "--max-points",
        type=int,
        default=80000,
        help="Deterministically subsample each LiDAR frame to this many points for smaller Rerun files.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional .rrd output path. Defaults to data/rerun/<run>_<sequence>.rrd.",
    )
    parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer while logging.")
    return parser.parse_args()


def load_xyz(path: Path, max_points: int | None) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
    if max_points is not None and max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points


def camera_record_to_lidar_box(record: dict, calib: Calibration) -> np.ndarray:
    camera_box = np.array(
        [[record["x"], record["y"], record["z"], record["l"], record["h"], record["w"], record["rotation_y"]]],
        dtype=np.float32,
    )
    return boxes3d_kitti_camera_to_lidar(camera_box, calib)[0]


def parse_kitti_tracking_file(path: Path, classes: set[str], has_score: bool) -> dict[int, list[dict]]:
    by_frame = defaultdict(list)
    if not path.exists():
        return by_frame
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_name = parts[2]
        if class_name not in classes:
            continue
        record = {
            "frame": int(parts[0]),
            "track_id": int(parts[1]),
            "class": class_name,
            "h": float(parts[10]),
            "w": float(parts[11]),
            "l": float(parts[12]),
            "x": float(parts[13]),
            "y": float(parts[14]),
            "z": float(parts[15]),
            "rotation_y": float(parts[16]),
            "score": float(parts[17]) if has_score and len(parts) > 17 else None,
        }
        by_frame[record["frame"]].append(record)
    return by_frame


def parse_diagnostics(path: Path) -> dict[tuple[int, int], str]:
    events = {}
    if not path.exists():
        return events
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("event") not in EVENT_COLORS:
                continue
            try:
                frame = int(row["frame"])
                track_id = int(row["track_id"])
            except (TypeError, ValueError):
                continue
            events[(frame, track_id)] = row["event"]
    return events


def build_quaternions(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    half_heading = boxes[:, 6] * 0.5
    quaternions = np.zeros((len(boxes), 4), dtype=np.float32)
    quaternions[:, 2] = np.sin(half_heading)
    quaternions[:, 3] = np.cos(half_heading)
    return quaternions


def log_boxes(entity: str, records: list[dict], calib: Calibration, events: dict[tuple[int, int], str] | None = None) -> None:
    if not records:
        rr.log(entity, rr.Clear(recursive=True))
        return
    boxes = np.stack([camera_record_to_lidar_box(record, calib) for record in records], axis=0)
    labels = []
    colors = []
    for record in records:
        event = events.get((record["frame"], record["track_id"]), "track") if events is not None else "gt"
        score = record["score"]
        if score is None:
            labels.append(f"GT {record['class']}#{record['track_id']}")
            colors.append(GT_COLOR)
        else:
            labels.append(f"{record['class']}#{record['track_id']} {event} {score:.2f}")
            colors.append(EVENT_COLORS.get(event, CLASS_COLORS.get(record["class"], np.array([255, 255, 255], dtype=np.uint8))))
    rr.log(
        entity,
        rr.Boxes3D(
            centers=boxes[:, :3],
            sizes=boxes[:, 3:6],
            quaternions=build_quaternions(boxes),
            colors=np.stack(colors, axis=0),
            labels=labels,
            show_labels=True,
        ),
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    classes = set(args.classes)
    sequence = args.sequence
    output_path = Path(args.save) if args.save else REPO_ROOT / "data" / "rerun" / f"{run_dir.name}_{sequence}.rrd"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calib = Calibration(data_root / "calib" / f"{sequence}.txt")
    gt_by_frame = parse_kitti_tracking_file(data_root / "label_02" / f"{sequence}.txt", classes, has_score=False)
    pred_by_frame = parse_kitti_tracking_file(run_dir / "data" / f"{sequence}.txt", classes, has_score=True)
    events = parse_diagnostics(run_dir / "diagnostics.csv")

    frame_paths = sorted((data_root / "velodyne" / sequence).glob("*.bin"))
    frame_paths = [path for path in frame_paths if int(path.stem) >= args.start_frame]
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    rr.init(f"kitti_mot_{run_dir.name}_{sequence}", spawn=args.spawn)
    rr.save(str(output_path))

    for index, frame_path in enumerate(frame_paths, start=1):
        frame = int(frame_path.stem)
        points = load_xyz(frame_path, args.max_points)
        rr.set_time("frame", sequence=frame)
        rr.log("lidar/points", rr.Points3D(points, colors=np.broadcast_to(POINT_COLOR, (len(points), 3))))
        log_boxes("gt/boxes", gt_by_frame.get(frame, []), calib)
        log_boxes("prediction/boxes", pred_by_frame.get(frame, []), calib, events=events)
        if index == 1 or index % 50 == 0 or index == len(frame_paths):
            print(f"{sequence} [{index}/{len(frame_paths)}] frame={frame}", flush=True)

    print(f"Saved Rerun recording to {output_path}", flush=True)


if __name__ == "__main__":
    main()
