import argparse
import sys
from pathlib import Path

from matplotlib import cm
import numpy as np
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from utonia import UtoniaTracker
from pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402


def load_xyz(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def parse_labels(path: Path):
    records = []
    for line in path.read_text().splitlines():
        parts = line.split()
        records.append(
            {
                "frame": int(parts[0]),
                "track_id": int(parts[1]),
                "type": parts[2],
                "h": float(parts[10]),
                "w": float(parts[11]),
                "l": float(parts[12]),
                "x": float(parts[13]),
                "y": float(parts[14]),
                "z": float(parts[15]),
                "rotation_y": float(parts[16]),
            }
        )
    return records


def choose_track_id(records):
    frame0 = [r for r in records if r["frame"] == 0]
    for cls in ("Car", "Van"):
        for record in frame0:
            if record["type"] == cls:
                return record["track_id"]


def canonical_label(name: str) -> str:
    return "Car" if name == "Van" else name


def plasma(sim):
    sim = ((sim + 1.0) * 0.5).clip(0.0, 1.0)
    return (cm.plasma(sim)[:, :3] * 255).astype(np.uint8)


def point_colors(full_coord: np.ndarray, state) -> np.ndarray:
    colors = np.full((len(full_coord), 3), 140, dtype=np.uint8)
    crop_indices = np.asarray(state["crop_indices"], dtype=np.int64)
    colors[crop_indices] = plasma(state["sim"])
    return colors


def gt_by_frame(records, track_id):
    return {r["frame"]: r for r in records if r["track_id"] == track_id}


def record_to_camera_box(record) -> np.ndarray:
    return np.array(
        [[record["x"], record["y"], record["z"], record["l"], record["h"], record["w"], record["rotation_y"]]],
        dtype=np.float32,
    )


def record_to_lidar_box(record, calib: Calibration) -> np.ndarray:
    return boxes3d_kitti_camera_to_lidar(record_to_camera_box(record), calib)[0]


def load_detection_boxes(path: Path, score_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def label_to_name(label: int) -> str:
    return {1: "Car", 2: "Pedestrian", 3: "Cyclist"}.get(int(label), str(int(label)))


def select_init_box(
    source: str,
    first_frame: int,
    gt_record,
    calib: Calibration,
    detections_root: Path,
    score_thresh: float,
) -> tuple[np.ndarray, str]:
    gt_box = record_to_lidar_box(gt_record, calib)
    if source == "gt":
        return gt_box, canonical_label(gt_record["type"])

    boxes, scores, labels = load_detection_boxes(
        detections_root / f"{first_frame:06d}.npz",
        score_thresh,
    )
    if len(boxes) == 0:
        raise RuntimeError(f"No detections found in {detections_root / f'{first_frame:06d}.npz'}")

    gt_name = canonical_label(gt_record["type"])
    keep = np.array([label_to_name(label) == gt_name for label in labels], dtype=bool)
    if keep.any():
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    distances = np.linalg.norm(boxes[:, :3] - gt_box[None, :3], axis=1)
    best = int(np.argmin(distances))
    return boxes[best], f"{label_to_name(labels[best])} {scores[best]:.2f}"


def box_quaternion(box: np.ndarray) -> np.ndarray:
    half = box[6] * 0.5
    return np.array([[0.0, 0.0, np.sin(half), np.cos(half)]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument(
        "--init-source",
        choices=["gt", "pointpillar", "pointrcnn"],
        default="gt",
        help="Source of the initialization box on the first frame.",
    )
    parser.add_argument(
        "--detections-root",
        default=None,
        help="Optional detection root ending at <model>/npz. Defaults to data/detections/<init-source>/npz.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Detection score threshold used when init-source is a detector.",
    )
    parser.add_argument(
        "--tracker-mode",
        choices=["full", "local_crop"],
        default="local_crop",
        help="How much of the frame Utonia encodes on each step.",
    )
    parser.add_argument(
        "--local-crop-radius",
        type=float,
        default=8.0,
        help="Crop radius around the current track when tracker-mode is local_crop.",
    )
    parser.add_argument(
        "--local-crop-min-points",
        type=int,
        default=2048,
        help="Minimum points kept in the local crop before falling back to nearest points.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick debugging.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the rerun viewer automatically.",
    )
    args = parser.parse_args()

    velodyne_dir = Path(args.sequence_dir)
    seq = velodyne_dir.name
    label_path = velodyne_dir.parents[1] / "label_02" / f"{seq}.txt"
    calib_path = velodyne_dir.parents[1] / "calib" / f"{seq}.txt"
    calib = Calibration(calib_path)

    records = parse_labels(label_path)
    track_id = choose_track_id(records)
    gt = gt_by_frame(records, track_id)
    frame_ids = sorted(gt.keys())
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    detections_root = Path(args.detections_root) if args.detections_root else REPO_ROOT / "data" / "detections" / args.init_source / "npz"

    tracker = UtoniaTracker(
        mode=args.tracker_mode,
        init_radius=1.6,
        cluster_radius=1.6,
        gate_radius=5.0,
        local_crop_radius=args.local_crop_radius,
        local_crop_min_points=args.local_crop_min_points,
    )
    first_frame = frame_ids[0]
    first_coord = load_xyz(velodyne_dir / f"{first_frame:06d}.bin")
    init_box, init_label = select_init_box(
        source=args.init_source,
        first_frame=first_frame,
        gt_record=gt[first_frame],
        calib=calib,
        detections_root=detections_root / seq,
        score_thresh=args.score_thresh,
    )
    init_state = tracker.initialize_from_box(first_coord, init_box)
    gt_box = record_to_lidar_box(gt[first_frame], calib)

    rr.init("utonia_kitti_tracker", spawn=not args.no_spawn)
    rr.set_time("frame", sequence=first_frame)
    rr.log(
        "points",
        rr.Points3D(first_coord, colors=point_colors(first_coord, init_state)),
    )
    rr.log(
        "track/object",
        rr.Points3D(
            init_state["coord"][init_state["mask"]],
            colors=np.tile([[0, 255, 0]], (int(init_state["mask"].sum()), 1)),
        ),
    )
    rr.log(
        "track/pred",
        rr.Points3D(init_state["centroid"][None], colors=np.array([[255, 0, 0]], dtype=np.uint8)),
    )
    rr.log(
        "track/init_box",
        rr.Boxes3D(
            centers=init_box[None, :3],
            sizes=init_box[None, 3:6],
            quaternions=box_quaternion(init_box),
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            labels=[f"init {args.init_source}: {init_label}"],
            show_labels=True,
        ),
    )
    rr.log(
        "track/gt_box",
        rr.Boxes3D(
            centers=gt_box[None, :3],
            sizes=gt_box[None, 3:6],
            quaternions=box_quaternion(gt_box),
            colors=np.array([[0, 255, 255]], dtype=np.uint8),
            labels=[canonical_label(gt[first_frame]["type"])],
            show_labels=True,
        ),
    )

    for frame_id in frame_ids[1:]:
        coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
        state = tracker.step(coord)
        gt_box = record_to_lidar_box(gt[frame_id], calib)
        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(coord, colors=point_colors(coord, state)))
        rr.log(
            "track/object",
            rr.Points3D(
                state["coord"][state["mask"]],
                colors=np.tile([[0, 255, 0]], (int(state["mask"].sum()), 1)),
            ),
        )
        rr.log(
            "track/pred",
            rr.Points3D(state["centroid"][None], colors=np.array([[255, 0, 0]], dtype=np.uint8)),
        )
        rr.log(
            "track/gt_box",
            rr.Boxes3D(
                centers=gt_box[None, :3],
                sizes=gt_box[None, 3:6],
                quaternions=box_quaternion(gt_box),
                colors=np.array([[0, 255, 255]], dtype=np.uint8),
                labels=[canonical_label(gt[frame_id]["type"])],
                show_labels=True,
            ),
        )


if __name__ == "__main__":
    main()
