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


TRACKER_PRESETS = {
    "Car": {
        "gate_radius": 5.0,
        "cluster_radius": 1.6,
        "init_points": 64,
        "min_points": 64,
        "box_height_filter_ratio": 0.15,
    },
    "Van": {
        "gate_radius": 5.0,
        "cluster_radius": 1.6,
        "init_points": 64,
        "min_points": 64,
        "box_height_filter_ratio": 0.15,
    },
    "Cyclist": {
        "gate_radius": 2.5,
        "cluster_radius": 1.0,
        "init_points": 24,
        "min_points": 24,
        "box_height_filter_ratio": 0.10,
    },
    "Pedestrian": {
        "gate_radius": 1.5,
        "cluster_radius": 0.7,
        "init_points": 16,
        "min_points": 16,
        "box_height_filter_ratio": 0.10,
    },
}


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


def list_tracks_for_class(records, target_class: str) -> list[dict]:
    frame0 = [r for r in records if r["frame"] == 0]
    target_class = canonical_label(target_class)
    return [record for record in frame0 if canonical_label(record["type"]) == target_class]


def resolve_track_id(
    records,
    track_id: int | None,
    target_class: str | None,
) -> int:
    if track_id is not None:
        return track_id

    if target_class is not None:
        matches = list_tracks_for_class(records, target_class)
        print(f"Tracks for class {canonical_label(target_class)} in frame 0:")
        for record in matches:
            print(
                f"  track_id={record['track_id']} type={record['type']} "
                f"xyz=({record['x']:.2f}, {record['y']:.2f}, {record['z']:.2f})"
            )
        if not matches:
            raise RuntimeError(f"No frame-0 tracks found for class {target_class}")
        return matches[0]["track_id"]

    track_id = choose_track_id(records)
    if track_id is None:
        raise RuntimeError("No default frame-0 Car/Van track found")
    return track_id


def resolve_tracker_param(
    override: float | int | None,
    target_class: str,
    key: str,
):
    if override is not None:
        return override
    return TRACKER_PRESETS.get(target_class, TRACKER_PRESETS["Car"])[key]


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


def pred_color(status: str) -> np.ndarray:
    return np.array([[255, 0, 0]], dtype=np.uint8) if status == "active" else np.array([[255, 165, 0]], dtype=np.uint8)


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


def match_update_box(
    source: str,
    frame_id: int,
    gt_record,
    target_class: str,
    calib: Calibration,
    detections_root: Path,
    score_thresh: float,
    pred_centroid: np.ndarray,
    match_radius: float,
) -> tuple[np.ndarray | None, str | None]:
    if source == "gt":
        if gt_record is None:
            return None, None
        return record_to_lidar_box(gt_record, calib), canonical_label(gt_record["type"])

    boxes, scores, labels = load_detection_boxes(
        detections_root / f"{frame_id:06d}.npz",
        score_thresh,
    )
    if len(boxes) == 0:
        return None, None

    keep = np.array([label_to_name(label) == canonical_label(target_class) for label in labels], dtype=bool)
    if not keep.any():
        return None, None
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    distances = np.linalg.norm(boxes[:, :3] - pred_centroid[None], axis=1)
    best = int(np.argmin(distances))
    if distances[best] > match_radius:
        return None, None
    return boxes[best], f"{label_to_name(labels[best])} {scores[best]:.2f}"


def box_quaternion(box: np.ndarray) -> np.ndarray:
    half = box[6] * 0.5
    return np.array([[0.0, 0.0, np.sin(half), np.cos(half)]], dtype=np.float32)


def empty_boxes() -> rr.Boxes3D:
    return rr.Boxes3D(
        centers=np.zeros((0, 3), dtype=np.float32),
        sizes=np.zeros((0, 3), dtype=np.float32),
        quaternions=np.zeros((0, 4), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        labels=[],
        show_labels=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument(
        "--init-source",
        choices=["gt", "pointpillar", "pointrcnn"],
        default="pointrcnn",
        help="Source of the initialization box on the first frame.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Track id to follow from the KITTI labels.",
    )
    parser.add_argument(
        "--target-class",
        choices=["Car", "Van", "Pedestrian", "Cyclist"],
        default=None,
        help="Choose the first frame-0 track of this class and print all matching frame-0 tracks.",
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
        "--detection-overlap-thresh",
        type=float,
        default=0.3,
        help="Use a matched detection box only if it overlaps enough with the tracked points.",
    )
    parser.add_argument(
        "--gate-radius",
        type=float,
        default=None,
        help="Override the xy motion gate radius. Defaults to a class-aware preset.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=None,
        help="Override the target clustering radius. Defaults to a class-aware preset.",
    )
    parser.add_argument(
        "--init-points",
        type=int,
        default=None,
        help="Override minimum support points during initialization. Defaults to a class-aware preset.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="Override minimum support points during tracking. Defaults to a class-aware preset.",
    )
    parser.add_argument(
        "--box-height-filter-ratio",
        type=float,
        default=None,
        help="Ignore the bottom ratio of support masks. Defaults to a class-aware preset.",
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
    track_id = resolve_track_id(records, track_id=args.track_id, target_class=args.target_class)
    gt = gt_by_frame(records, track_id)
    if not gt:
        raise RuntimeError(f"Track id {track_id} not found in {label_path}")
    annotated_frame_ids = sorted(gt.keys())
    first_frame = annotated_frame_ids[0]
    target_class = canonical_label(gt[first_frame]["type"])
    sequence_frame_ids = sorted(int(path.stem) for path in velodyne_dir.glob("*.bin") if int(path.stem) >= first_frame)
    if args.max_frames is not None:
        sequence_frame_ids = sequence_frame_ids[: args.max_frames]

    gate_radius = resolve_tracker_param(args.gate_radius, target_class, "gate_radius")
    cluster_radius = resolve_tracker_param(args.cluster_radius, target_class, "cluster_radius")
    init_points = resolve_tracker_param(args.init_points, target_class, "init_points")
    min_points = resolve_tracker_param(args.min_points, target_class, "min_points")
    box_height_filter_ratio = resolve_tracker_param(
        args.box_height_filter_ratio,
        target_class,
        "box_height_filter_ratio",
    )

    print(
        f"Tracking track_id={track_id} class={target_class} "
        f"gate_radius={gate_radius} cluster_radius={cluster_radius} "
        f"init_points={init_points} min_points={min_points} "
        f"box_height_filter_ratio={box_height_filter_ratio}"
    )

    detections_root = Path(args.detections_root) if args.detections_root else REPO_ROOT / "data" / "detections" / args.init_source / "npz"

    tracker = UtoniaTracker(
        mode=args.tracker_mode,
        init_radius=1.6,
        cluster_radius=cluster_radius,
        gate_radius=gate_radius,
        local_crop_radius=args.local_crop_radius,
        local_crop_min_points=args.local_crop_min_points,
        init_points=init_points,
        min_points=min_points,
        detection_overlap_threshold=args.detection_overlap_thresh,
        box_height_filter_ratio=box_height_filter_ratio,
    )
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
        rr.Points3D(init_state["centroid"][None], colors=pred_color(init_state["status"]), labels=[init_state["status"]]),
    )
    rr.log("track/used_box", empty_boxes())
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

    for frame_id in sequence_frame_ids[1:]:
        coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
        pred_centroid = tracker.predict_position().detach().cpu().numpy()
        gt_record = gt.get(frame_id)
        matched_box, matched_label = match_update_box(
            source=args.init_source,
            frame_id=frame_id,
            gt_record=gt_record,
            target_class=target_class,
            calib=calib,
            detections_root=detections_root / seq,
            score_thresh=args.score_thresh,
            pred_centroid=pred_centroid,
            match_radius=tracker.gate_radius,
        )
        state = tracker.step(coord, detection_box=matched_box)
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
            rr.Points3D(state["centroid"][None], colors=pred_color(state["status"]), labels=[state["status"]]),
        )
        if state["used_box"] is None:
            rr.log("track/used_box", empty_boxes())
        else:
            rr.log(
                "track/used_box",
                rr.Boxes3D(
                    centers=state["used_box"][None, :3],
                    sizes=state["used_box"][None, 3:6],
                    quaternions=box_quaternion(state["used_box"]),
                    colors=np.array([[255, 255, 0]], dtype=np.uint8),
                    labels=[f"used {args.init_source}: {matched_label}"],
                    show_labels=True,
                ),
            )
        gt_box = None if gt_record is None else record_to_lidar_box(gt_record, calib)
        rr.log(
            "track/gt_box",
            empty_boxes() if gt_box is None else rr.Boxes3D(
                centers=gt_box[None, :3],
                sizes=gt_box[None, 3:6],
                quaternions=box_quaternion(gt_box),
                colors=np.array([[0, 255, 255]], dtype=np.uint8),
                labels=[canonical_label(gt_record["type"])],
                show_labels=True,
            ),
        )


if __name__ == "__main__":
    main()
