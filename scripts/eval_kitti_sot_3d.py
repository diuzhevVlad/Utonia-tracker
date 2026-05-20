import argparse
import csv
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.utils.box_utils import boxes_to_corners_3d  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from scripts.kitti_tracker_test import (  # noqa: E402
    TRACKER_PRESETS,
    load_xyz,
    match_update_box,
    parse_labels,
    record_to_lidar_box,
    resolve_tracker_param,
    select_init_box,
)
from utonia import UtoniaTracker  # noqa: E402


KITTI_SPLITS = {
    "train": [f"{index:04d}" for index in range(0, 17)],
    "valid": [f"{index:04d}" for index in range(17, 19)],
    "test": [f"{index:04d}" for index in range(19, 21)],
    "full": [f"{index:04d}" for index in range(21)],
    "tiny": ["0000"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with velodyne/, calib/, and label_02/.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(KITTI_SPLITS.keys()),
        default="test",
        help="KITTI sequence split matching common 3D SOT codebases.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional explicit sequence subset. Overrides --split.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Car", "Pedestrian"],
        choices=sorted(TRACKER_PRESETS.keys()),
        help="Classes to evaluate. Defaults to official KITTI tracking classes; pass Van/Cyclist explicitly for 3D SOT all-class comparisons.",
    )
    parser.add_argument(
        "--init-source",
        choices=["gt", "pointpillar", "pointrcnn"],
        default="gt",
        help="Initialization source. GT is the standard fair-comparison protocol.",
    )
    parser.add_argument(
        "--update-source",
        choices=["none", "gt", "pointpillar", "pointrcnn"],
        default="none",
        help="Per-frame external update source. Using none is the closest published-style comparison.",
    )
    parser.add_argument(
        "--init-policy",
        choices=["immediate", "stable"],
        default="immediate",
        help="Initialization policy. Immediate GT init is the standard published-style protocol.",
    )
    parser.add_argument(
        "--detections-base-root",
        default=str(REPO_ROOT / "data" / "detections"),
        help="Base directory containing <model>/npz/<sequence>/<frame>.npz.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--min-track-length", type=int, default=1)
    parser.add_argument("--report-min-track-length", type=int, default=10)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--track-list-csv", default=None)
    parser.add_argument("--stable-init-min-points", type=int, default=None)
    parser.add_argument("--stable-init-consecutive", type=int, default=2)
    parser.add_argument("--max-track-frames", type=int, default=None)
    parser.add_argument("--local-crop-radius", type=float, default=8.0)
    parser.add_argument("--local-crop-min-points", type=int, default=2048)
    parser.add_argument("--detection-overlap-thresh", type=float, default=0.3)
    parser.add_argument("--gate-radius", type=float, default=None)
    parser.add_argument("--car-gate-radius", type=float, default=None)
    parser.add_argument("--pedestrian-gate-radius", type=float, default=None)
    parser.add_argument("--cluster-radius", type=float, default=None)
    parser.add_argument("--init-points", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--box-height-filter-ratio", type=float, default=None)
    parser.add_argument("--lost-height-ratio", type=float, default=0.5)
    parser.add_argument("--lost-bad-frames", type=int, default=2)
    parser.add_argument("--pre-lost-window", type=int, default=5)
    parser.add_argument("--sparse-support-threshold", type=int, default=0)
    parser.add_argument("--sparse-sim-threshold", type=float, default=0.45)
    parser.add_argument("--sparse-cluster-radius-scale", type=float, default=1.5)
    parser.add_argument("--sparse-no-proto-update", action="store_true")
    parser.add_argument("--recovery-max-frames", type=int, default=0)
    parser.add_argument("--recovery-crop-radius-scale", type=float, default=2.0)
    parser.add_argument("--recovery-gate-radius-scale", type=float, default=2.0)
    parser.add_argument("--recovery-sim-threshold", type=float, default=0.55)
    parser.add_argument(
        "--metric-space",
        choices=["3d", "bev"],
        default="3d",
        help="Metric space used for Success/Precision summaries. Use bev for legacy behavior.",
    )
    parser.add_argument("--frame-cache-size", type=int, default=256)
    parser.add_argument(
        "--success-steps",
        type=int,
        default=21,
        help="Number of thresholds in the Success AUC curve. Open3DSOT/PTT use 21.",
    )
    parser.add_argument(
        "--precision-steps",
        type=int,
        default=21,
        help="Number of thresholds in the Precision AUC curve. Open3DSOT/PTT use 21.",
    )
    parser.add_argument(
        "--precision-max-distance",
        type=float,
        default=2.0,
        help="Maximum center distance used in Precision AUC. Open3DSOT/PTT use 2.0.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data" / "eval_sot_3d"),
        help="Directory where per-frame, per-track, and summary outputs are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name. Defaults to a descriptive auto-generated name.",
    )
    return parser.parse_args()


STABLE_INIT_PRESETS = {
    "Car": 20,
    "Van": 20,
    "Cyclist": 12,
    "Pedestrian": 10,
}


class SuccessMetric:
    def __init__(self, n: int = 21, max_overlap: float = 1.0):
        self.thresholds = np.linspace(0.0, max_overlap, n)

    def value(self, overlaps: np.ndarray) -> float:
        if overlaps.size == 0:
            return 0.0
        curve = np.array([(overlaps >= threshold).mean() for threshold in self.thresholds], dtype=np.float32)
        return float(np.trapezoid(curve, x=self.thresholds) * 100.0 / self.thresholds[-1])


class PrecisionMetric:
    def __init__(self, n: int = 21, max_distance: float = 2.0):
        self.thresholds = np.linspace(0.0, max_distance, n)
        self.max_distance = max_distance

    def value(self, distances: np.ndarray) -> float:
        if distances.size == 0:
            return 0.0
        curve = np.array([(distances <= threshold).mean() for threshold in self.thresholds], dtype=np.float32)
        return float(np.trapezoid(curve, x=self.thresholds) * 100.0 / self.max_distance)


def sequence_ids_from_args(args) -> list[str]:
    return args.sequences or KITTI_SPLITS[args.split]


def read_track_list(path: Path) -> set[tuple[str, int]]:
    with path.open() as handle:
        rows = csv.DictReader(handle)
        return {(row["sequence"], int(row["track_id"])) for row in rows}


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_float(value):
    if value is None:
        return None
    return float(value)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


FRAME_CACHE_SIZE = 256
FRAME_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()


def configure_frame_cache(maxsize: int) -> None:
    global FRAME_CACHE_SIZE
    FRAME_CACHE_SIZE = max(0, maxsize)
    FRAME_CACHE.clear()


def load_xyz_cached(path: str) -> np.ndarray:
    if FRAME_CACHE_SIZE == 0:
        return load_xyz(Path(path))
    if path in FRAME_CACHE:
        FRAME_CACHE.move_to_end(path)
        return FRAME_CACHE[path]
    coord = load_xyz(Path(path))
    FRAME_CACHE[path] = coord
    if len(FRAME_CACHE) > FRAME_CACHE_SIZE:
        FRAME_CACHE.popitem(last=False)
    return coord


def build_tracklets(data_root: Path, sequence_ids: list[str], classes: set[str], min_track_length: int):
    label_root = data_root / "label_02"
    tracklets = []
    for sequence in sequence_ids:
        records = parse_labels(label_root / f"{sequence}.txt")
        by_track = defaultdict(list)
        for record in records:
            by_track[record["track_id"]].append(record)
        for track_id, track_records in sorted(by_track.items()):
            if track_id < 0:
                continue
            track_records = sorted(track_records, key=lambda record: record["frame"])
            target_class = track_records[0]["type"]
            if target_class not in classes:
                continue
            if len(track_records) < min_track_length:
                continue
            tracklets.append(
                {
                    "sequence": sequence,
                    "track_id": track_id,
                    "target_class": target_class,
                    "records": track_records,
                    "track_length": len(track_records),
                }
            )
    return tracklets


def resolve_detection_root(base_root: Path, source: str) -> Path | None:
    if source in {"none", "gt"}:
        return None
    return base_root / source / "npz"


def box_corners_bev(box: np.ndarray) -> np.ndarray:
    corners = boxes_to_corners_3d(box[None].astype(np.float32))[0]
    return corners[:4, :2]


def estimate_bev_intersection_and_areas(box_a: np.ndarray, box_b: np.ndarray) -> tuple[float, float, float]:
    try:
        poly_a = box_corners_bev(box_a).astype(np.float32)
        poly_b = box_corners_bev(box_b).astype(np.float32)
        area_a = abs(cv2.contourArea(poly_a))
        area_b = abs(cv2.contourArea(poly_b))
        inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)
        return float(inter), float(area_a), float(area_b)
    except cv2.error:
        return 0.0, 0.0, 0.0


def estimate_overlap_bev(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter, area_a, area_b = estimate_bev_intersection_and_areas(box_a, box_b)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def estimate_overlap_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_bev, area_a, area_b = estimate_bev_intersection_and_areas(box_a, box_b)
    if inter_bev <= 0:
        return 0.0
    min_a = float(box_a[2] - box_a[5] * 0.5)
    max_a = float(box_a[2] + box_a[5] * 0.5)
    min_b = float(box_b[2] - box_b[5] * 0.5)
    max_b = float(box_b[2] + box_b[5] * 0.5)
    inter_h = max(0.0, min(max_a, max_b) - max(min_a, min_b))
    inter_vol = inter_bev * inter_h
    vol_a = area_a * float(box_a[5])
    vol_b = area_b * float(box_b[5])
    union = vol_a + vol_b - inter_vol
    return 0.0 if union <= 0 else float(inter_vol / union)


def estimate_accuracy_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    return float(np.linalg.norm(box_a[:3] - box_b[:3]))


def metric_label(metric_space: str) -> str:
    return "3d" if metric_space == "3d" else "bev"


def metric_values(row: dict, metric_space: str) -> tuple[float, float]:
    if metric_space == "3d":
        return row["iou_3d"], row["accuracy_3d"]
    return row["bev_overlap"], row["bev_accuracy"]


def estimate_accuracy_bev(box_a: np.ndarray, box_b: np.ndarray) -> float:
    return float(np.linalg.norm(box_a[:2] - box_b[:2]))


def row_gt_box(record, calib: Calibration) -> np.ndarray:
    return record_to_lidar_box(record, calib).astype(np.float32)


def build_pred_box(init_box: np.ndarray, init_centroid: np.ndarray, current_centroid: np.ndarray) -> np.ndarray:
    pred_box = init_box.copy()
    center_offset = init_box[:3] - init_centroid
    pred_box[:3] = current_centroid + center_offset
    return pred_box


def resolve_tracker_kwargs(args, target_class: str) -> dict:
    gate_override = args.gate_radius
    if target_class == "Car" and args.car_gate_radius is not None:
        gate_override = args.car_gate_radius
    if target_class == "Pedestrian" and args.pedestrian_gate_radius is not None:
        gate_override = args.pedestrian_gate_radius
    return {
        "gate_radius": resolve_tracker_param(gate_override, target_class, "gate_radius"),
        "cluster_radius": resolve_tracker_param(args.cluster_radius, target_class, "cluster_radius"),
        "init_points": resolve_tracker_param(args.init_points, target_class, "init_points"),
        "min_points": resolve_tracker_param(args.min_points, target_class, "min_points"),
        "box_height_filter_ratio": resolve_tracker_param(
            args.box_height_filter_ratio,
            target_class,
            "box_height_filter_ratio",
        ),
    }


def find_init_frame(
    frame_ids: list[int],
    gt: dict[int, dict],
    tracker: UtoniaTracker,
    velodyne_dir: Path,
    init_source: str,
    init_detections_root: Path | None,
    score_thresh: float,
    calib: Calibration,
    stable_min_points: int,
    stable_consecutive: int,
    init_policy: str,
) -> tuple[int | None, np.ndarray | None, int | None]:
    streak = 0
    for frame_id in frame_ids:
        try:
            init_box, _ = select_init_box(
                source=init_source,
                first_frame=frame_id,
                gt_record=gt[frame_id],
                calib=calib,
                detections_root=(init_detections_root / velodyne_dir.name) if init_detections_root is not None else Path("."),
                score_thresh=score_thresh,
            )
        except RuntimeError:
            streak = 0
            continue

        coord = load_xyz_cached(str(velodyne_dir / f"{frame_id:06d}.bin"))
        raw_support, filtered_support = tracker.count_box_support_points(coord, init_box)
        if init_policy == "immediate":
            return frame_id, init_box, raw_support
        if filtered_support >= stable_min_points:
            streak += 1
            if streak >= stable_consecutive:
                return frame_id, init_box, raw_support
        else:
            streak = 0
    return None, None, None


def summarize_track(
    frame_rows: list[dict],
    success_metric: SuccessMetric,
    precision_metric: PrecisionMetric,
    metric_space: str,
) -> dict:
    label = metric_label(metric_space)
    values = [metric_values(row, metric_space) for row in frame_rows]
    overlaps = np.array([value[0] for value in values], dtype=np.float32)
    accuracies = np.array([value[1] for value in values], dtype=np.float32)
    first_lost = next((row["frame_id"] for row in frame_rows if row["status"] == "lost"), None)
    raw_support = np.array([row["raw_box_support_points"] for row in frame_rows], dtype=np.float32)
    filtered_support = np.array([row["filtered_box_support_points"] for row in frame_rows], dtype=np.float32)
    target_support = np.array([row["support_count"] for row in frame_rows], dtype=np.float32)
    pre_lost_support = next(
        (
            row["pre_lost_window_min_support"]
            for row in frame_rows
            if row["pre_lost_window_min_support"] is not None
        ),
        None,
    )
    pre_lost_target_support = next(
        (
            row["pre_lost_window_min_target_support"]
            for row in frame_rows
            if row["pre_lost_window_min_target_support"] is not None
        ),
        None,
    )
    summary = {
        "sequence": frame_rows[0]["sequence"],
        "track_id": frame_rows[0]["track_id"],
        "class": frame_rows[0]["class"],
        "track_length": frame_rows[0]["track_length"],
        "gt_first_frame": frame_rows[0]["gt_first_frame"],
        "init_frame": frame_rows[0]["init_frame"],
        "init_delay": frame_rows[0]["init_delay"],
        "init_policy": frame_rows[0]["init_policy"],
        "init_support_points": frame_rows[0]["init_support_points"],
        "num_frames": len(frame_rows),
        "success": success_metric.value(overlaps),
        "precision": precision_metric.value(accuracies),
        "mean_overlap": float(np.mean(overlaps)),
        "mean_accuracy": float(np.mean(accuracies)),
        "median_accuracy": float(np.median(accuracies)),
        "lost_rate": sum(int(row["status"] == "lost") for row in frame_rows) / len(frame_rows),
        "first_lost_frame": first_lost,
        "used_detection_frames": sum(int(row["used_detection"]) for row in frame_rows),
        "min_raw_box_support_points": int(raw_support.min()) if raw_support.size else None,
        "median_raw_box_support_points": float(np.median(raw_support)) if raw_support.size else None,
        "min_filtered_box_support_points": int(filtered_support.min()) if filtered_support.size else None,
        "median_filtered_box_support_points": float(np.median(filtered_support)) if filtered_support.size else None,
        "min_target_support_count": int(target_support.min()) if target_support.size else None,
        "median_target_support_count": float(np.median(target_support)) if target_support.size else None,
        "pre_lost_window_min_support": pre_lost_support,
        "pre_lost_window_min_target_support": pre_lost_target_support,
        "bad_update_frames": sum(int(row["bad_update"]) for row in frame_rows),
        "sparse_mode_frames": sum(int(row["sparse_mode"]) for row in frame_rows),
        "recovered_frames": sum(int(row["recovered"]) for row in frame_rows),
    }
    summary[f"success_{label}"] = summary["success"]
    summary[f"precision_{label}"] = summary["precision"]
    summary[f"mean_{label}_overlap"] = summary["mean_overlap"]
    summary[f"mean_{label}_accuracy"] = summary["mean_accuracy"]
    summary[f"median_{label}_accuracy"] = summary["median_accuracy"]
    return summary


def aggregate_frame_rows(
    frame_rows: list[dict],
    success_metric: SuccessMetric,
    precision_metric: PrecisionMetric,
    metric_space: str,
) -> dict:
    if not frame_rows:
        return {}
    label = metric_label(metric_space)
    values = [metric_values(row, metric_space) for row in frame_rows]
    overlaps = np.array([value[0] for value in values], dtype=np.float32)
    accuracies = np.array([value[1] for value in values], dtype=np.float32)
    summary = {
        "num_frames": len(frame_rows),
        "success": success_metric.value(overlaps),
        "precision": precision_metric.value(accuracies),
        "mean_overlap": float(np.mean(overlaps)),
        "mean_accuracy": float(np.mean(accuracies)),
        "median_accuracy": float(np.median(accuracies)),
        "lost_rate": sum(int(row["status"] == "lost") for row in frame_rows) / len(frame_rows),
        "used_detection_rate": sum(int(row["used_detection"]) for row in frame_rows) / len(frame_rows),
    }
    summary[f"success_{label}"] = summary["success"]
    summary[f"precision_{label}"] = summary["precision"]
    summary[f"mean_{label}_overlap"] = summary["mean_overlap"]
    summary[f"mean_{label}_accuracy"] = summary["mean_accuracy"]
    summary[f"median_{label}_accuracy"] = summary["median_accuracy"]
    return summary


def aggregate_track_rows(track_rows: list[dict]) -> dict:
    if not track_rows:
        return {}
    lost = [row["first_lost_frame"] for row in track_rows if row["first_lost_frame"] is not None]
    return {
        "num_tracklets": len(track_rows),
        "mean_track_length": float(np.mean([row["track_length"] for row in track_rows])),
        "mean_success": float(np.mean([row["success"] for row in track_rows])),
        "mean_precision": float(np.mean([row["precision"] for row in track_rows])),
        "lost_track_rate": sum(int(row["first_lost_frame"] is not None) for row in track_rows) / len(track_rows),
        "median_first_lost_frame": float(np.median(lost)) if lost else None,
    }


def summarize_subset(
    frame_rows: list[dict],
    track_rows: list[dict],
    success_metric: SuccessMetric,
    precision_metric: PrecisionMetric,
    metric_space: str,
) -> dict:
    return {
        "frame_metrics": aggregate_frame_rows(frame_rows, success_metric, precision_metric, metric_space),
        "track_metrics": aggregate_track_rows(track_rows),
    }


def main():
    args = parse_args()
    if args.init_source != "gt" or args.update_source != "none" or args.init_policy != "immediate":
        print(
            "Warning: published-style comparability is strongest with --init-source gt --update-source none --init-policy immediate"
        )

    data_root = Path(args.data_root)
    detections_base_root = Path(args.detections_base_root)
    output_root = Path(args.output_root)
    sequence_ids = sequence_ids_from_args(args)
    configure_frame_cache(args.frame_cache_size)
    run_name = args.run_name or f"3d_{args.metric_space}_{args.split}__init_{args.init_source}__update_{args.update_source}__policy_{args.init_policy}__classes_{'-'.join(args.classes)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracklets = build_tracklets(
        data_root=data_root,
        sequence_ids=sequence_ids,
        classes=set(args.classes),
        min_track_length=args.min_track_length,
    )
    if args.track_list_csv is not None:
        selected_tracks = read_track_list(Path(args.track_list_csv))
        tracklets = [
            tracklet
            for tracklet in tracklets
            if (tracklet["sequence"], tracklet["track_id"]) in selected_tracks
        ]
    tracklets = sorted(tracklets, key=lambda tracklet: len(tracklet["records"]), reverse=True)
    if args.max_tracks is not None:
        tracklets = tracklets[: args.max_tracks]

    init_det_root = resolve_detection_root(detections_base_root, args.init_source)
    update_det_root = resolve_detection_root(detections_base_root, args.update_source)

    shared_model_tracker = UtoniaTracker(mode="local_crop")
    shared_model_tracker.build_model()
    shared_model = shared_model_tracker.model

    success_metric = SuccessMetric(n=args.success_steps)
    precision_metric = PrecisionMetric(n=args.precision_steps, max_distance=args.precision_max_distance)

    frame_rows = []
    track_rows = []
    skipped_tracklets = []

    print(
        f"Evaluating {len(tracklets)} 3D SOT tracklets "
        f"with metric_space={args.metric_space}",
        flush=True,
    )
    eval_started_at = time.monotonic()
    for index, tracklet in enumerate(tracklets, start=1):
        track_started_at = time.monotonic()
        sequence = tracklet["sequence"]
        track_id = tracklet["track_id"]
        target_class = tracklet["target_class"]
        records = tracklet["records"]
        track_length = tracklet["track_length"]
        gt = {record["frame"]: record for record in records}
        velodyne_dir = data_root / "velodyne" / sequence
        frame_ids = [frame_id for frame_id in sorted(gt.keys()) if (velodyne_dir / f"{frame_id:06d}.bin").exists()]
        if not frame_ids:
            continue
        if args.max_track_frames is not None:
            frame_ids = frame_ids[: args.max_track_frames]

        tracker_kwargs = resolve_tracker_kwargs(args, target_class)
        stable_init_min_points = args.stable_init_min_points or STABLE_INIT_PRESETS.get(target_class, STABLE_INIT_PRESETS["Car"])

        elapsed = time.monotonic() - eval_started_at
        avg_track_seconds = elapsed / max(index - 1, 1)
        eta_seconds = avg_track_seconds * (len(tracklets) - index + 1)
        print(
            f"[{index}/{len(tracklets)}] "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)} "
            f"seq={sequence} track_id={track_id} class={target_class} frames={len(frame_ids)}",
            flush=True,
        )

        calib = Calibration(data_root / "calib" / f"{sequence}.txt")
        tracker = UtoniaTracker(
            model=shared_model,
            mode="local_crop",
            init_radius=1.6,
            cluster_radius=tracker_kwargs["cluster_radius"],
            gate_radius=tracker_kwargs["gate_radius"],
            local_crop_radius=args.local_crop_radius,
            local_crop_min_points=args.local_crop_min_points,
            init_points=tracker_kwargs["init_points"],
            min_points=tracker_kwargs["min_points"],
            detection_overlap_threshold=args.detection_overlap_thresh,
            box_height_filter_ratio=tracker_kwargs["box_height_filter_ratio"],
            lost_height_ratio=args.lost_height_ratio,
            lost_bad_frames=args.lost_bad_frames,
            sparse_support_threshold=args.sparse_support_threshold,
            sparse_sim_threshold=args.sparse_sim_threshold,
            sparse_cluster_radius_scale=args.sparse_cluster_radius_scale,
            sparse_no_proto_update=args.sparse_no_proto_update,
            recovery_max_frames=args.recovery_max_frames,
            recovery_crop_radius_scale=args.recovery_crop_radius_scale,
            recovery_gate_radius_scale=args.recovery_gate_radius_scale,
            recovery_sim_threshold=args.recovery_sim_threshold,
        )

        gt_first_frame = frame_ids[0]
        init_frame, init_box, init_support_points = find_init_frame(
            frame_ids=frame_ids,
            gt=gt,
            tracker=tracker,
            velodyne_dir=velodyne_dir,
            init_source=args.init_source,
            init_detections_root=init_det_root,
            score_thresh=args.score_thresh,
            calib=calib,
            stable_min_points=stable_init_min_points,
            stable_consecutive=args.stable_init_consecutive,
            init_policy=args.init_policy,
        )
        if init_frame is None or init_box is None:
            skipped_tracklets.append(
                {
                    "sequence": sequence,
                    "track_id": track_id,
                    "class": target_class,
                    "reason": f"No valid {args.init_policy} init frame found",
                }
            )
            print("  skipped: no valid init frame found", flush=True)
            continue

        state = tracker.initialize_from_box(
            load_xyz_cached(str(velodyne_dir / f"{init_frame:06d}.bin")),
            init_box,
        )
        init_centroid = np.asarray(state["centroid"], dtype=np.float32)

        track_frame_rows = []
        active_frame_ids = [frame_id for frame_id in frame_ids if frame_id >= init_frame]
        for frame_id in active_frame_ids:
            record = gt[frame_id]
            if frame_id != init_frame:
                detection_meta = {
                    "matched_detection_found": False,
                    "matched_detection_distance": None,
                    "matched_detection_score": None,
                }
                detection_box = None
                if args.update_source != "none":
                    detection_box, _, detection_meta = match_update_box(
                        source=args.update_source,
                        frame_id=frame_id,
                        gt_record=record,
                        target_class=target_class,
                        calib=calib,
                        detections_root=(update_det_root / sequence) if update_det_root is not None else Path("."),
                        score_thresh=args.score_thresh,
                        pred_centroid=tracker.predict_position().detach().cpu().numpy(),
                        match_radius=tracker.gate_radius,
                    )
                state = tracker.step(
                    load_xyz_cached(str(velodyne_dir / f"{frame_id:06d}.bin")),
                    detection_box=detection_box,
                )
            else:
                detection_meta = {
                    "matched_detection_found": False,
                    "matched_detection_distance": None,
                    "matched_detection_score": None,
                }

            pred_box = build_pred_box(init_box, init_centroid, np.asarray(state["centroid"], dtype=np.float32))
            gt_box = row_gt_box(record, calib)
            raw_box_support, filtered_box_support = tracker.count_box_support_points(
                load_xyz_cached(str(velodyne_dir / f"{frame_id:06d}.bin")),
                gt_box,
            )
            bev_overlap = estimate_overlap_bev(pred_box, gt_box)
            bev_accuracy = estimate_accuracy_bev(pred_box, gt_box)
            iou_3d = estimate_overlap_3d(pred_box, gt_box)
            accuracy_3d = estimate_accuracy_3d(pred_box, gt_box)

            row = {
                "sequence": sequence,
                "track_id": track_id,
                "class": target_class,
                "track_length": track_length,
                "gt_first_frame": gt_first_frame,
                "init_frame": init_frame,
                "init_delay": init_frame - gt_first_frame,
                "init_policy": args.init_policy,
                "init_support_points": init_support_points,
                "frame_id": frame_id,
                "frame_in_track": frame_id - gt_first_frame,
                "frame_since_init": frame_id - init_frame,
                "status": state.get("status", "active"),
                "used_detection": int(state.get("used_box") is not None),
                "matched_detection_found": int(detection_meta["matched_detection_found"]),
                "matched_detection_distance": maybe_float(detection_meta["matched_detection_distance"]),
                "matched_detection_score": maybe_float(detection_meta["matched_detection_score"]),
                "support_count": int(state.get("support_count", 0)),
                "support_height": maybe_float(state.get("support_height")),
                "raw_box_support_points": raw_box_support,
                "filtered_box_support_points": filtered_box_support,
                "bad_update": int(state.get("bad_update", False)),
                "bad_update_reason": state.get("bad_update_reason", ""),
                "jump_xy": maybe_float(state.get("jump_xy")),
                "used_box_overlap": maybe_float(state.get("used_box_overlap")),
                "sparse_mode": int(state.get("sparse_mode", False)),
                "recovered": int(state.get("recovered", False)),
                "pre_lost_window_min_support": None,
                "pre_lost_window_min_target_support": None,
                "pred_x": float(pred_box[0]),
                "pred_y": float(pred_box[1]),
                "pred_z": float(pred_box[2]),
                "pred_dx": float(pred_box[3]),
                "pred_dy": float(pred_box[4]),
                "pred_dz": float(pred_box[5]),
                "pred_heading": float(pred_box[6]),
                "gt_x": float(gt_box[0]),
                "gt_y": float(gt_box[1]),
                "gt_z": float(gt_box[2]),
                "gt_dx": float(gt_box[3]),
                "gt_dy": float(gt_box[4]),
                "gt_dz": float(gt_box[5]),
                "gt_heading": float(gt_box[6]),
                "bev_overlap": bev_overlap,
                "bev_accuracy": bev_accuracy,
                "iou_3d": iou_3d,
                "accuracy_3d": accuracy_3d,
            }
            frame_rows.append(row)
            track_frame_rows.append(row)

        first_lost_index = next(
            (
                row_index
                for row_index, row in enumerate(track_frame_rows)
                if row["status"] == "lost"
            ),
            None,
        )
        if first_lost_index is not None:
            start = max(0, first_lost_index - args.pre_lost_window)
            window_rows = track_frame_rows[start : first_lost_index + 1]
            min_support = min(row["filtered_box_support_points"] for row in window_rows)
            min_target_support = min(row["support_count"] for row in window_rows)
            for row in track_frame_rows:
                row["pre_lost_window_min_support"] = min_support
                row["pre_lost_window_min_target_support"] = min_target_support

        track_rows.append(summarize_track(track_frame_rows, success_metric, precision_metric, args.metric_space))
        print(
            f"  done in {format_duration(time.monotonic() - track_started_at)}",
            flush=True,
        )

    filtered_track_rows = [row for row in track_rows if row["track_length"] >= args.report_min_track_length]
    filtered_keys = {(row["sequence"], row["track_id"]) for row in filtered_track_rows}
    filtered_frame_rows = [row for row in frame_rows if (row["sequence"], row["track_id"]) in filtered_keys]

    summary = {
        "config": vars(args),
        "num_tracklets": len(track_rows),
        "num_skipped_tracklets": len(skipped_tracklets),
        "skipped_tracklets": skipped_tracklets,
        "reports": {
            "all_tracks": summarize_subset(
                frame_rows,
                track_rows,
                success_metric,
                precision_metric,
                args.metric_space,
            ),
            f"tracks_ge_{args.report_min_track_length}": summarize_subset(
                filtered_frame_rows,
                filtered_track_rows,
                success_metric,
                precision_metric,
                args.metric_space,
            ),
        },
        "by_class": {},
    }
    all_classes = sorted({row["class"] for row in frame_rows})
    for class_name in all_classes:
        class_frame_rows = [row for row in frame_rows if row["class"] == class_name]
        class_track_rows = [row for row in track_rows if row["class"] == class_name]
        class_filtered_track_rows = [row for row in filtered_track_rows if row["class"] == class_name]
        class_filtered_keys = {(row["sequence"], row["track_id"]) for row in class_filtered_track_rows}
        class_filtered_frame_rows = [
            row for row in class_frame_rows if (row["sequence"], row["track_id"]) in class_filtered_keys
        ]
        summary["by_class"][class_name] = {
            "all_tracks": summarize_subset(class_frame_rows, class_track_rows, success_metric, precision_metric, args.metric_space),
            f"tracks_ge_{args.report_min_track_length}": summarize_subset(
                class_filtered_frame_rows,
                class_filtered_track_rows,
                success_metric,
                precision_metric,
                args.metric_space,
            ),
        }

    frame_fieldnames = [
        "sequence",
        "track_id",
        "class",
        "track_length",
        "gt_first_frame",
        "init_frame",
        "init_delay",
        "init_policy",
        "init_support_points",
        "frame_id",
        "frame_in_track",
        "frame_since_init",
        "status",
        "used_detection",
        "matched_detection_found",
        "matched_detection_distance",
        "matched_detection_score",
        "support_count",
        "support_height",
        "raw_box_support_points",
        "filtered_box_support_points",
        "bad_update",
        "bad_update_reason",
        "jump_xy",
        "used_box_overlap",
        "sparse_mode",
        "recovered",
        "pre_lost_window_min_support",
        "pre_lost_window_min_target_support",
        "pred_x",
        "pred_y",
        "pred_z",
        "pred_dx",
        "pred_dy",
        "pred_dz",
        "pred_heading",
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_dx",
        "gt_dy",
        "gt_dz",
        "gt_heading",
        "bev_overlap",
        "bev_accuracy",
        "iou_3d",
        "accuracy_3d",
    ]
    track_fieldnames = sorted({key for row in track_rows for key in row.keys()})
    save_csv(run_dir / "per_frame.csv", frame_rows, frame_fieldnames)
    save_csv(run_dir / "per_track.csv", track_rows, track_fieldnames)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved evaluation outputs to {run_dir}")
    print(json.dumps(summary["reports"]["all_tracks"]["frame_metrics"], indent=2))


if __name__ == "__main__":
    main()
