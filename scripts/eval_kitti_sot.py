import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from scripts.kitti_tracker_test import (  # noqa: E402
    TRACKER_PRESETS,
    canonical_label,
    gt_by_frame,
    load_xyz,
    match_update_box,
    parse_labels,
    record_to_lidar_box,
    resolve_tracker_param,
    select_init_box,
)
from utonia import UtoniaTracker  # noqa: E402


STABLE_INIT_PRESETS = {
    "Car": 20,
    "Van": 20,
    "Cyclist": 12,
    "Pedestrian": 10,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with velodyne/, calib/, and label_02/.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Car", "Cyclist", "Pedestrian"],
        choices=sorted(TRACKER_PRESETS.keys()),
        help="Classes to evaluate as internal SOT tracklets.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional subset of KITTI tracking sequences, e.g. 0000 0001.",
    )
    parser.add_argument(
        "--init-source",
        choices=["gt", "pointpillar", "pointrcnn"],
        default="gt",
        help="How the track is initialized on the first annotated frame.",
    )
    parser.add_argument(
        "--update-source",
        choices=["none", "gt", "pointpillar", "pointrcnn"],
        default="none",
        help="Optional source for detector-guided prototype updates during tracking.",
    )
    parser.add_argument(
        "--detections-base-root",
        default=str(REPO_ROOT / "data" / "detections"),
        help="Base directory containing <model>/npz/<sequence>/<frame>.npz.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Detection score threshold for init/update sources that use detections.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=1,
        help="Only include GT tracklets with at least this many annotated frames.",
    )
    parser.add_argument(
        "--report-min-track-length",
        type=int,
        default=10,
        help="Also report a separate summary for tracklets with at least this many frames.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Optional cap on the number of tracklets to evaluate.",
    )
    parser.add_argument(
        "--track-list-csv",
        default=None,
        help="Optional CSV with sequence and track_id columns to restrict evaluation to a specific subset.",
    )
    parser.add_argument(
        "--init-policy",
        choices=["immediate", "stable"],
        default="immediate",
        help="When to initialize the tracker inside each GT tracklet.",
    )
    parser.add_argument(
        "--stable-init-min-points",
        type=int,
        default=None,
        help="Override the class-aware filtered support threshold used by stable init.",
    )
    parser.add_argument(
        "--stable-init-consecutive",
        type=int,
        default=2,
        help="How many consecutive frames must satisfy the stable init support threshold.",
    )
    parser.add_argument(
        "--max-track-frames",
        type=int,
        default=None,
        help="Optional per-track frame cap for quick smoke tests.",
    )
    parser.add_argument(
        "--local-crop-radius",
        type=float,
        default=8.0,
        help="Utonia local crop radius for internal evaluation.",
    )
    parser.add_argument(
        "--local-crop-min-points",
        type=int,
        default=2048,
        help="Minimum local crop points before nearest-point fallback.",
    )
    parser.add_argument(
        "--detection-overlap-thresh",
        type=float,
        default=0.3,
        help="Use a matched detection box only if it overlaps enough with the tracked support points.",
    )
    parser.add_argument(
        "--gate-radius",
        type=float,
        default=None,
        help="Override the class-aware xy motion gate radius.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=None,
        help="Override the class-aware clustering radius.",
    )
    parser.add_argument(
        "--init-points",
        type=int,
        default=None,
        help="Override the class-aware initialization support count.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="Override the class-aware tracking support count.",
    )
    parser.add_argument(
        "--box-height-filter-ratio",
        type=float,
        default=None,
        help="Override the class-aware height-ratio support filtering.",
    )
    parser.add_argument(
        "--lost-height-ratio",
        type=float,
        default=0.5,
        help="Lost-state height collapse threshold ratio.",
    )
    parser.add_argument(
        "--lost-bad-frames",
        type=int,
        default=2,
        help="Number of consecutive bad frames needed before switching to lost.",
    )
    parser.add_argument(
        "--projection-min-points",
        type=int,
        default=6,
        help="Minimum valid tracked points needed to form a 2D prediction box.",
    )
    parser.add_argument(
        "--projection-percentile-low",
        type=float,
        default=5.0,
        help="Lower percentile for robust projected 2D boxes.",
    )
    parser.add_argument(
        "--projection-percentile-high",
        type=float,
        default=95.0,
        help="Upper percentile for robust projected 2D boxes.",
    )
    parser.add_argument(
        "--projection-margin",
        type=float,
        default=4.0,
        help="Extra pixel margin around projected percentile boxes.",
    )
    parser.add_argument(
        "--precision-thresholds",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 40.0],
        help="Pixel thresholds used for internal precision curves.",
    )
    parser.add_argument(
        "--success-thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds used for internal success curves.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data" / "eval_sot"),
        help="Directory where per-frame, per-track, and summary outputs are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name. Defaults to a descriptive auto-generated name.",
    )
    return parser.parse_args()


def build_tracklets(data_root: Path, sequences: list[str] | None, classes: set[str], min_track_length: int):
    label_root = data_root / "label_02"
    sequence_ids = sequences or sorted(path.stem for path in label_root.glob("*.txt"))
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
            target_class = canonical_label(track_records[0]["type"])
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


def read_track_list(path: Path) -> set[tuple[str, int]]:
    with path.open() as handle:
        rows = csv.DictReader(handle)
        return {(row["sequence"], int(row["track_id"])) for row in rows}


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
) -> tuple[int | None, np.ndarray | None, int | None, list[dict]]:
    support_log = []
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
        except RuntimeError as error:
            support_log.append(
                {
                    "frame_id": frame_id,
                    "raw_support_points": None,
                    "filtered_support_points": None,
                    "init_box_found": False,
                    "init_box_error": str(error),
                }
            )
            streak = 0
            continue

        coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
        raw_support, filtered_support = tracker.count_box_support_points(coord, init_box)
        support_log.append(
            {
                "frame_id": frame_id,
                "raw_support_points": raw_support,
                "filtered_support_points": filtered_support,
                "init_box_found": True,
                "init_box_error": "",
            }
        )

        if init_policy == "immediate":
            return frame_id, init_box, raw_support, support_log

        if filtered_support >= stable_min_points:
            streak += 1
            if streak >= stable_consecutive:
                return frame_id, init_box, raw_support, support_log
        else:
            streak = 0

    return None, None, None, support_log


def project_points_to_box(
    points_lidar: np.ndarray,
    calib: Calibration,
    min_points: int,
    percentile_low: float,
    percentile_high: float,
    margin: float,
) -> tuple[np.ndarray | None, int]:
    if len(points_lidar) < min_points:
        return None, 0

    points_img, depth = calib.lidar_to_img(points_lidar)
    valid = np.isfinite(points_img).all(axis=1) & np.isfinite(depth) & (depth > 0)
    points_img = points_img[valid]
    if len(points_img) < min_points:
        return None, int(len(points_img))

    x1 = np.percentile(points_img[:, 0], percentile_low) - margin
    y1 = np.percentile(points_img[:, 1], percentile_low) - margin
    x2 = np.percentile(points_img[:, 0], percentile_high) + margin
    y2 = np.percentile(points_img[:, 1], percentile_high) + margin
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None, int(len(points_img))
    return np.array([x1, y1, x2, y2], dtype=np.float32), int(len(points_img))


def box_iou_2d(box_a: np.ndarray | None, box_b: np.ndarray) -> float:
    if box_a is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def center_error_px(box_a: np.ndarray | None, box_b: np.ndarray) -> float | None:
    if box_a is None:
        return None
    ax = 0.5 * (box_a[0] + box_a[2])
    ay = 0.5 * (box_a[1] + box_a[3])
    bx = 0.5 * (box_b[0] + box_b[2])
    by = 0.5 * (box_b[1] + box_b[3])
    return float(math.hypot(ax - bx, ay - by))


def summarize_track(frame_rows: list[dict], precision_thresholds: list[float], success_thresholds: list[float]) -> dict:
    num_frames = len(frame_rows)
    num_valid = sum(int(row["pred_valid"]) for row in frame_rows)
    iou_all = [row["iou_2d"] for row in frame_rows]
    iou_valid = [row["iou_2d"] for row in frame_rows if row["pred_valid"]]
    center_valid = [row["center_error_px"] for row in frame_rows if row["center_error_px"] is not None]
    first_lost = next((row["frame_id"] for row in frame_rows if row["status"] == "lost"), None)
    out = {
        "sequence": frame_rows[0]["sequence"],
        "track_id": frame_rows[0]["track_id"],
        "class": frame_rows[0]["class"],
        "track_length": frame_rows[0]["track_length"],
        "gt_first_frame": frame_rows[0]["gt_first_frame"],
        "init_frame": frame_rows[0]["init_frame"],
        "init_delay": frame_rows[0]["init_delay"],
        "init_policy": frame_rows[0]["init_policy"],
        "init_support_points": frame_rows[0]["init_support_points"],
        "num_frames": num_frames,
        "num_valid_predictions": num_valid,
        "valid_rate": num_valid / num_frames,
        "mean_iou_all": float(np.mean(iou_all)),
        "mean_iou_valid": float(np.mean(iou_valid)) if iou_valid else None,
        "mean_center_error_px": float(np.mean(center_valid)) if center_valid else None,
        "median_center_error_px": float(np.median(center_valid)) if center_valid else None,
        "first_lost_frame": first_lost,
        "used_detection_frames": sum(int(row["used_detection"]) for row in frame_rows),
        "lost_frames": sum(int(row["status"] == "lost") for row in frame_rows),
        "bad_update_frames": sum(int(row["bad_update"]) for row in frame_rows),
    }
    for threshold in precision_thresholds:
        key = f"precision@{threshold:g}px"
        out[key] = sum(int(row["center_error_px"] is not None and row["center_error_px"] <= threshold) for row in frame_rows) / num_frames
    for threshold in success_thresholds:
        key = f"success@{threshold:g}"
        out[key] = sum(int(row["iou_2d"] >= threshold) for row in frame_rows) / num_frames
    return out


def aggregate_track_rows(track_rows: list[dict]) -> dict:
    if not track_rows:
        return {}
    first_lost = [row["first_lost_frame"] for row in track_rows if row["first_lost_frame"] is not None]
    return {
        "num_tracklets": len(track_rows),
        "mean_track_length": float(np.mean([row["track_length"] for row in track_rows])),
        "mean_valid_rate": float(np.mean([row["valid_rate"] for row in track_rows])),
        "mean_iou_all": float(np.mean([row["mean_iou_all"] for row in track_rows])),
        "mean_center_error_px": float(np.mean([row["mean_center_error_px"] for row in track_rows if row["mean_center_error_px"] is not None]))
        if any(row["mean_center_error_px"] is not None for row in track_rows)
        else None,
        "lost_track_rate": sum(int(row["first_lost_frame"] is not None) for row in track_rows) / len(track_rows),
        "median_first_lost_frame": float(np.median(first_lost)) if first_lost else None,
    }


def aggregate_frame_rows(frame_rows: list[dict], precision_thresholds: list[float], success_thresholds: list[float]) -> dict:
    if not frame_rows:
        return {}
    num_frames = len(frame_rows)
    num_valid = sum(int(row["pred_valid"]) for row in frame_rows)
    valid_centers = [row["center_error_px"] for row in frame_rows if row["center_error_px"] is not None]
    result = {
        "num_frames": num_frames,
        "valid_rate": num_valid / num_frames,
        "mean_iou_all": float(np.mean([row["iou_2d"] for row in frame_rows])),
        "mean_center_error_px": float(np.mean(valid_centers)) if valid_centers else None,
        "median_center_error_px": float(np.median(valid_centers)) if valid_centers else None,
        "lost_rate": sum(int(row["status"] == "lost") for row in frame_rows) / num_frames,
        "used_detection_rate": sum(int(row["used_detection"]) for row in frame_rows) / num_frames,
        "bad_update_rate": sum(int(row["bad_update"]) for row in frame_rows) / num_frames,
        "mean_support_height": float(np.mean([row["support_height"] for row in frame_rows if row["support_height"] is not None]))
        if any(row["support_height"] is not None for row in frame_rows)
        else None,
        "mean_support_count": float(np.mean([row["support_count"] for row in frame_rows if row["support_count"] is not None]))
        if any(row["support_count"] is not None for row in frame_rows)
        else None,
        "mean_jump_xy": float(np.mean([row["jump_xy"] for row in frame_rows if row["jump_xy"] is not None]))
        if any(row["jump_xy"] is not None for row in frame_rows)
        else None,
    }
    result["precision_curve"] = {
        f"{threshold:g}": sum(int(row["center_error_px"] is not None and row["center_error_px"] <= threshold) for row in frame_rows) / num_frames
        for threshold in precision_thresholds
    }
    result["success_curve"] = {
        f"{threshold:g}": sum(int(row["iou_2d"] >= threshold) for row in frame_rows) / num_frames
        for threshold in success_thresholds
    }
    return result


def row_bbox(record) -> np.ndarray:
    return np.array(
        [record["bbox_left"], record["bbox_top"], record["bbox_right"], record["bbox_bottom"]],
        dtype=np.float32,
    )


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


def maybe_int(value):
    if value is None:
        return None
    return int(value)


def summarize_subset(
    frame_rows: list[dict],
    track_rows: list[dict],
    precision_thresholds: list[float],
    success_thresholds: list[float],
) -> dict:
    return {
        "frame_metrics": aggregate_frame_rows(frame_rows, precision_thresholds, success_thresholds),
        "track_metrics": aggregate_track_rows(track_rows),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    detections_base_root = Path(args.detections_base_root)
    output_root = Path(args.output_root)
    run_name = args.run_name or f"init_{args.init_source}__update_{args.update_source}__classes_{'-'.join(args.classes)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracklets = build_tracklets(
        data_root=data_root,
        sequences=args.sequences,
        classes={canonical_label(name) for name in args.classes},
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

    frame_rows = []
    track_rows = []
    skipped_tracklets = []

    print(f"Evaluating {len(tracklets)} tracklets")
    for index, tracklet in enumerate(tracklets, start=1):
        sequence = tracklet["sequence"]
        track_id = tracklet["track_id"]
        target_class = canonical_label(tracklet["target_class"])
        records = tracklet["records"]
        track_length = tracklet["track_length"]
        gt = {record["frame"]: record for record in records}
        velodyne_dir = data_root / "velodyne" / sequence
        frame_ids = [frame_id for frame_id in sorted(gt.keys()) if (velodyne_dir / f"{frame_id:06d}.bin").exists()]
        if not frame_ids:
            continue
        if args.max_track_frames is not None:
            frame_ids = frame_ids[: args.max_track_frames]

        gate_radius = resolve_tracker_param(args.gate_radius, target_class, "gate_radius")
        cluster_radius = resolve_tracker_param(args.cluster_radius, target_class, "cluster_radius")
        init_points = resolve_tracker_param(args.init_points, target_class, "init_points")
        min_points = resolve_tracker_param(args.min_points, target_class, "min_points")
        box_height_filter_ratio = resolve_tracker_param(
            args.box_height_filter_ratio,
            target_class,
            "box_height_filter_ratio",
        )
        stable_init_min_points = args.stable_init_min_points or STABLE_INIT_PRESETS.get(target_class, STABLE_INIT_PRESETS["Car"])

        print(
            f"[{index}/{len(tracklets)}] seq={sequence} track_id={track_id} class={target_class} frames={len(frame_ids)}"
        )

        calib = Calibration(data_root / "calib" / f"{sequence}.txt")
        tracker = UtoniaTracker(
            model=shared_model,
            mode="local_crop",
            init_radius=1.6,
            cluster_radius=cluster_radius,
            gate_radius=gate_radius,
            local_crop_radius=args.local_crop_radius,
            local_crop_min_points=args.local_crop_min_points,
            init_points=init_points,
            min_points=min_points,
            detection_overlap_threshold=args.detection_overlap_thresh,
            box_height_filter_ratio=box_height_filter_ratio,
            lost_height_ratio=args.lost_height_ratio,
            lost_bad_frames=args.lost_bad_frames,
        )

        gt_first_frame = frame_ids[0]
        init_frame, init_box, init_raw_support_points, init_support_log = find_init_frame(
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
            print("  skipped: no valid init frame found")
            continue
        first_frame = init_frame
        first_record = gt[first_frame]
        state = tracker.initialize_from_box(
            load_xyz(velodyne_dir / f"{first_frame:06d}.bin"),
            init_box,
        )
        frame_ids = [frame_id for frame_id in frame_ids if frame_id >= first_frame]
        if args.max_track_frames is not None:
            frame_ids = frame_ids[: args.max_track_frames]

        track_frame_rows = []
        for frame_id in frame_ids:
            record = gt[frame_id]
            gt_box_2d = row_bbox(record)
            gt_bbox_height = float(record["bbox_bottom"] - record["bbox_top"])
            if frame_id != first_frame:
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
                else:
                    detection_meta = {
                        "matched_detection_found": False,
                        "matched_detection_distance": None,
                        "matched_detection_score": None,
                    }
                state = tracker.step(
                    load_xyz(velodyne_dir / f"{frame_id:06d}.bin"),
                    detection_box=detection_box,
                )
            else:
                detection_meta = {
                    "matched_detection_found": False,
                    "matched_detection_distance": None,
                    "matched_detection_score": None,
                }

            object_points = state["coord"][state["mask"]] if len(state["coord"]) else np.zeros((0, 3), dtype=np.float32)
            pred_box_2d, projected_points = project_points_to_box(
                points_lidar=object_points,
                calib=calib,
                min_points=args.projection_min_points,
                percentile_low=args.projection_percentile_low,
                percentile_high=args.projection_percentile_high,
                margin=args.projection_margin,
            )
            if state["status"] == "lost":
                pred_box_2d = None
                projected_points = 0

            iou_2d = box_iou_2d(pred_box_2d, gt_box_2d)
            center_err = center_error_px(pred_box_2d, gt_box_2d)
            row = {
                "sequence": sequence,
                "track_id": track_id,
                "class": target_class,
                "track_length": track_length,
                "gt_first_frame": gt_first_frame,
                "init_frame": first_frame,
                "init_delay": first_frame - gt_first_frame,
                "init_policy": args.init_policy,
                "init_support_points": init_raw_support_points,
                "frame_id": frame_id,
                "frame_in_track": frame_id - gt_first_frame,
                "frame_since_init": frame_id - first_frame,
                "status": state.get("status", "active"),
                "status_transition": state.get("status_transition", ""),
                "bad_update": int(state.get("bad_update", False)),
                "bad_update_reason": state.get("bad_update_reason", ""),
                "jump_xy": maybe_float(state.get("jump_xy")),
                "support_height": maybe_float(state.get("support_height")),
                "support_count": maybe_int(state.get("support_count")),
                "pred_valid": int(pred_box_2d is not None),
                "pred_x1": maybe_float(None if pred_box_2d is None else pred_box_2d[0]),
                "pred_y1": maybe_float(None if pred_box_2d is None else pred_box_2d[1]),
                "pred_x2": maybe_float(None if pred_box_2d is None else pred_box_2d[2]),
                "pred_y2": maybe_float(None if pred_box_2d is None else pred_box_2d[3]),
                "gt_x1": float(gt_box_2d[0]),
                "gt_y1": float(gt_box_2d[1]),
                "gt_x2": float(gt_box_2d[2]),
                "gt_y2": float(gt_box_2d[3]),
                "gt_bbox_height_px": gt_bbox_height,
                "gt_occlusion": int(record["occlusion"]),
                "gt_truncation": float(record["truncation"]),
                "iou_2d": float(iou_2d),
                "center_error_px": maybe_float(center_err),
                "used_detection": int(state.get("used_box") is not None),
                "used_box_overlap": maybe_float(state.get("used_box_overlap")),
                "matched_detection_found": int(detection_meta["matched_detection_found"]),
                "matched_detection_distance": maybe_float(detection_meta["matched_detection_distance"]),
                "matched_detection_score": maybe_float(detection_meta["matched_detection_score"]),
                "tracked_points": int(state["mask"].sum()) if len(state["mask"]) else 0,
                "projected_points": projected_points,
                "failure_reason": "" if pred_box_2d is not None else ("lost" if state.get("status") == "lost" else "too_few_projected_points"),
            }
            frame_rows.append(row)
            track_frame_rows.append(row)

        track_rows.append(
            summarize_track(
                frame_rows=track_frame_rows,
                precision_thresholds=args.precision_thresholds,
                success_thresholds=args.success_thresholds,
            )
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
                args.precision_thresholds,
                args.success_thresholds,
            ),
            f"tracks_ge_{args.report_min_track_length}": summarize_subset(
                filtered_frame_rows,
                filtered_track_rows,
                args.precision_thresholds,
                args.success_thresholds,
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
            "all_tracks": summarize_subset(
                class_frame_rows,
                class_track_rows,
                args.precision_thresholds,
                args.success_thresholds,
            ),
            f"tracks_ge_{args.report_min_track_length}": summarize_subset(
                class_filtered_frame_rows,
                class_filtered_track_rows,
                args.precision_thresholds,
                args.success_thresholds,
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
        "status_transition",
        "bad_update",
        "bad_update_reason",
        "jump_xy",
        "support_height",
        "support_count",
        "pred_valid",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "gt_bbox_height_px",
        "gt_occlusion",
        "gt_truncation",
        "iou_2d",
        "center_error_px",
        "used_detection",
        "used_box_overlap",
        "matched_detection_found",
        "matched_detection_distance",
        "matched_detection_score",
        "tracked_points",
        "projected_points",
        "failure_reason",
    ]
    track_fieldnames = sorted({key for row in track_rows for key in row.keys()})
    save_csv(run_dir / "per_frame.csv", frame_rows, frame_fieldnames)
    save_csv(run_dir / "per_track.csv", track_rows, track_fieldnames)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved evaluation outputs to {run_dir}")
    print(json.dumps(summary["reports"]["all_tracks"]["frame_metrics"], indent=2))


if __name__ == "__main__":
    main()
