import argparse
import csv
import json
import sys
import time
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
from scripts.eval_kitti_sot_3d import (  # noqa: E402
    build_pred_box,
    estimate_accuracy_3d,
    estimate_overlap_3d,
)
from scripts.kitti_tracker_test import (  # noqa: E402
    TRACKER_PRESETS,
    canonical_label,
    label_to_name,
    load_detection_boxes,
    load_xyz,
    parse_labels,
    record_to_lidar_box,
    resolve_tracker_param,
)
from utonia import UtoniaTracker  # noqa: E402


DEFAULT_SEQUENCES = ["0019", "0020"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root.",
    )
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--classes", nargs="+", default=["Car", "Pedestrian"])
    parser.add_argument(
        "--detections-root",
        default=str(REPO_ROOT / "data" / "detections" / "pointpillar" / "npz"),
    )
    parser.add_argument(
        "--high-score-thresh",
        type=float,
        default=0.7,
        help="Detector score threshold considered reliable enough to keep a track detector-updated.",
    )
    parser.add_argument(
        "--low-score-thresh",
        type=float,
        default=0.3,
        help="Lower threshold used only to diagnose whether a miss had weak detector evidence.",
    )
    parser.add_argument("--car-match-radius", type=float, default=4.0)
    parser.add_argument("--pedestrian-match-radius", type=float, default=1.5)
    parser.add_argument("--max-cases-per-class", type=int, default=8)
    parser.add_argument("--max-gap-frames", type=int, default=5)
    parser.add_argument(
        "--require-low-score-miss",
        action="store_true",
        help="Select only gaps where the first missed frame also has no matching low-score detection.",
    )
    parser.add_argument("--local-crop-radius", type=float, default=8.0)
    parser.add_argument("--local-crop-min-points", type=int, default=2048)
    parser.add_argument("--car-gate-radius", type=float, default=12.0)
    parser.add_argument("--pedestrian-gate-radius", type=float, default=2.5)
    parser.add_argument("--fallback-sim-thresh", type=float, default=0.55)
    parser.add_argument("--fallback-min-points-car", type=int, default=32)
    parser.add_argument("--fallback-min-points-pedestrian", type=int, default=8)
    parser.add_argument("--run-name", default="utonia_detector_gap_recovery_pointpillar_score07")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data" / "detector_gap_recovery"))
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def tracker_kwargs(args, class_name: str) -> dict:
    gate = args.car_gate_radius if class_name == "Car" else args.pedestrian_gate_radius
    return {
        "gate_radius": resolve_tracker_param(gate, class_name, "gate_radius"),
        "cluster_radius": resolve_tracker_param(None, class_name, "cluster_radius"),
        "init_points": resolve_tracker_param(None, class_name, "init_points"),
        "min_points": resolve_tracker_param(None, class_name, "min_points"),
        "box_height_filter_ratio": resolve_tracker_param(None, class_name, "box_height_filter_ratio"),
    }


def match_radius(args, class_name: str) -> float:
    return args.pedestrian_match_radius if class_name == "Pedestrian" else args.car_match_radius


def fallback_min_points(args, class_name: str) -> int:
    if class_name == "Pedestrian":
        return args.fallback_min_points_pedestrian
    return args.fallback_min_points_car


def detection_match(
    detections_root: Path,
    sequence: str,
    frame_id: int,
    gt_box: np.ndarray,
    class_name: str,
    score_thresh: float,
    radius: float,
) -> dict:
    boxes, scores, labels = load_detection_boxes(
        detections_root / sequence / f"{frame_id:06d}.npz",
        score_thresh,
    )
    if len(boxes) == 0:
        return {
            "matched": False,
            "box": None,
            "score": None,
            "distance": None,
            "iou_3d": None,
            "num_class_detections": 0,
        }
    keep = np.array([label_to_name(label) == class_name for label in labels], dtype=bool)
    if not keep.any():
        return {
            "matched": False,
            "box": None,
            "score": None,
            "distance": None,
            "iou_3d": None,
            "num_class_detections": 0,
        }
    boxes = boxes[keep]
    scores = scores[keep]
    distances = np.linalg.norm(boxes[:, :2] - gt_box[None, :2], axis=1)
    best = int(np.argmin(distances))
    iou_3d = estimate_overlap_3d(boxes[best], gt_box)
    matched = bool(distances[best] <= radius and iou_3d > 0.01)
    return {
        "matched": matched,
        "box": boxes[best] if matched else None,
        "score": float(scores[best]),
        "distance": float(distances[best]),
        "iou_3d": float(iou_3d),
        "num_class_detections": int(len(boxes)),
    }


def group_gt_tracks(records: list[dict], classes: set[str]) -> dict[tuple[str, int], list[dict]]:
    tracks = defaultdict(list)
    for record in records:
        class_name = canonical_label(record["type"])
        if record["track_id"] < 0 or class_name not in classes:
            continue
        tracks[(class_name, record["track_id"])].append(record)
    return {key: sorted(value, key=lambda row: row["frame"]) for key, value in tracks.items()}


def find_gap_cases(args, data_root: Path, detections_root: Path) -> list[dict]:
    classes = set(args.classes)
    selected = []
    selected_counts = defaultdict(int)
    for sequence in args.sequences:
        calib = Calibration(data_root / "calib" / f"{sequence}.txt")
        records = parse_labels(data_root / "label_02" / f"{sequence}.txt")
        tracks = group_gt_tracks(records, classes)
        for (class_name, track_id), track_records in sorted(tracks.items()):
            if selected_counts[class_name] >= args.max_cases_per_class:
                continue
            by_frame = {record["frame"]: record for record in track_records}
            frames = sorted(by_frame)
            for prev_frame, gap_start in zip(frames[:-1], frames[1:], strict=False):
                if gap_start != prev_frame + 1:
                    continue
                prev_gt = record_to_lidar_box(by_frame[prev_frame], calib).astype(np.float32)
                curr_gt = record_to_lidar_box(by_frame[gap_start], calib).astype(np.float32)
                prev_match = detection_match(
                    detections_root,
                    sequence,
                    prev_frame,
                    prev_gt,
                    class_name,
                    args.high_score_thresh,
                    match_radius(args, class_name),
                )
                if not prev_match["matched"]:
                    continue
                curr_high = detection_match(
                    detections_root,
                    sequence,
                    gap_start,
                    curr_gt,
                    class_name,
                    args.high_score_thresh,
                    match_radius(args, class_name),
                )
                if curr_high["matched"]:
                    continue
                curr_low = detection_match(
                    detections_root,
                    sequence,
                    gap_start,
                    curr_gt,
                    class_name,
                    args.low_score_thresh,
                    match_radius(args, class_name),
                )
                if args.require_low_score_miss and curr_low["matched"]:
                    continue
                gap_frames = []
                for frame_id in frames[frames.index(gap_start) :]:
                    if frame_id - gap_start >= args.max_gap_frames:
                        break
                    gt_box = record_to_lidar_box(by_frame[frame_id], calib).astype(np.float32)
                    high = detection_match(
                        detections_root,
                        sequence,
                        frame_id,
                        gt_box,
                        class_name,
                        args.high_score_thresh,
                        match_radius(args, class_name),
                    )
                    if high["matched"]:
                        break
                    low = detection_match(
                        detections_root,
                        sequence,
                        frame_id,
                        gt_box,
                        class_name,
                        args.low_score_thresh,
                        match_radius(args, class_name),
                    )
                    gap_frames.append(
                        {
                            "frame": frame_id,
                            "gt_box": gt_box,
                            "low_match": low,
                        }
                    )
                if not gap_frames:
                    continue
                selected.append(
                    {
                        "sequence": sequence,
                        "class": class_name,
                        "track_id": track_id,
                        "init_frame": prev_frame,
                        "init_box": prev_match["box"],
                        "init_gt_box": prev_gt,
                        "init_match_score": prev_match["score"],
                        "init_match_distance": prev_match["distance"],
                        "init_match_iou_3d": prev_match["iou_3d"],
                        "gap_start": gap_start,
                        "first_low_score_match": curr_low["matched"],
                        "first_low_score": curr_low["score"],
                        "first_low_distance": curr_low["distance"],
                        "first_low_iou_3d": curr_low["iou_3d"],
                        "gap_frames": gap_frames,
                    }
                )
                selected_counts[class_name] += 1
                break
    return selected


def mean_similarity(state: dict) -> float | None:
    sim = np.asarray(state.get("sim", []), dtype=np.float32)
    mask = np.asarray(state.get("mask", []), dtype=bool)
    if sim.size == 0 or mask.size == 0 or int(mask.sum()) == 0:
        return None
    return float(sim[mask].mean())


def run_case(args, data_root: Path, shared_model, case: dict) -> tuple[list[dict], dict]:
    sequence = case["sequence"]
    class_name = case["class"]
    velodyne_dir = data_root / "velodyne" / sequence
    kwargs = tracker_kwargs(args, class_name)
    tracker = UtoniaTracker(
        model=shared_model,
        mode="local_crop",
        init_radius=1.6,
        cluster_radius=kwargs["cluster_radius"],
        gate_radius=kwargs["gate_radius"],
        local_crop_radius=args.local_crop_radius,
        local_crop_min_points=args.local_crop_min_points,
        init_points=kwargs["init_points"],
        min_points=kwargs["min_points"],
        box_height_filter_ratio=kwargs["box_height_filter_ratio"],
        lost_height_ratio=0.3,
        lost_bad_frames=4,
    )
    init_state = tracker.initialize_from_box(
        load_xyz(velodyne_dir / f"{case['init_frame']:06d}.bin"),
        case["init_box"],
    )
    init_centroid = np.asarray(init_state["centroid"], dtype=np.float32)
    rows = []
    accepted_frames = 0
    for frame_info in case["gap_frames"]:
        frame_id = frame_info["frame"]
        state = tracker.step(load_xyz(velodyne_dir / f"{frame_id:06d}.bin"))
        pred_box = build_pred_box(
            case["init_box"],
            init_centroid,
            np.asarray(state["centroid"], dtype=np.float32),
        )
        gt_box = frame_info["gt_box"]
        support_count = int(state.get("support_count", 0))
        mean_sim = mean_similarity(state)
        jump_xy = state.get("jump_xy")
        accepted = (
            support_count >= fallback_min_points(args, class_name)
            and mean_sim is not None
            and mean_sim >= args.fallback_sim_thresh
            and (jump_xy is None or float(jump_xy) <= kwargs["gate_radius"])
            and state.get("status") == "active"
        )
        accepted_frames += int(accepted)
        rows.append(
            {
                "sequence": sequence,
                "track_id": case["track_id"],
                "class": class_name,
                "init_frame": case["init_frame"],
                "frame": frame_id,
                "gap_index": frame_id - case["gap_start"],
                "init_match_score": case["init_match_score"],
                "init_match_distance": case["init_match_distance"],
                "init_match_iou_3d": case["init_match_iou_3d"],
                "low_score_match": int(frame_info["low_match"]["matched"]),
                "low_score": frame_info["low_match"]["score"],
                "low_distance": frame_info["low_match"]["distance"],
                "low_iou_3d": frame_info["low_match"]["iou_3d"],
                "utonia_accepted": int(accepted),
                "utonia_iou_3d": estimate_overlap_3d(pred_box, gt_box),
                "utonia_accuracy_3d": estimate_accuracy_3d(pred_box, gt_box),
                "support_count": support_count,
                "mean_similarity": mean_sim,
                "jump_xy": jump_xy,
                "status": state.get("status"),
                "bad_update": int(state.get("bad_update", False)),
                "bad_update_reason": state.get("bad_update_reason", ""),
            }
        )
    summary = {
        "sequence": sequence,
        "track_id": case["track_id"],
        "class": class_name,
        "init_frame": case["init_frame"],
        "gap_start": case["gap_start"],
        "gap_frames": len(rows),
        "accepted_frames": accepted_frames,
        "mean_utonia_iou_3d": float(np.mean([row["utonia_iou_3d"] for row in rows])) if rows else 0.0,
        "median_utonia_accuracy_3d": float(np.median([row["utonia_accuracy_3d"] for row in rows])) if rows else None,
        "low_score_match_frames": sum(int(row["low_score_match"]) for row in rows),
        "first_low_score_match": int(case["first_low_score_match"]),
        "first_low_score": case["first_low_score"],
        "first_low_distance": case["first_low_distance"],
        "first_low_iou_3d": case["first_low_iou_3d"],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    detections_root = Path(args.detections_root)
    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = find_gap_cases(args, data_root, detections_root)
    print(f"Selected {len(cases)} detector-gap cases", flush=True)

    shared_tracker = UtoniaTracker(mode="local_crop")
    shared_tracker.build_model()
    shared_model = shared_tracker.model

    started = time.monotonic()
    frame_rows = []
    case_rows = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] seq={case['sequence']} track={case['track_id']} "
            f"class={case['class']} init={case['init_frame']} gap={case['gap_start']} "
            f"frames={len(case['gap_frames'])}",
            flush=True,
        )
        rows, summary = run_case(args, data_root, shared_model, case)
        frame_rows.extend(rows)
        case_rows.append(summary)

    write_csv(output_dir / "per_frame.csv", frame_rows)
    write_csv(output_dir / "per_case.csv", case_rows)

    by_class = {}
    for class_name in sorted(set(row["class"] for row in frame_rows)):
        class_rows = [row for row in frame_rows if row["class"] == class_name]
        by_class[class_name] = {
            "frames": len(class_rows),
            "accepted_rate": float(np.mean([row["utonia_accepted"] for row in class_rows])) if class_rows else 0.0,
            "mean_iou_3d": float(np.mean([row["utonia_iou_3d"] for row in class_rows])) if class_rows else 0.0,
            "median_accuracy_3d": float(np.median([row["utonia_accuracy_3d"] for row in class_rows])) if class_rows else None,
            "low_score_match_rate": float(np.mean([row["low_score_match"] for row in class_rows])) if class_rows else 0.0,
            "mean_support_count": float(np.mean([row["support_count"] for row in class_rows])) if class_rows else 0.0,
            "mean_similarity": float(np.mean([row["mean_similarity"] for row in class_rows if row["mean_similarity"] is not None]))
            if any(row["mean_similarity"] is not None for row in class_rows)
            else None,
        }
    summary = {
        "config": vars(args),
        "num_cases": len(cases),
        "num_frames": len(frame_rows),
        "runtime_seconds": time.monotonic() - started,
        "overall": {
            "accepted_rate": float(np.mean([row["utonia_accepted"] for row in frame_rows])) if frame_rows else 0.0,
            "mean_iou_3d": float(np.mean([row["utonia_iou_3d"] for row in frame_rows])) if frame_rows else 0.0,
            "median_accuracy_3d": float(np.median([row["utonia_accuracy_3d"] for row in frame_rows])) if frame_rows else None,
            "low_score_match_rate": float(np.mean([row["low_score_match"] for row in frame_rows])) if frame_rows else 0.0,
        },
        "by_class": by_class,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Finished in {format_duration(summary['runtime_seconds'])}; saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
