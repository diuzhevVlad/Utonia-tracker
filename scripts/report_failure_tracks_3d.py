import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="3D SOT evaluation run directory containing per_track.csv, per_frame.csv, and summary.json.",
    )
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root used for command generation.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-k-per-class", type=int, default=5)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Car", "Pedestrian"],
        choices=["Car", "Pedestrian", "Van", "Cyclist"],
        help="Classes to include. Defaults to official KITTI tracking classes.",
    )
    parser.add_argument("--report-min-track-length", type=int, default=10)
    parser.add_argument("--context-before", type=int, default=20)
    parser.add_argument("--watch-frames", type=int, default=90)
    parser.add_argument("--drift-distance", type=float, default=2.0)
    parser.add_argument("--low-overlap", type=float, default=0.1)
    parser.add_argument(
        "--output-prefix",
        default="failure_tracks_3d",
        help="Output file prefix inside the run directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def maybe_int(value: str | None) -> int | None:
    return None if value in {None, ""} else int(float(value))


def metric_columns(config: dict, frame_rows: list[dict]) -> tuple[str, str, str]:
    metric_space = config.get("metric_space", "bev")
    if metric_space == "3d" and frame_rows and "iou_3d" in frame_rows[0]:
        return "iou_3d", "accuracy_3d", "3d"
    if frame_rows and "bev_overlap" in frame_rows[0]:
        return "bev_overlap", "bev_accuracy", "bev"
    raise RuntimeError("Could not resolve metric columns from per_frame.csv")


def track_metric(track: dict, generic: str, specific: str) -> float:
    if generic in track and track[generic] != "":
        return float(track[generic])
    if specific in track and track[specific] != "":
        return float(track[specific])
    return 0.0


def failure_score(track: dict, overlap_prefix: str) -> tuple:
    first_lost = maybe_int(track.get("first_lost_frame"))
    num_frames = int(track["num_frames"])
    lost_position = first_lost / max(num_frames, 1) if first_lost is not None else 1.0
    mean_accuracy = track_metric(track, "mean_accuracy", f"mean_{overlap_prefix}_accuracy")
    success = track_metric(track, "success", f"success_{overlap_prefix}")
    precision = track_metric(track, "precision", f"precision_{overlap_prefix}")
    return (
        0 if first_lost is not None else 1,
        lost_position,
        success,
        precision,
        -mean_accuracy,
        -int(track["track_length"]),
    )


def pick_failure_frame(
    frame_rows: list[dict],
    overlap_col: str,
    accuracy_col: str,
    drift_distance: float,
    low_overlap: float,
) -> tuple[int, str, float, float]:
    for row in frame_rows:
        if row["status"] == "lost":
            return (
                int(row["frame_id"]),
                "lost",
                float(row[overlap_col]),
                float(row[accuracy_col]),
            )

    drift_rows = [
        row for row in frame_rows if float(row[accuracy_col]) >= drift_distance
    ]
    if drift_rows:
        row = drift_rows[0]
        return (
            int(row["frame_id"]),
            "drift",
            float(row[overlap_col]),
            float(row[accuracy_col]),
        )

    low_overlap_rows = [
        row for row in frame_rows if float(row[overlap_col]) <= low_overlap
    ]
    if low_overlap_rows:
        row = low_overlap_rows[0]
        return (
            int(row["frame_id"]),
            "low_overlap",
            float(row[overlap_col]),
            float(row[accuracy_col]),
        )

    row = min(frame_rows, key=lambda current: float(current[overlap_col]))
    return (
        int(row["frame_id"]),
        "min_overlap",
        float(row[overlap_col]),
        float(row[accuracy_col]),
    )


def build_eval_command(
    data_root: Path,
    sequence: str,
    track_id: str,
    config: dict,
    classes: list[str],
    start_frame: int,
    watch_frames: int,
) -> str:
    return (
        "conda run -n utonia python scripts/eval_kitti_sot_3d.py "
        f"--data-root {data_root} "
        f"--sequences {sequence} "
        f"--track-list-csv <(printf 'sequence,track_id\\n{sequence},{track_id}\\n') "
        f"--classes {' '.join(classes)} "
        f"--init-source {config.get('init_source', 'gt')} "
        f"--update-source {config.get('update_source', 'none')} "
        f"--init-policy {config.get('init_policy', 'immediate')} "
        f"--metric-space {config.get('metric_space', '3d')} "
        f"--max-track-frames {watch_frames}"
    )


def build_watch_command(
    data_root: Path,
    sequence: str,
    track_id: str,
    config: dict,
    start_frame: int,
    watch_frames: int,
) -> str:
    init_source = config.get("init_source", "gt")
    update_source = config.get("update_source", "none")
    return (
        "conda run -n utonia python scripts/kitti_tracker_test.py "
        f"{data_root / 'velodyne' / sequence} "
        f"--track-id {track_id} "
        f"--init-source {init_source} "
        f"--update-source {update_source} "
        f"--start-frame {start_frame} "
        f"--max-frames {watch_frames} "
        "--no-spawn"
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    per_track = read_csv(run_dir / "per_track.csv")
    per_frame = read_csv(run_dir / "per_frame.csv")
    config = json.loads((run_dir / "summary.json").read_text())["config"]
    overlap_col, accuracy_col, metric_prefix = metric_columns(config, per_frame)

    frames_by_key = defaultdict(list)
    for row in per_frame:
        frames_by_key[(row["sequence"], row["track_id"])].append(row)
    for rows in frames_by_key.values():
        rows.sort(key=lambda row: int(row["frame_id"]))

    allowed_classes = set(args.classes)
    eligible_tracks = [
        row
        for row in per_track
        if int(row["track_length"]) >= args.report_min_track_length
        and row["class"] in allowed_classes
    ]
    if not eligible_tracks:
        raise RuntimeError(
            f"No eligible tracks found for classes {sorted(allowed_classes)} "
            f"with track length >= {args.report_min_track_length}"
        )
    ranked = sorted(
        eligible_tracks,
        key=lambda row: failure_score(row, metric_prefix),
    )

    selected_keys = set()
    selected = []
    for row in ranked[: args.top_k]:
        selected_keys.add((row["sequence"], row["track_id"]))
        selected.append(row)

    by_class = defaultdict(list)
    for row in eligible_tracks:
        by_class[row["class"]].append(row)
    for class_name, rows in by_class.items():
        for row in sorted(rows, key=lambda current: failure_score(current, metric_prefix))[: args.top_k_per_class]:
            key = (row["sequence"], row["track_id"])
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(row)

    report_rows = []
    class_counter = Counter()
    reason_counter = Counter()
    for track in selected:
        key = (track["sequence"], track["track_id"])
        frame_rows = frames_by_key[key]
        failure_frame, reason, overlap, accuracy = pick_failure_frame(
            frame_rows,
            overlap_col,
            accuracy_col,
            drift_distance=args.drift_distance,
            low_overlap=args.low_overlap,
        )
        first_frame = int(frame_rows[0]["frame_id"])
        watch_start = max(first_frame, failure_frame - args.context_before)
        class_counter[track["class"]] += 1
        reason_counter[reason] += 1
        report_rows.append(
            {
                "sequence": track["sequence"],
                "track_id": int(track["track_id"]),
                "class": track["class"],
                "track_length": int(track["track_length"]),
                "num_frames": int(track["num_frames"]),
                "success": track_metric(track, "success", f"success_{metric_prefix}"),
                "precision": track_metric(track, "precision", f"precision_{metric_prefix}"),
                "mean_overlap": track_metric(track, "mean_overlap", f"mean_{metric_prefix}_overlap"),
                "mean_accuracy": track_metric(track, "mean_accuracy", f"mean_{metric_prefix}_accuracy"),
                "median_accuracy": track_metric(track, "median_accuracy", f"median_{metric_prefix}_accuracy"),
                "lost_rate": float(track["lost_rate"]),
                "min_raw_box_support_points": maybe_int(track.get("min_raw_box_support_points")),
                "median_raw_box_support_points": maybe_float(track.get("median_raw_box_support_points")),
                "min_filtered_box_support_points": maybe_int(track.get("min_filtered_box_support_points")),
                "median_filtered_box_support_points": maybe_float(track.get("median_filtered_box_support_points")),
                "min_target_support_count": maybe_int(track.get("min_target_support_count")),
                "median_target_support_count": maybe_float(track.get("median_target_support_count")),
                "pre_lost_window_min_support": maybe_int(track.get("pre_lost_window_min_support")),
                "pre_lost_window_min_target_support": maybe_int(track.get("pre_lost_window_min_target_support")),
                "bad_update_frames": maybe_int(track.get("bad_update_frames")),
                "sparse_mode_frames": maybe_int(track.get("sparse_mode_frames")),
                "recovered_frames": maybe_int(track.get("recovered_frames")),
                "first_lost_frame": maybe_int(track.get("first_lost_frame")),
                "failure_frame": failure_frame,
                "failure_reason": reason,
                "failure_overlap": overlap,
                "failure_accuracy": accuracy,
                "watch_start_frame": watch_start,
                "watch_frames": args.watch_frames,
                "watch_command": build_watch_command(
                    data_root,
                    track["sequence"],
                    track["track_id"],
                    config,
                    watch_start,
                    args.watch_frames,
                ),
                "limited_eval_command": build_eval_command(
                    data_root,
                    track["sequence"],
                    track["track_id"],
                    config,
                    args.classes,
                    watch_start,
                    args.watch_frames,
                ),
            }
        )

    report_rows.sort(
        key=lambda row: (
            row["sequence"],
            row["failure_frame"],
            row["track_id"],
        )
    )

    csv_path = run_dir / f"{args.output_prefix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    md_path = run_dir / f"{args.output_prefix}.md"
    lines = [
        "# 3D SOT Failure Tracks",
        "",
        f"Run: `{run_dir.name}`",
        f"Metric columns: `{overlap_col}`, `{accuracy_col}`",
        f"Classes: `{args.classes}`",
        f"Selected tracks: `{len(report_rows)}`",
        f"Class counts: `{dict(sorted(class_counter.items()))}`",
        f"Failure reasons: `{dict(sorted(reason_counter.items()))}`",
        "",
    ]
    for row in report_rows:
        lines.extend(
            [
                f"## {row['sequence']} track {row['track_id']} ({row['class']})",
                "",
                f"- Length: `{row['track_length']}` frames",
                f"- Success: `{row['success']:.2f}`",
                f"- Precision: `{row['precision']:.2f}`",
                f"- Mean overlap: `{row['mean_overlap']:.4f}`",
                f"- Mean accuracy: `{row['mean_accuracy']:.4f}` m",
                f"- Min filtered GT-box support: `{row['min_filtered_box_support_points']}`",
                f"- Median filtered GT-box support: `{row['median_filtered_box_support_points']}`",
                f"- Pre-lost min GT-box support: `{row['pre_lost_window_min_support']}`",
                f"- Pre-lost min target support: `{row['pre_lost_window_min_target_support']}`",
                f"- Bad-update frames: `{row['bad_update_frames']}`",
                f"- Sparse-mode frames: `{row['sparse_mode_frames']}`",
                f"- Recovered frames: `{row['recovered_frames']}`",
                f"- First lost frame: `{row['first_lost_frame']}`",
                f"- Failure frame: `{row['failure_frame']}`",
                f"- Failure reason: `{row['failure_reason']}`",
                f"- Failure overlap: `{row['failure_overlap']:.4f}`",
                f"- Failure accuracy: `{row['failure_accuracy']:.4f}` m",
                "- Watch command:",
                "```bash",
                row["watch_command"],
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print(f"Class counts: {dict(sorted(class_counter.items()))}")
    print(f"Failure reasons: {dict(sorted(reason_counter.items()))}")
    for row in report_rows[: args.top_k]:
        print(
            f"{row['sequence']} track={row['track_id']} class={row['class']} "
            f"reason={row['failure_reason']} frame={row['failure_frame']} "
            f"success={row['success']:.2f} precision={row['precision']:.2f}"
        )


if __name__ == "__main__":
    main()
