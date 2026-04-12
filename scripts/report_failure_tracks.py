import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "data" / "eval_sot" / "full_allseq_gt_pointpillar"),
        help="Evaluation run directory containing per_track.csv and per_frame.csv.",
    )
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root used for command generation.",
    )
    parser.add_argument(
        "--top-k-per-sequence",
        type=int,
        default=1,
        help="How many failure tracks to keep per sequence.",
    )
    parser.add_argument(
        "--report-min-track-length",
        type=int,
        default=10,
        help="Ignore shorter tracks when selecting representative failures.",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=20,
        help="Frames of context before the failure frame in the suggested command.",
    )
    parser.add_argument(
        "--watch-frames",
        type=int,
        default=80,
        help="Suggested max frames to visualize in kitti_tracker_test.py.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str):
    return None if value in {None, ""} else float(value)


def maybe_int(value: str):
    return None if value in {None, ""} else int(float(value))


def failure_sort_key(track: dict):
    first_lost = maybe_int(track["first_lost_frame"])
    num_frames = int(track["num_frames"])
    loss_ratio = first_lost / num_frames if first_lost is not None and num_frames > 0 else 1.0
    mean_iou = float(track["mean_iou_all"])
    valid_rate = float(track["valid_rate"])
    return (
        0 if first_lost is not None else 1,
        loss_ratio,
        mean_iou,
        valid_rate,
        -int(track["track_length"]),
    )


def pick_failure_frame(frame_rows: list[dict]) -> tuple[int, str]:
    for row in frame_rows:
        if row["status"] == "lost":
            reason = row["bad_update_reason"] or row["failure_reason"] or "lost"
            return int(row["frame_id"]), reason
    bad_rows = [row for row in frame_rows if row["bad_update"] == "1"]
    if bad_rows:
        row = bad_rows[0]
        return int(row["frame_id"]), row["bad_update_reason"] or "bad_update"
    row = min(frame_rows, key=lambda current: float(current["iou_2d"]))
    return int(row["frame_id"]), row["failure_reason"] or "low_iou"


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    per_track = read_csv(run_dir / "per_track.csv")
    per_frame = read_csv(run_dir / "per_frame.csv")

    config = None
    import json

    with (run_dir / "summary.json").open() as handle:
        config = json.load(handle)["config"]

    eligible_tracks = [
        row for row in per_track if int(row["track_length"]) >= args.report_min_track_length
    ]
    frames_by_key = defaultdict(list)
    for row in per_frame:
        key = (row["sequence"], row["track_id"])
        frames_by_key[key].append(row)
    for rows in frames_by_key.values():
        rows.sort(key=lambda row: int(row["frame_id"]))

    selected = []
    by_sequence = defaultdict(list)
    for row in eligible_tracks:
        by_sequence[row["sequence"]].append(row)

    for sequence, tracks in sorted(by_sequence.items()):
        ranked = sorted(tracks, key=failure_sort_key)
        for track in ranked[: args.top_k_per_sequence]:
            key = (track["sequence"], track["track_id"])
            frame_rows = frames_by_key[key]
            failure_frame, failure_reason = pick_failure_frame(frame_rows)
            first_frame = int(frame_rows[0]["frame_id"])
            watch_start = max(first_frame, failure_frame - args.context_before)
            init_source = config["init_source"]
            update_source = config["update_source"]
            command = (
                f'python scripts/kitti_tracker_test.py "{Path(args.data_root) / "velodyne" / sequence}" '
                f'--track-id {track["track_id"]} '
                f'--init-source {init_source} '
                f'--update-source {update_source if update_source != "none" else init_source} '
                f'--start-frame {watch_start} '
                f'--max-frames {args.watch_frames}'
            )
            selected.append(
                {
                    "sequence": sequence,
                    "track_id": int(track["track_id"]),
                    "class": track["class"],
                    "track_length": int(track["track_length"]),
                    "mean_iou_all": float(track["mean_iou_all"]),
                    "valid_rate": float(track["valid_rate"]),
                    "first_lost_frame": maybe_int(track["first_lost_frame"]),
                    "failure_frame": failure_frame,
                    "failure_reason": failure_reason,
                    "watch_start_frame": watch_start,
                    "watch_frames": args.watch_frames,
                    "command": command,
                }
            )

    selected.sort(key=lambda row: (row["sequence"], row["failure_frame"], row["track_id"]))

    csv_path = run_dir / "failure_tracks.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    md_path = run_dir / "failure_tracks.md"
    lines = [
        f"# Failure Tracks\n",
        f"Run: `{run_dir.name}`\n",
        f"Top {args.top_k_per_sequence} failure track(s) per sequence, filtered to track length >= {args.report_min_track_length}.\n",
        "",
    ]
    for row in selected:
        lines.extend(
            [
                f"## Sequence {row['sequence']}\n",
                f"- Track: `{row['track_id']}` ({row['class']})\n",
                f"- Track length: `{row['track_length']}`\n",
                f"- Mean IoU: `{row['mean_iou_all']:.4f}`\n",
                f"- Valid rate: `{row['valid_rate']:.4f}`\n",
                f"- First lost frame: `{row['first_lost_frame']}`\n",
                f"- Suggested failure frame: `{row['failure_frame']}`\n",
                f"- Failure reason: `{row['failure_reason']}`\n",
                f"- Watch command:\n",
                "```bash",
                row["command"],
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    for row in selected:
        print(
            f"{row['sequence']}: track_id={row['track_id']} class={row['class']} "
            f"failure_frame={row['failure_frame']} reason={row['failure_reason']}"
        )


if __name__ == "__main__":
    main()
