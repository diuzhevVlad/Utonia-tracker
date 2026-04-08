import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "basic_kalman",
        "description": "Geometry baseline with Kalman motion, BEV IoU, class thresholds, and birth suppression.",
        "args": ["--basic"],
    },
    {
        "name": "utonia_default",
        "description": "Full cropped Utonia tracker with appearance, recovery, BEV IoU, and Kalman motion.",
        "args": [],
    },
    {
        "name": "utonia_no_appearance",
        "description": "Utonia tracker with appearance matching disabled and recovery disabled.",
        "args": ["--appearance-weight", "0", "--disable-recovery"],
    },
    {
        "name": "utonia_no_recovery",
        "description": "Utonia tracker without point-level recovery for lost tracks.",
        "args": ["--disable-recovery"],
    },
    {
        "name": "utonia_no_bev_iou",
        "description": "Utonia tracker without BEV IoU in the association cost.",
        "args": ["--disable-bev-iou"],
    },
    {
        "name": "utonia_velocity_motion",
        "description": "Utonia tracker with constant-velocity motion instead of Kalman prediction.",
        "args": ["--motion-model", "velocity"],
    },
]


METRIC_COLUMNS = [
    "car_HOTA",
    "car_AssA",
    "car_IDF1",
    "car_MOTA",
    "car_IDSW",
    "ped_HOTA",
    "ped_AssA",
    "ped_IDF1",
    "ped_MOTA",
    "ped_IDSW",
    "profile_total_ms",
    "profile_encode_ms",
    "profile_crop_ms",
    "profile_box_features_ms",
    "profile_matching_ms",
    "profile_recovery_ms",
    "appearance_skipped",
    "num_recovered_tracks",
    "num_suppressed_spawns",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_dir",
        help="KITTI tracking training dir, e.g. /path/to/trackkitti/training",
    )
    parser.add_argument(
        "--sequence",
        default="0000",
        help="Single KITTI sequence id used for the quick ablation, default: 0000",
    )
    parser.add_argument(
        "--detections-root",
        default="data/detections/openpcdet_pointpillar",
        help="Root of precomputed detections",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Minimum detection score",
    )
    parser.add_argument(
        "--output-root",
        default="data/kitti_eval/minimal_ablation",
        help="Where to store ablation runs and summaries",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace, experiment: dict) -> Path:
    output_root = Path(args.output_root).resolve()
    tracker_name = experiment["name"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "kitti_mot_evaluate.py"),
        str(Path(args.training_dir).resolve()),
        "--sequences",
        args.sequence,
        "--detections-root",
        args.detections_root,
        "--score-thresh",
        str(args.score_thresh),
        "--classes",
        "car",
        "pedestrian",
        "--tracker-name",
        tracker_name,
        "--output-root",
        str(output_root),
        *experiment["args"],
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    return output_root / "trackers" / "kitti" / "kitti_2d_box_train" / tracker_name


def read_metric_file(path: Path, prefix: str) -> dict[str, float]:
    if not path.is_file():
        return {}
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values = lines[1].split()
    result = {}
    for header, value in zip(headers, values):
        try:
            result[f"{prefix}_{header}"] = float(value)
        except ValueError:
            continue
    return result


def read_profile(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {
        "profile_total_ms": raw.get("total_s", 0.0) * 1000.0,
        "profile_encode_ms": raw.get("encode_frame_s", 0.0) * 1000.0,
        "profile_crop_ms": raw.get("feature_crop_s", 0.0) * 1000.0,
        "profile_box_features_ms": raw.get("box_features_s", 0.0) * 1000.0,
        "profile_matching_ms": raw.get("matching_s", 0.0) * 1000.0,
        "profile_recovery_ms": raw.get("recovery_s", 0.0) * 1000.0,
        "appearance_skipped": raw.get("appearance_skipped", 0.0),
        "num_recovered_tracks": raw.get("num_recovered_tracks", 0.0),
        "num_suppressed_spawns": raw.get("num_suppressed_spawns", 0.0),
    }


def collect_result(experiment: dict, tracker_root: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment": experiment["name"],
        "description": experiment["description"],
    }
    row.update(read_metric_file(tracker_root / "car_summary.txt", "car"))
    row.update(read_metric_file(tracker_root / "pedestrian_summary.txt", "ped"))
    row.update(read_profile(tracker_root / "profile_summary.json"))
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["experiment", "description", *METRIC_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Experiment",
        "Car HOTA",
        "Car IDF1",
        "Ped HOTA",
        "Ped IDF1",
        "Runtime ms",
        "Encode ms",
        "Recovery ms",
        "Recoveries",
        "Appearance skipped",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    f"{row.get('car_HOTA', 0.0):.2f}",
                    f"{row.get('car_IDF1', 0.0):.2f}",
                    f"{row.get('ped_HOTA', 0.0):.2f}",
                    f"{row.get('ped_IDF1', 0.0):.2f}",
                    f"{row.get('profile_total_ms', 0.0):.1f}",
                    f"{row.get('profile_encode_ms', 0.0):.1f}",
                    f"{row.get('profile_recovery_ms', 0.0):.1f}",
                    f"{row.get('num_recovered_tracks', 0.0):.2f}",
                    f"{row.get('appearance_skipped', 0.0):.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Minimal MOT Ablation",
        "",
        f"Sequence: `{args.sequence}`",
        f"Detections: `{args.detections_root}`",
        f"Score threshold: `{args.score_thresh}`",
        "",
        "This is a quick detector-based ablation intended to identify which modules matter most before larger runs.",
        "",
        markdown_table(rows),
        "",
        "## Experiment Notes",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['experiment']}`: {row['description']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        print(f"running {experiment['name']}")
        tracker_root = run_experiment(args, experiment)
        rows.append(collect_result(experiment, tracker_root))

    summary_root = Path(args.output_root).resolve() / "summaries"
    write_csv(rows, summary_root / f"minimal_{args.sequence}.csv")
    write_markdown(rows, summary_root / f"minimal_{args.sequence}.md", args)
    print(f"saved summary to {summary_root / f'minimal_{args.sequence}.md'}")


if __name__ == "__main__":
    main()
