import argparse
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "data" / "eval_sot_3d" / "diagnostic_kitti3d_strict_gt_none_20260519"),
        help="3D SOT run directory containing per_track.csv and official failure report.",
    )
    parser.add_argument(
        "--failure-csv",
        default=None,
        help="Failure CSV. Defaults to <run-dir>/official_failure_tracks_3d.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to <run-dir>/official_diagnostic_subset.csv.",
    )
    parser.add_argument("--classes", nargs="+", default=["Car", "Pedestrian"])
    parser.add_argument("--min-track-length", type=int, default=10)
    parser.add_argument("--controls-per-failure", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def is_lost(track: dict) -> bool:
    return track.get("first_lost_frame") not in {None, ""}


def success_value(track: dict) -> float:
    if track.get("success"):
        return float(track["success"])
    if track.get("success_3d"):
        return float(track["success_3d"])
    return 0.0


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    failure_csv = Path(args.failure_csv) if args.failure_csv else run_dir / "official_failure_tracks_3d.csv"
    output = Path(args.output) if args.output else run_dir / "official_diagnostic_subset.csv"
    classes = set(args.classes)

    failures = [
        row
        for row in read_csv(failure_csv)
        if row["class"] in classes and int(row["track_length"]) >= args.min_track_length
    ]
    failure_keys = {(row["sequence"], row["track_id"]) for row in failures}
    failure_counts = Counter(row["class"] for row in failures)

    per_track = [
        row
        for row in read_csv(run_dir / "per_track.csv")
        if row["class"] in classes
        and int(row["track_length"]) >= args.min_track_length
        and (row["sequence"], row["track_id"]) not in failure_keys
        and not is_lost(row)
    ]

    selected = []
    for row in failures:
        selected.append(
            {
                "sequence": row["sequence"],
                "track_id": row["track_id"],
                "class": row["class"],
                "subset_role": "failure",
            }
        )

    for class_name, failure_count in sorted(failure_counts.items()):
        needed = failure_count * args.controls_per_failure
        class_controls = [
            row for row in per_track if row["class"] == class_name
        ]
        class_controls = sorted(
            class_controls,
            key=lambda row: (success_value(row), int(row["track_length"])),
            reverse=True,
        )
        for row in class_controls[:needed]:
            selected.append(
                {
                    "sequence": row["sequence"],
                    "track_id": row["track_id"],
                    "class": row["class"],
                    "subset_role": "control",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "track_id", "class", "subset_role"],
        )
        writer.writeheader()
        writer.writerows(selected)

    selected_counts = Counter((row["class"], row["subset_role"]) for row in selected)
    print(f"Saved {output}")
    print(f"Selected {len(selected)} tracks")
    print(dict(sorted(selected_counts.items())))


if __name__ == "__main__":
    main()
