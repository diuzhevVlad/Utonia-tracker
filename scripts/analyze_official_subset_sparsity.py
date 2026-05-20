import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--subset-csv",
        default=str(
            REPO_ROOT
            / "data"
            / "eval_sot_3d"
            / "diagnostic_kitti3d_strict_gt_none_20260519"
            / "official_diagnostic_subset.csv"
        ),
    )
    parser.add_argument("--output-prefix", default="official_sparsity_analysis")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def maybe_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


def maybe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def median(values: list[int | float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def mean(values: list[int | float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.mean(clean))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    subset = {
        (row["sequence"], row["track_id"]): row["subset_role"]
        for row in read_csv(Path(args.subset_csv))
    }
    tracks = [
        row
        for row in read_csv(run_dir / "per_track.csv")
        if (row["sequence"], row["track_id"]) in subset
    ]

    groups = defaultdict(list)
    for row in tracks:
        groups[(subset[(row["sequence"], row["track_id"])], row["class"])].append(row)
        groups[(subset[(row["sequence"], row["track_id"])], "all")].append(row)

    rows = []
    for (role, class_name), group in sorted(groups.items()):
        rows.append(
            {
                "subset_role": role,
                "class": class_name,
                "num_tracks": len(group),
                "lost_tracks": sum(row["first_lost_frame"] != "" for row in group),
                "mean_success_3d": mean([maybe_float(row.get("success_3d") or row.get("success")) for row in group]),
                "mean_precision_3d": mean([maybe_float(row.get("precision_3d") or row.get("precision")) for row in group]),
                "median_init_support": median([maybe_int(row.get("init_support_points")) for row in group]),
                "median_min_filtered_support": median([maybe_int(row.get("min_filtered_box_support_points")) for row in group]),
                "median_pre_lost_support": median([maybe_int(row.get("pre_lost_window_min_support")) for row in group]),
                "median_min_target_support": median([maybe_int(row.get("min_target_support_count")) for row in group]),
            }
        )

    csv_path = run_dir / f"{args.output_prefix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = run_dir / f"{args.output_prefix}.md"
    lines = [
        "# Official Subset Sparsity Analysis",
        "",
        f"Run: `{run_dir.name}`",
        f"Subset: `{args.subset_csv}`",
        "",
        "| Role | Class | Tracks | Lost | Mean Success 3D | Median Init Support | Median Min Filtered Support | Median Pre-Lost Support |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['subset_role']} | {row['class']} | {row['num_tracks']} | {row['lost_tracks']} | "
            f"{fmt(row['mean_success_3d'])} | {fmt(row['median_init_support'])} | "
            f"{fmt(row['median_min_filtered_support'])} | {fmt(row['median_pre_lost_support'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    for row in rows:
        print(
            f"{row['subset_role']} {row['class']}: tracks={row['num_tracks']} "
            f"lost={row['lost_tracks']} median_init={fmt(row['median_init_support'])} "
            f"median_min_filtered={fmt(row['median_min_filtered_support'])} "
            f"median_pre_lost={fmt(row['median_pre_lost_support'])}"
        )


if __name__ == "__main__":
    main()
