import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKEVAL_ROOT = REPO_ROOT / "third_party" / "TrackEval"

if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

# TrackEval still uses removed NumPy aliases in some dataset files.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import trackeval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with label_02/ and velodyne/.",
    )
    parser.add_argument(
        "--tracker-dir",
        required=True,
        help="Run directory containing data/<sequence>.txt in KITTI tracking result format.",
    )
    parser.add_argument("--tracker-name", default=None)
    parser.add_argument("--sequences", nargs="+", default=["0019", "0020"])
    parser.add_argument("--classes", nargs="+", default=["car", "pedestrian"])
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "data" / "trackeval_kitti"),
        help="Directory where TrackEval-compatible gt/tracker layout is created.",
    )
    parser.add_argument("--split-name", default="training")
    parser.add_argument("--use-parallel", action="store_true")
    return parser.parse_args()


def sequence_length(data_root: Path, sequence: str) -> int:
    frame_paths = sorted((data_root / "velodyne" / sequence).glob("*.bin"))
    if frame_paths:
        return int(frame_paths[-1].stem) + 1
    label_path = data_root / "label_02" / f"{sequence}.txt"
    max_frame = 0
    for line in label_path.read_text().splitlines():
        if line.strip():
            max_frame = max(max_frame, int(line.split()[0]))
    return max_frame + 1


def prepare_layout(args) -> tuple[Path, Path, str]:
    data_root = Path(args.data_root)
    tracker_dir = Path(args.tracker_dir)
    tracker_name = args.tracker_name or tracker_dir.name
    work_dir = Path(args.work_dir)
    gt_folder = work_dir / "gt" / "kitti_2d_box_train"
    trackers_folder = work_dir / "trackers" / "kitti_2d_box_train"
    tracker_data_folder = trackers_folder / tracker_name / "data"

    (gt_folder / "label_02").mkdir(parents=True, exist_ok=True)
    tracker_data_folder.mkdir(parents=True, exist_ok=True)

    seqmap_path = gt_folder / f"evaluate_tracking.seqmap.{args.split_name}"
    with seqmap_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ")
        for sequence in args.sequences:
            writer.writerow([sequence, 0, 0, sequence_length(data_root, sequence)])
            src_gt = data_root / "label_02" / f"{sequence}.txt"
            dst_gt = gt_folder / "label_02" / f"{sequence}.txt"
            if dst_gt.exists() or dst_gt.is_symlink():
                dst_gt.unlink()
            try:
                dst_gt.symlink_to(src_gt)
            except OSError:
                shutil.copy2(src_gt, dst_gt)

            src_tracker = tracker_dir / "data" / f"{sequence}.txt"
            dst_tracker = tracker_data_folder / f"{sequence}.txt"
            if not src_tracker.exists():
                raise FileNotFoundError(src_tracker)
            shutil.copy2(src_tracker, dst_tracker)

    return gt_folder, trackers_folder, tracker_name


def main():
    args = parse_args()
    gt_folder, trackers_folder, tracker_name = prepare_layout(args)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": args.use_parallel,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": args.classes,
            "SPLIT_TO_EVAL": args.split_name,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "PRINT_CONFIG": True,
        }
    )
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_config)
    output_msg, output_res = evaluator.evaluate(
        [trackeval.datasets.Kitti2DBox(dataset_config)],
        metrics_list,
    )
    summary = {
        "tracker_name": tracker_name,
        "gt_folder": str(gt_folder),
        "trackers_folder": str(trackers_folder),
        "output_msg": output_msg,
        "output_res": output_res,
    }
    summary_path = Path(args.tracker_dir) / "trackeval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
