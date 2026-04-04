import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.detectors.openpcdet import (
    DEFAULT_KITTI_POINTPILLAR_CFG,
    DEFAULT_KITTI_POINTPILLAR_CKPT,
    discover_kitti_sequence_ids,
    precompute_kitti_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_dir",
        help="KITTI tracking training dir, e.g. /path/to/trackkitti/training",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Optional sequence ids such as 0000 0001 0020",
    )
    parser.add_argument(
        "--cfg-file",
        default=str(DEFAULT_KITTI_POINTPILLAR_CFG),
        help="OpenPCDet config file",
    )
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_KITTI_POINTPILLAR_CKPT),
        help="OpenPCDet pretrained checkpoint",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.1,
        help="Minimum detector score to export",
    )
    parser.add_argument(
        "--output-root",
        default="data/detections/openpcdet_pointpillar",
        help="Where to save exported detections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    cfg_file = Path(args.cfg_file).resolve()
    ckpt = Path(args.ckpt).resolve()
    output_root = Path(args.output_root).resolve()

    if not cfg_file.is_file():
        raise FileNotFoundError(f"OpenPCDet config not found: {cfg_file}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"OpenPCDet checkpoint not found: {ckpt}")

    sequence_ids = args.sequences or discover_kitti_sequence_ids(training_dir)
    precompute_kitti_detections(
        training_dir=training_dir,
        sequence_ids=sequence_ids,
        cfg_file=cfg_file,
        ckpt=ckpt,
        score_thresh=args.score_thresh,
        output_root=output_root,
        detector_name="PointPillars",
    )


if __name__ == "__main__":
    main()
