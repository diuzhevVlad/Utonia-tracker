import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.detectors.utonia_segmentor import (
    KITTI_SEG_CLASSES,
    LinearSegHead,
    detections_from_segmentation,
    predict_probs_chunked,
)
from utonia.tracking.encoder import UtoniaFrameEncoder
from utonia.tracking.visualization import load_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dir", help="KITTI tracking training dir")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument(
        "--ckpt",
        default="ckpt/utonia_kitti_seg_head.pt",
        help="Trained segmentation-head checkpoint",
    )
    parser.add_argument(
        "--point-score-thresh",
        type=float,
        default=0.35,
        help="Minimum point probability used before clustering",
    )
    parser.add_argument("--max-chunk-points", type=int, default=50000)
    parser.add_argument(
        "--output-root",
        default="data/detections/utonia_seg_linear",
        help="Where to save exported detections",
    )
    return parser.parse_args()


def discover_sequence_ids(training_dir: Path) -> list[str]:
    velodyne_root = training_dir / "velodyne"
    if not velodyne_root.is_dir():
        raise FileNotFoundError(f"KITTI velodyne directory not found: {velodyne_root}")
    return sorted(path.name for path in velodyne_root.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    sequence_ids = args.sequences or discover_sequence_ids(training_dir)
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    encoder = UtoniaFrameEncoder(scale=float(config["scale"]))
    encoder.build_model()
    encoder.build_transform()
    head = LinearSegHead(
        in_channels=int(config["in_channels"]),
        num_classes=int(config["num_classes"]),
    ).to(encoder.device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    output_root = Path(args.output_root).resolve()
    velodyne_root = training_dir / "velodyne"

    for sequence_id in sequence_ids:
        sequence_dir = velodyne_root / sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {sequence_dir}")
        out_dir = output_root / sequence_id
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "meta.npz",
            source=np.asarray(["UtoniaSegLinear"]),
            detector=np.asarray(["UtoniaLinearSeg"]),
            box_format=np.asarray(["x_y_z_dx_dy_dz_heading"]),
            coordinate_frame=np.asarray(["lidar"]),
            class_names=np.asarray(KITTI_SEG_CLASSES[1:]),
            point_score_threshold=np.asarray([args.point_score_thresh], dtype=np.float32),
        )
        frame_paths = sorted(sequence_dir.glob("*.bin"))
        for index, point_path in enumerate(frame_paths, start=1):
            coord = load_xyz(point_path)
            probs = predict_probs_chunked(
                encoder,
                head,
                coord,
                max_points_per_chunk=args.max_chunk_points,
            )
            boxes, scores, label_ids, label_names = detections_from_segmentation(coord, probs)
            if scores.size:
                keep = scores >= args.point_score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                label_ids = label_ids[keep]
                label_names = label_names[keep]
            np.savez_compressed(
                out_dir / f"{point_path.stem}.npz",
                boxes_lidar=boxes.astype(np.float32, copy=False),
                scores=scores.astype(np.float32, copy=False),
                label_ids=label_ids.astype(np.int32, copy=False),
                label_names=label_names,
            )
            if index % 100 == 0 or index == len(frame_paths):
                print(f"sequence {sequence_id}: exported {index}/{len(frame_paths)} frames")


if __name__ == "__main__":
    main()
