import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.adapters import KittiGtDetectionSource
from utonia.tracking.detectors.utonia_segmentor import (
    KITTI_SEG_CLASSES,
    LinearSegHead,
    class_weights_from_batch,
    encode_features_chunked,
    point_labels_from_detections,
    sample_training_indices,
)
from utonia.tracking.encoder import UtoniaFrameEncoder
from utonia.tracking.visualization import load_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dir", help="KITTI tracking training dir")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every N-th frame from each sequence during training",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=16384)
    parser.add_argument("--background-ratio", type=float, default=1.0)
    parser.add_argument("--max-chunk-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="ckpt/utonia_kitti_seg_head.pt",
        help="Output checkpoint path",
    )
    return parser.parse_args()


def discover_sequence_ids(training_dir: Path) -> list[str]:
    velodyne_root = training_dir / "velodyne"
    if not velodyne_root.is_dir():
        raise FileNotFoundError(f"KITTI velodyne directory not found: {velodyne_root}")
    return sorted(path.name for path in velodyne_root.iterdir() if path.is_dir())


def build_frame_list(
    training_dir: Path,
    sequence_ids: list[str],
    frame_stride: int,
) -> list[tuple[Path, int]]:
    frames: list[tuple[Path, int]] = []
    velodyne_root = training_dir / "velodyne"
    for sequence_id in sequence_ids:
        sequence_dir = velodyne_root / sequence_id
        source = KittiGtDetectionSource(sequence_dir)
        for frame_id in source.frame_ids()[::frame_stride]:
            point_path = sequence_dir / f"{frame_id:06d}.bin"
            if point_path.is_file():
                frames.append((sequence_dir, frame_id))
    return frames


def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    sequence_ids = args.sequences or discover_sequence_ids(training_dir)
    frame_list = build_frame_list(training_dir, sequence_ids, args.frame_stride)
    if not frame_list:
        raise ValueError("No training frames found")
    print(f"training on {len(frame_list)} frames from {len(sequence_ids)} sequences")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    encoder = UtoniaFrameEncoder(scale=args.scale)
    encoder.build_model()
    encoder.build_transform()

    first_sequence_dir, first_frame_id = frame_list[0]
    first_coord = load_xyz(first_sequence_dir / f"{first_frame_id:06d}.bin")
    first_feat = encode_features_chunked(
        encoder,
        first_coord,
        max_points_per_chunk=args.max_chunk_points,
    )
    head = LinearSegHead(first_feat.shape[1], len(KITTI_SEG_CLASSES)).to(encoder.device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    sources = {seq: KittiGtDetectionSource(training_dir / "velodyne" / seq) for seq in sequence_ids}

    for epoch in range(args.epochs):
        order = rng.permutation(len(frame_list))
        total_loss = 0.0
        total_frames = 0
        for step, idx in enumerate(order, start=1):
            sequence_dir, frame_id = frame_list[int(idx)]
            coord = load_xyz(sequence_dir / f"{frame_id:06d}.bin")
            frame = sources[sequence_dir.name].get_frame_detections(frame_id)
            labels_np = point_labels_from_detections(coord, frame.detections)
            sample_idx_np = sample_training_indices(
                labels_np,
                max_points=args.max_points,
                background_ratio=args.background_ratio,
                rng=rng,
            )
            sample_idx = torch.as_tensor(sample_idx_np, dtype=torch.long, device=encoder.device)
            target = torch.as_tensor(labels_np[sample_idx_np], dtype=torch.long, device=encoder.device)

            feat = encode_features_chunked(
                encoder,
                coord,
                max_points_per_chunk=args.max_chunk_points,
            )

            logits = head(feat[sample_idx.cpu()].to(encoder.device))
            weights = class_weights_from_batch(target, len(KITTI_SEG_CLASSES)).to(encoder.device)
            loss = F.cross_entropy(logits, target, weight=weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_frames += 1
            if step % 100 == 0 or step == len(frame_list):
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step}/{len(frame_list)} "
                    f"loss={total_loss / max(total_frames, 1):.4f}"
                )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": {
                "in_channels": first_feat.shape[1],
                "num_classes": len(KITTI_SEG_CLASSES),
                "class_names": KITTI_SEG_CLASSES,
                "scale": args.scale,
            },
            "train_args": vars(args),
        },
        output_path,
    )
    print(f"saved segmentation head to {output_path}")


if __name__ == "__main__":
    main()
