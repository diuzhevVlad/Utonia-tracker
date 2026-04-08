import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"
OPENPCDET_TOOLS = OPENPCDET_ROOT / "tools"


if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


MODEL_DEFAULTS = {
    "pointpillar": {
        "cfg": OPENPCDET_TOOLS / "cfgs" / "kitti_models" / "pointpillar.yaml",
        "ckpt": OPENPCDET_ROOT / "checkpoints" / "pointpillar_kitti.pth",
    },
    "pointrcnn": {
        "cfg": OPENPCDET_TOOLS / "cfgs" / "kitti_models" / "pointrcnn.yaml",
        "ckpt": OPENPCDET_ROOT / "checkpoints" / "pointrcnn_kitti.pth",
    },
}


class TrackingDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, sequence_dir: Path, logger):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=sequence_dir,
            logger=logger,
        )
        self.sequence_dir = sequence_dir
        self.sample_file_list = sorted(sequence_dir.glob("*.bin"))

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        path = self.sample_file_list[index]
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return self.prepare_data(
            data_dict={
                "points": points,
                "frame_id": path.stem,
            }
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DEFAULTS.keys()),
        required=True,
        help="Detector to run via vendored OpenPCDet.",
    )
    parser.add_argument(
        "--data_root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with calib/, label_02/, velodyne/.",
    )
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "data" / "detections"),
        help="Output root for saved detections.",
    )
    parser.add_argument(
        "--sequence",
        nargs="*",
        default=None,
        help="Optional list of sequence ids, e.g. 0000 0001.",
    )
    parser.add_argument(
        "--cfg_file",
        default=None,
        help="Optional OpenPCDet config override.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint override.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=None,
        help="Optional score threshold override applied after model inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute detections even when output .npz already exists.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional per-sequence frame limit for smoke tests.",
    )
    return parser.parse_args()


def resolve_sequences(velodyne_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [velodyne_root / seq for seq in requested]
    return sorted(path for path in velodyne_root.iterdir() if path.is_dir())


def filter_predictions(pred_dict, score_thresh: float | None):
    pred_boxes = pred_dict["pred_boxes"].detach().cpu().numpy()
    pred_scores = pred_dict["pred_scores"].detach().cpu().numpy()
    pred_labels = pred_dict["pred_labels"].detach().cpu().numpy()

    if score_thresh is not None:
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

    return pred_boxes, pred_scores, pred_labels


def save_outputs(
    output_root: Path,
    sequence: str,
    frame_id: str,
    class_names: list[str],
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
) -> None:
    npz_dir = output_root / "npz" / sequence
    txt_dir = output_root / "txt_lidar" / sequence
    npz_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_dir / f"{frame_id}.npz",
        pred_boxes=pred_boxes.astype(np.float32),
        pred_scores=pred_scores.astype(np.float32),
        pred_labels=pred_labels.astype(np.int32),
    )

    label_names = [class_names[int(label) - 1] for label in pred_labels]
    lines = []
    for name, score, box in zip(label_names, pred_scores, pred_boxes, strict=False):
        values = " ".join(f"{value:.6f}" for value in box.tolist())
        lines.append(f"{name} {score:.6f} {values}")
    (txt_dir / f"{frame_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    args = parse_args()
    defaults = MODEL_DEFAULTS[args.model]
    cfg_file = Path(args.cfg_file) if args.cfg_file else defaults["cfg"]
    ckpt = Path(args.ckpt) if args.ckpt else defaults["ckpt"]
    data_root = Path(args.data_root)
    output_root = Path(args.output_root) / args.model
    velodyne_root = data_root / "velodyne"

    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_file}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not velodyne_root.exists():
        raise FileNotFoundError(f"Velodyne root not found: {velodyne_root}")

    os.chdir(OPENPCDET_TOOLS)
    cfg_from_yaml_file(str(cfg_file), cfg)

    logger = common_utils.create_logger()
    logger.info("Running %s detections on %s", args.model, data_root)
    logger.info("Using config: %s", cfg_file)
    logger.info("Using checkpoint: %s", ckpt)
    logger.info("Saving to: %s", output_root)

    sequences = resolve_sequences(velodyne_root, args.sequence)
    dataset_for_model = TrackingDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        sequence_dir=sequences[0],
        logger=logger,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset_for_model,
    )
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for sequence_dir in sequences:
            sequence = sequence_dir.name
            dataset = TrackingDataset(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                sequence_dir=sequence_dir,
                logger=logger,
            )
            frame_count = len(dataset)
            if args.max_frames is not None:
                frame_count = min(frame_count, args.max_frames)
            logger.info("Sequence %s: %d frames", sequence, frame_count)
            for index in tqdm(range(frame_count), desc=f"{args.model}:{sequence}"):
                frame_path = dataset.sample_file_list[index]
                frame_id = frame_path.stem
                out_path = output_root / "npz" / sequence / f"{frame_id}.npz"
                if out_path.exists() and not args.overwrite:
                    continue

                data_dict = dataset[index]
                data_dict = dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                pred_boxes, pred_scores, pred_labels = filter_predictions(
                    pred_dicts[0], args.score_thresh
                )
                save_outputs(
                    output_root=output_root,
                    sequence=sequence,
                    frame_id=frame_id,
                    class_names=cfg.CLASS_NAMES,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                )


if __name__ == "__main__":
    main()
