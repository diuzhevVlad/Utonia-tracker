from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OPENPCDET_ROOT = ROOT / "third_party" / "OpenPCDet"
OPENPCDET_TOOLS = OPENPCDET_ROOT / "tools"

if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))
if str(OPENPCDET_TOOLS) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_TOOLS))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


DEFAULT_KITTI_POINTPILLAR_CFG = OPENPCDET_TOOLS / "cfgs/kitti_models/pointpillar.yaml"
DEFAULT_KITTI_POINTPILLAR_CKPT = OPENPCDET_ROOT / "checkpoints/pointpillar_kitti.pth"


class BinDemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path: Path, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )
        self.sample_file_list = sorted(str(path) for path in root_path.glob("*.bin"))

    def __len__(self) -> int:
        return len(self.sample_file_list)

    def __getitem__(self, index: int):
        points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        return self.prepare_data(
            data_dict={
                "points": points,
                "frame_id": Path(self.sample_file_list[index]).stem,
            }
        )


def discover_kitti_sequence_ids(training_dir: Path) -> list[str]:
    velodyne_root = training_dir / "velodyne"
    if not velodyne_root.is_dir():
        raise FileNotFoundError(f"KITTI velodyne directory not found: {velodyne_root}")
    return sorted(path.name for path in velodyne_root.iterdir() if path.is_dir())


def build_dataset(sequence_dir: Path):
    logger = common_utils.create_logger()
    return BinDemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, sequence_dir, logger=logger)


def build_model(cfg_file: Path, ckpt: Path, sequence_dir: Path):
    old_cwd = Path.cwd()
    os.chdir(OPENPCDET_TOOLS)
    try:
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(old_cwd)
    logger = common_utils.create_logger()
    dataset = build_dataset(sequence_dir)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return model


def export_sequence(dataset, model, out_dir: Path, score_thresh: float, detector_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = np.asarray(dataset.class_names)
    np.savez_compressed(
        out_dir / "meta.npz",
        source=np.asarray(["OpenPCDet"]),
        detector=np.asarray([detector_name]),
        box_format=np.asarray(["x_y_z_dx_dy_dz_heading"]),
        coordinate_frame=np.asarray(["lidar"]),
        class_names=class_names,
        score_threshold=np.asarray([score_thresh], dtype=np.float32),
    )

    with torch.no_grad():
        for index in range(len(dataset)):
            data_dict = dataset.collate_batch([dataset[index]])
            frame_id = str(data_dict["frame_id"][0])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred = pred_dicts[0]

            scores = pred["pred_scores"].detach().cpu().numpy()
            keep = scores >= score_thresh
            boxes = pred["pred_boxes"].detach().cpu().numpy()[keep]
            scores = scores[keep]
            label_ids = pred["pred_labels"].detach().cpu().numpy()[keep]
            label_names = (
                class_names[label_ids - 1] if label_ids.size else np.asarray([], dtype=class_names.dtype)
            )

            np.savez_compressed(
                out_dir / f"{frame_id}.npz",
                boxes_lidar=boxes.astype(np.float32, copy=False),
                scores=scores.astype(np.float32, copy=False),
                label_ids=label_ids.astype(np.int32, copy=False),
                label_names=label_names,
            )

            if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                print(f"  exported {index + 1}/{len(dataset)} frames")


def precompute_kitti_detections(
    training_dir: Path,
    sequence_ids: list[str],
    cfg_file: Path = DEFAULT_KITTI_POINTPILLAR_CFG,
    ckpt: Path = DEFAULT_KITTI_POINTPILLAR_CKPT,
    score_thresh: float = 0.1,
    output_root: Path = Path("data/detections/openpcdet_pointpillar"),
    detector_name: str = "PointPillars",
) -> None:
    velodyne_root = training_dir / "velodyne"
    model = None

    for sequence_id in sequence_ids:
        sequence_dir = velodyne_root / sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {sequence_dir}")

        if model is None:
            model = build_model(cfg_file, ckpt, sequence_dir)

        print(f"precomputing detections for sequence {sequence_id}")
        dataset = build_dataset(sequence_dir)
        export_sequence(
            dataset=dataset,
            model=model,
            out_dir=output_root / sequence_id,
            score_thresh=score_thresh,
            detector_name=detector_name,
        )
