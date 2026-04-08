from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from ..encoder import rotation_matrix_z
from ..types import Box3D


KITTI_SEG_CLASSES = (
    "background",
    "Car",
    "Van",
    "Pedestrian",
    "Cyclist",
)
KITTI_SEG_CLASS_TO_ID = {name: index for index, name in enumerate(KITTI_SEG_CLASSES)}
KITTI_SEG_FOREGROUND = KITTI_SEG_CLASSES[1:]


@dataclass
class ClusterConfig:
    eps: float
    min_samples: int
    min_points: int
    min_score: float


CLUSTER_CONFIG = {
    "Car": ClusterConfig(eps=1.2, min_samples=16, min_points=24, min_score=0.35),
    "Van": ClusterConfig(eps=1.3, min_samples=16, min_points=24, min_score=0.35),
    "Pedestrian": ClusterConfig(eps=0.6, min_samples=8, min_points=12, min_score=0.35),
    "Cyclist": ClusterConfig(eps=0.8, min_samples=10, min_points=14, min_score=0.35),
}


class LinearSegHead(nn.Module):
    """Strict linear segmentation head over frozen Utonia features."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.linear(feat)


def point_labels_from_detections(
    coord: np.ndarray,
    detections,
    class_to_id: dict[str, int] = KITTI_SEG_CLASS_TO_ID,
) -> np.ndarray:
    """Create pseudo point labels from GT boxes by oriented box inclusion."""

    labels = np.zeros(coord.shape[0], dtype=np.int64)
    best_dist = np.full(coord.shape[0], np.inf, dtype=np.float32)
    for detection in detections:
        class_id = class_to_id.get(detection.label)
        if class_id is None:
            continue
        rotation = rotation_matrix_z(-detection.box.yaw)
        local = (coord - detection.box.center[None, :]) @ rotation.T
        half_size = detection.box.size * 0.5
        inside = np.all(np.abs(local) <= half_size[None, :], axis=1)
        if not np.any(inside):
            continue
        dist = np.linalg.norm(coord - detection.box.center[None, :], axis=1)
        update = inside & (dist < best_dist)
        labels[update] = class_id
        best_dist[update] = dist[update]
    return labels


def sample_training_indices(
    labels: np.ndarray,
    max_points: int = 16384,
    background_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a balanced subset of points for cheap linear-head training."""

    rng = rng or np.random.default_rng()
    fg_idx = np.flatnonzero(labels > 0)
    bg_idx = np.flatnonzero(labels == 0)

    if fg_idx.size == 0:
        if bg_idx.size <= max_points:
            return bg_idx
        return rng.choice(bg_idx, size=max_points, replace=False)

    if fg_idx.size > max_points:
        return np.sort(rng.choice(fg_idx, size=max_points, replace=False))

    max_bg = min(bg_idx.size, int(round(fg_idx.size * background_ratio)))
    remaining = max(max_points - fg_idx.size, 0)
    bg_take = min(max_bg, remaining)
    if bg_take > 0:
        bg_sel = rng.choice(bg_idx, size=bg_take, replace=False)
        return np.sort(np.concatenate([fg_idx, bg_sel]))
    return np.sort(fg_idx)


def class_weights_from_batch(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return inverse-frequency weights for one sampled batch."""

    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.ones_like(counts)
    positive = counts > 0
    weights[positive] = counts[positive].sum() / counts[positive]
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def encode_features_chunked(
    encoder,
    coord: np.ndarray,
    max_points_per_chunk: int = 50000,
) -> torch.Tensor:
    """Encode a large frame in chunks to keep GPU memory bounded."""

    if coord.shape[0] <= max_points_per_chunk:
        _, feat = encoder.encode_frame(coord)
        return feat.detach().cpu()

    order = np.argsort(coord[:, 0])
    feat_cpu = None
    for start in range(0, coord.shape[0], max_points_per_chunk):
        chunk_idx = order[start : start + max_points_per_chunk]
        _, feat = encoder.encode_frame(coord[chunk_idx])
        feat = feat.detach().cpu()
        if feat_cpu is None:
            feat_cpu = torch.empty(
                (coord.shape[0], feat.shape[1]),
                dtype=feat.dtype,
            )
        feat_cpu[torch.as_tensor(chunk_idx, dtype=torch.long)] = feat
        del feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert feat_cpu is not None
    return feat_cpu


def predict_probs_chunked(
    encoder,
    head: nn.Module,
    coord: np.ndarray,
    max_points_per_chunk: int = 50000,
) -> np.ndarray:
    """Predict per-point segmentation probabilities with chunked encoding."""

    feat_cpu = encode_features_chunked(
        encoder=encoder,
        coord=coord,
        max_points_per_chunk=max_points_per_chunk,
    )
    with torch.inference_mode():
        logits = head(feat_cpu.to(encoder.device))
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def fit_box_from_cluster(points: np.ndarray) -> Box3D:
    """Fit a coarse oriented box from clustered foreground points."""

    centroid = points.mean(axis=0)
    centered_xy = points[:, :2] - centroid[:2][None, :]
    cov = centered_xy.T @ centered_xy / max(len(points), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = float(np.arctan2(principal[1], principal[0]))
    rotation = rotation_matrix_z(-yaw)
    local = (points - centroid[None, :]) @ rotation.T
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    size = np.maximum(local_max - local_min, 1e-3)
    center_local = 0.5 * (local_min + local_max)
    center = centroid + center_local @ rotation
    return Box3D(center=center.astype(np.float32), size=size.astype(np.float32), yaw=yaw)


def detections_from_segmentation(
    coord: np.ndarray,
    probs: np.ndarray,
    class_names: tuple[str, ...] = KITTI_SEG_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster foreground points and convert them into detector-style boxes."""

    pred = probs.argmax(axis=1)
    boxes = []
    scores = []
    label_ids = []
    label_names = []

    for class_id, class_name in enumerate(class_names[1:], start=1):
        cfg = CLUSTER_CONFIG[class_name]
        class_mask = (pred == class_id) & (probs[:, class_id] >= cfg.min_score)
        if int(class_mask.sum()) < cfg.min_points:
            continue
        class_points = coord[class_mask]
        class_probs = probs[class_mask, class_id]
        clustering = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
        cluster_ids = clustering.fit_predict(class_points)
        for cluster_id in np.unique(cluster_ids):
            if cluster_id < 0:
                continue
            cluster_mask = cluster_ids == cluster_id
            if int(cluster_mask.sum()) < cfg.min_points:
                continue
            cluster_points = class_points[cluster_mask]
            cluster_probs = class_probs[cluster_mask]
            box = fit_box_from_cluster(cluster_points)
            boxes.append(box.as_array())
            scores.append(float(cluster_probs.mean()))
            label_ids.append(class_id)
            label_names.append(class_name)

    if not boxes:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.asarray([], dtype=str),
        )

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(label_ids, dtype=np.int32),
        np.asarray(label_names, dtype=str),
    )
