from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..model import load
from .. import transform
from .types import Box3D


def rotation_matrix_z(yaw: float) -> np.ndarray:
    """Return a 3D rotation matrix around the z axis."""

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class UtoniaFrameEncoder:
    """Shared Utonia feature encoder for single and multi-object tracking."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        device: str | None = None,
        scale: float = 0.2,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.model = model
        self.transform = None

    def _ensure_ready(self) -> None:
        """Lazily build the model and transform on first use."""

        if self.model is None:
            self.build_model()
        if self.transform is None:
            self.build_transform()

    def build_model(self) -> None:
        """Load the pretrained Utonia backbone."""

        self.model = load(
            "utonia",
            repo_id="Pointcept/Utonia",
            custom_config={"enc_patch_size": [1024] * 5, "enable_flash": False},
        ).to(self.device)
        self.model.eval()

    def build_transform(self) -> None:
        """Build the default preprocessing transform used by tracking."""

        self.transform = transform.default(self.scale, apply_z_positive=False)

    def preprocess(self, coord: np.ndarray) -> dict[str, torch.Tensor]:
        """Convert raw XYZ points into a Utonia input dict."""

        self._ensure_ready()
        point = {
            "coord": coord.copy(),
            "color": np.zeros_like(coord),
            "normal": np.zeros_like(coord),
        }
        return self.transform(point)

    def encode_frame(self, coord: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a point cloud and return original points with per-point features."""

        point = self.preprocess(coord)
        with torch.inference_mode():
            for key, value in point.items():
                point[key] = value.to(self.device)
            point = self.model(point)
            for _ in range(2):
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            while "pooling_parent" in point:
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            feat = F.normalize(point.feat[point.inverse], dim=1)
        coord_t = torch.as_tensor(coord, dtype=torch.float32, device=self.device)
        return coord_t, feat

    def select_roi_coord(
        self,
        coord: np.ndarray,
        boxes: list[Box3D],
        mode: str = "detections_and_tracks",
        crop_margin: float = 2.0,
        min_points: int = 2048,
    ) -> tuple[np.ndarray | None, dict[str, int | bool]]:
        """Return a local crop or `None` when appearance should be skipped."""

        stats: dict[str, int | bool] = {
            "input_points": int(coord.shape[0]),
            "roi_boxes": int(len(boxes)),
            "roi_points": int(coord.shape[0]),
            "roi_skip_appearance": False,
        }
        if mode == "full" or not boxes:
            return coord, stats

        mask = np.zeros(coord.shape[0], dtype=bool)
        for box in boxes:
            half_size = box.size * (0.5 * crop_margin)
            local = np.abs(coord - box.center[None, :])
            mask |= np.all(local <= half_size[None, :], axis=1)

        roi_points = int(mask.sum())
        stats["roi_points"] = roi_points
        if roi_points < min_points:
            stats["roi_skip_appearance"] = True
            return None, stats
        return coord[mask].copy(), stats

    def box_feature(
        self,
        coord_t: torch.Tensor,
        feat_t: torch.Tensor,
        box: Box3D,
        min_points: int = 64,
        box_margin: float = 1.1,
    ) -> np.ndarray:
        """Pool a normalized appearance embedding for one box."""

        center = torch.as_tensor(box.center, dtype=torch.float32, device=self.device)
        rotation = torch.as_tensor(
            rotation_matrix_z(-box.yaw),
            dtype=torch.float32,
            device=self.device,
        )
        local = (coord_t - center) @ rotation.T
        half_size = torch.as_tensor(
            box.size * (0.5 * box_margin),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.all(torch.abs(local) <= half_size, dim=1)
        # Very sparse boxes fall back to nearest points in the encoded crop.
        if int(mask.sum()) < min_points:
            dist = torch.linalg.norm(coord_t - center, dim=1)
            topk = torch.topk(
                dist,
                k=min(min_points, dist.numel()),
                largest=False,
            ).indices
            mask = torch.zeros_like(dist, dtype=torch.bool)
            mask[topk] = True
        feature = F.normalize(feat_t[mask].mean(0), dim=0)
        return feature.detach().cpu().numpy()
