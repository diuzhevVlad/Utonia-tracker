from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import load
from . import transform


@dataclass
class TrackerState:
    prototype: torch.Tensor | None = None
    centroid: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    seed_index: int | None = None


class UtoniaTracker:
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        device: str | None = None,
        scale: float = 0.2,
        init_radius: float = 0.8,
        alpha: float = 0.9,
        cluster_radius: float = 1.2,
        gate_radius: float = 4.0,
        sim_threshold: float = 0.6,
        init_points: int = 64,
        min_points: int = 64,
        proto_momentum: float = 0.9,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.init_radius = init_radius
        self.alpha = alpha
        self.cluster_radius = cluster_radius
        self.gate_radius = gate_radius
        self.sim_threshold = sim_threshold
        self.init_points = init_points
        self.min_points = min_points
        self.proto_momentum = proto_momentum
        self.model = model
        self.transform = None
        self.state = TrackerState()

    def _ensure_ready(self) -> None:
        if self.model is None:
            self.build_model()
        if self.transform is None:
            self.build_transform()

    def build_model(self) -> None:
        """Load the pretrained Utonia encoder used to extract per-point features."""
        self.model = load(
            "utonia",
            repo_id="Pointcept/Utonia",
            custom_config={"enc_patch_size": [1024] * 5, "enable_flash": False},
        ).to(self.device)
        self.model.eval()

    def build_transform(self) -> None:
        """Create the fixed preprocessing pipeline applied to every LiDAR frame."""
        self.transform = transform.default(self.scale, apply_z_positive=False)

    def preprocess(self, coord: np.ndarray) -> dict[str, torch.Tensor]:
        """Convert raw XYZ points into the dictionary format expected by Utonia."""
        self._ensure_ready()
        point = {
            "coord": coord.copy(),
            "color": np.zeros_like(coord),
            "normal": np.zeros_like(coord),
        }
        return self.transform(point)

    def encode_frame(self, coord: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one frame and return original-frame coordinates with normalized features."""
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

    def initialize(
        self, coord: np.ndarray, query_xyz: np.ndarray | list[float]
    ) -> dict[str, np.ndarray | int]:
        """Initialize the track from a query point and return visualization-friendly outputs."""
        coord_t, feat = self.encode_frame(coord)
        query_t = torch.as_tensor(query_xyz, dtype=torch.float32, device=self.device)
        dist = torch.linalg.norm(coord_t - query_t, dim=1)
        self.state.seed_index = int(torch.argmin(dist).detach().cpu())
        mask = dist < self.init_radius
        if int(mask.sum()) < self.init_points:
            topk = torch.topk(
                dist, k=min(self.init_points, dist.numel()), largest=False
            ).indices
            mask = torch.zeros_like(mask)
            mask[topk] = True
        self.state.prototype = F.normalize(feat[mask].mean(0), dim=0)
        self.state.centroid = coord_t[mask].mean(0)
        self.state.velocity = torch.zeros(3, dtype=torch.float32, device=self.device)
        sim = feat @ self.state.prototype
        return {
            "coord": coord_t.detach().cpu().numpy(),
            "sim": sim.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "centroid": self.state.centroid.detach().cpu().numpy(),
            "seed_index": self.state.seed_index,
        }

    def predict_position(self) -> torch.Tensor:
        """Predict the next centroid with a constant-velocity motion model."""
        return self.state.centroid + self.state.velocity

    def score_points(self, coord: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Combine appearance similarity and motion gating into one score per point."""
        sim = feat @ self.state.prototype
        dist = torch.linalg.norm(coord - self.predict_position(), dim=1)
        score = sim - self.alpha * (dist / self.gate_radius)
        score[dist > self.gate_radius] = -1e9
        if torch.all(dist > self.gate_radius):
            score = sim
        return score

    def extract_target(
        self, coord: torch.Tensor, sim: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Build a target mask around the highest-scoring anchor point."""
        anchor = torch.argmax(score)
        mask = (
            torch.linalg.norm(coord - coord[anchor], dim=1) < self.cluster_radius
        ) & (sim > self.sim_threshold)
        if int(mask.sum()) < self.min_points:
            topk = torch.topk(score, k=min(self.min_points, score.numel())).indices
            mask = torch.zeros_like(mask)
            mask[topk] = True
        return mask

    def update_state(
        self, coord: torch.Tensor, feat: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Update centroid, velocity, and prototype from the selected target points."""
        old_centroid = self.state.centroid.clone()
        old_prototype = self.state.prototype.clone()
        weight = (feat[mask] @ old_prototype).clamp_min(0) + 1e-6
        self.state.centroid = (
            torch.sum(coord[mask] * weight[:, None], dim=0) / weight.sum()
        )
        self.state.velocity = self.state.centroid - old_centroid
        new_prototype = torch.sum(feat[mask] * weight[:, None], dim=0)
        self.state.prototype = F.normalize(
            self.proto_momentum * old_prototype
            + (1.0 - self.proto_momentum) * new_prototype,
            dim=0,
        )

    def step(self, coord: np.ndarray) -> dict[str, np.ndarray | int]:
        """Track the target in one new frame and return values useful for visualization."""
        coord_t, feat = self.encode_frame(coord)
        sim = feat @ self.state.prototype
        score = self.score_points(coord_t, feat)
        mask = self.extract_target(coord_t, sim, score)
        self.update_state(coord_t, feat, mask)
        anchor_index = int(torch.argmax(score).detach().cpu())
        return {
            "coord": coord_t.detach().cpu().numpy(),
            "sim": sim.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "centroid": self.state.centroid.detach().cpu().numpy(),
            "anchor_index": anchor_index,
        }

    def track_sequence(
        self, frames: list[np.ndarray], query_xyz: np.ndarray | list[float]
    ) -> list[dict[str, np.ndarray | int]]:
        """Run tracking over a list of frames and return per-frame outputs."""
        states = [self.initialize(frames[0], query_xyz)]
        for frame in frames[1:]:
            states.append(self.step(frame))
        return states
