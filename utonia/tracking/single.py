from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import UtoniaFrameEncoder


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
        self.encoder = UtoniaFrameEncoder(model=model, device=device, scale=scale)
        self.device = self.encoder.device
        self.scale = scale
        self.init_radius = init_radius
        self.alpha = alpha
        self.cluster_radius = cluster_radius
        self.gate_radius = gate_radius
        self.sim_threshold = sim_threshold
        self.init_points = init_points
        self.min_points = min_points
        self.proto_momentum = proto_momentum
        self.state = TrackerState()

    @property
    def model(self) -> torch.nn.Module | None:
        return self.encoder.model

    @model.setter
    def model(self, value: torch.nn.Module | None) -> None:
        self.encoder.model = value

    @property
    def transform(self):
        return self.encoder.transform

    @transform.setter
    def transform(self, value) -> None:
        self.encoder.transform = value

    def build_model(self) -> None:
        self.encoder.build_model()

    def build_transform(self) -> None:
        self.encoder.build_transform()

    def preprocess(self, coord: np.ndarray) -> dict[str, torch.Tensor]:
        return self.encoder.preprocess(coord)

    def encode_frame(self, coord: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode_frame(coord)

    def initialize(
        self, coord: np.ndarray, query_xyz: np.ndarray | list[float]
    ) -> dict[str, np.ndarray | int]:
        coord_t, feat = self.encode_frame(coord)
        query_t = torch.as_tensor(query_xyz, dtype=torch.float32, device=self.device)
        dist = torch.linalg.norm(coord_t - query_t, dim=1)
        self.state.seed_index = int(torch.argmin(dist).detach().cpu())
        mask = dist < self.init_radius
        if int(mask.sum()) < self.init_points:
            topk = torch.topk(
                dist,
                k=min(self.init_points, dist.numel()),
                largest=False,
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
        return self.state.centroid + self.state.velocity

    def score_points(self, coord: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        sim = feat @ self.state.prototype
        dist = torch.linalg.norm(coord - self.predict_position(), dim=1)
        score = sim - self.alpha * (dist / self.gate_radius)
        score[dist > self.gate_radius] = -1e9
        if torch.all(dist > self.gate_radius):
            score = sim
        return score

    def extract_target(
        self,
        coord: torch.Tensor,
        sim: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
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
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
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
        self,
        frames: list[np.ndarray],
        query_xyz: np.ndarray | list[float],
    ) -> list[dict[str, np.ndarray | int]]:
        states = [self.initialize(frames[0], query_xyz)]
        for frame in frames[1:]:
            states.append(self.step(frame))
        return states
