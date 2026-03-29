import numpy as np
import torch
import torch.nn.functional as F

from .model import load
from . import transform


class UtoniaTracker:
    def __init__(
        self,
        model=None,
        device=None,
        scale=0.2,
        init_radius=0.8,
        cluster_radius=1.2,
        gate_radius=4.0,
        sim_threshold=0.6,
        init_points=64,
        min_points=64,
        spatial_weight=0.5,
        proto_momentum=0.9,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or load(
            "utonia",
            repo_id="Pointcept/Utonia",
            custom_config={"enc_patch_size": [1024] * 5, "enable_flash": False},
        ).to(self.device)
        self.model.eval()
        self.transform = transform.default(scale, apply_z_positive=False)
        self.init_radius = init_radius
        self.cluster_radius = cluster_radius
        self.gate_radius = gate_radius
        self.sim_threshold = sim_threshold
        self.init_points = init_points
        self.min_points = min_points
        self.spatial_weight = spatial_weight
        self.proto_momentum = proto_momentum
        self.prototype = None
        self.centroid = None
        self.velocity = None
        self.seed_index = None

    def _prepare(self, coord):
        point = {
            "coord": coord.copy(),
            "color": np.zeros_like(coord),
            "normal": np.zeros_like(coord),
        }
        return self.transform(point)

    def _encode(self, coord):
        point = self._prepare(coord)
        with torch.inference_mode():
            for key, value in point.items():
                if isinstance(value, torch.Tensor) and self.device == "cuda":
                    point[key] = value.cuda(non_blocking=True)
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
        coord = torch.as_tensor(coord, dtype=torch.float32, device=feat.device)
        return coord, feat

    def initialize(self, coord, query_xyz):
        coord, feat = self._encode(coord)
        query_xyz = torch.as_tensor(query_xyz, dtype=torch.float32, device=coord.device)
        dist2 = torch.sum((coord - query_xyz) ** 2, dim=1)
        self.seed_index = torch.argmin(dist2)
        mask = dist2.sqrt() < self.init_radius
        if int(mask.sum()) < self.init_points:
            topk = torch.topk(
                dist2, k=min(self.init_points, dist2.numel()), largest=False
            ).indices
            mask = torch.zeros_like(mask)
            mask[topk] = True
        self.centroid = coord[mask].mean(0)
        self.velocity = torch.zeros(3, device=coord.device)
        self.prototype = F.normalize(feat[mask].mean(0), dim=0)
        sim = feat @ self.prototype
        return {
            "coord": coord.detach().cpu().numpy(),
            "sim": sim.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "centroid": self.centroid.detach().cpu().numpy(),
            "seed_index": int(self.seed_index.detach().cpu()),
        }

    def step(self, coord):
        coord, feat = self._encode(coord)
        pred = self.centroid + self.velocity
        dist_pred = torch.linalg.norm(coord - pred, dim=1)
        sim = feat @ self.prototype
        combined = sim - self.spatial_weight * (dist_pred / self.gate_radius)
        combined[dist_pred > self.gate_radius] = -1e9
        if torch.all(dist_pred > self.gate_radius):
            combined = sim
        anchor = torch.argmax(combined)
        dist_anchor = torch.linalg.norm(coord - coord[anchor], dim=1)
        mask = (dist_anchor < self.cluster_radius) & (sim > self.sim_threshold)
        if int(mask.sum()) < self.min_points:
            topk = torch.topk(
                combined, k=min(self.min_points, combined.numel())
            ).indices
            mask = torch.zeros_like(mask)
            mask[topk] = True
        weight = sim[mask].clamp_min(0) + 1e-6
        new_centroid = (coord[mask] * weight[:, None]).sum(0) / weight.sum()
        new_prototype = F.normalize((feat[mask] * weight[:, None]).sum(0), dim=0)
        self.velocity = new_centroid - self.centroid
        self.centroid = new_centroid
        self.prototype = F.normalize(
            self.proto_momentum * self.prototype
            + (1.0 - self.proto_momentum) * new_prototype,
            dim=0,
        )
        return {
            "coord": coord.detach().cpu().numpy(),
            "sim": sim.detach().cpu().numpy(),
            "mask": mask.detach().cpu().numpy(),
            "centroid": self.centroid.detach().cpu().numpy(),
            "anchor_index": int(anchor.detach().cpu()),
        }
