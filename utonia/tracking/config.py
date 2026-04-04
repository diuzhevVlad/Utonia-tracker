from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class AssociationConfig:
    """Association thresholds and cost weights."""

    max_match_distance: float = 5.0
    max_missed: int = 2
    min_bev_iou: float = 0.0
    motion_weight: float = 1.0
    bev_iou_weight: float = 1.0
    appearance_weight: float = 0.0


@dataclass
class FeatureCropConfig:
    """Controls how much of the frame is encoded by Utonia."""

    mode: str = "detections_and_tracks"
    crop_margin: float = 2.0
    min_points: int = 2048


@dataclass
class MotionModelConfig:
    """Motion-model selection and noise parameters."""

    kind: str = "kalman"
    process_var: float = 1.0
    measurement_var: float = 1.0


@dataclass
class RecoveryConfig:
    """Point-level recovery settings for lost confirmed tracks."""

    enabled: bool = True
    max_missed: int = 3
    score_alpha: float = 0.7
    gate_radius: float = 3.0
    cluster_radius: float = 1.2
    sim_threshold: float = 0.35
    min_points: int = 48
    min_mean_similarity: float = 0.45
    max_center_distance: float = 2.0
    max_extent_scale: float = 1.5
    proto_momentum: float = 0.98


@dataclass
class SpawnConfig:
    """Suppresses duplicate track births near strong existing tracks."""

    enabled: bool = True
    same_class_min_bev_iou: float = 0.1
    cross_class_min_bev_iou: float = 0.25
    max_center_distance: float = 2.0
    max_track_missed: int = 1


def build_class_configs(
    base: AssociationConfig,
    enabled: bool = True,
) -> dict[str, AssociationConfig]:
    """Build light class-specific threshold overrides from one base config."""

    if not enabled:
        return {}
    return {
        "Car": replace(base, max_match_distance=base.max_match_distance * 1.0),
        "Van": replace(base, max_match_distance=base.max_match_distance * 1.0),
        "Cyclist": replace(
            base,
            max_match_distance=max(1.0, base.max_match_distance * 0.8),
            max_missed=max(1, int(round(base.max_missed * 0.8))),
        ),
        "Pedestrian": replace(
            base,
            max_match_distance=max(1.0, base.max_match_distance * 0.6),
            max_missed=max(1, int(round(base.max_missed * 0.6))),
        ),
    }
