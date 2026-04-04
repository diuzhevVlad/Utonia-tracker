from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class AssociationConfig:
    max_match_distance: float = 5.0
    max_missed: int = 2
    min_bev_iou: float = 0.0
    motion_weight: float = 1.0
    bev_iou_weight: float = 1.0
    appearance_weight: float = 0.0


@dataclass
class FeatureCropConfig:
    mode: str = "detections_and_tracks"
    crop_margin: float = 2.0
    min_points: int = 2048


@dataclass
class MotionModelConfig:
    kind: str = "kalman"
    process_var: float = 1.0
    measurement_var: float = 1.0


def build_class_configs(
    base: AssociationConfig,
    enabled: bool = True,
) -> dict[str, AssociationConfig]:
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
