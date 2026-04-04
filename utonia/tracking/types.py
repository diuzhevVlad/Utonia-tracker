from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def as_vector(
    value: np.ndarray | list[float] | tuple[float, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def normalize_np(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    norm = np.linalg.norm(feature)
    if norm <= 0:
        return feature
    return feature / norm


@dataclass
class Box3D:
    """Oriented 3D box in a named coordinate frame."""

    center: np.ndarray
    size: np.ndarray
    yaw: float
    coordinate_frame: str = "lidar"

    def __post_init__(self) -> None:
        self.center = as_vector(self.center, "center")
        self.size = as_vector(self.size, "size")
        if np.any(self.size <= 0):
            raise ValueError(f"size must be positive, got {self.size}")
        self.yaw = wrap_angle(self.yaw)

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [self.center, self.size, np.array([self.yaw], dtype=np.float32)]
        )


@dataclass
class Detection3D:
    """Dataset-neutral 3D detection used by the tracker."""

    box: Box3D
    label: str
    score: float = 1.0
    velocity: np.ndarray | None = None
    feature: np.ndarray | None = None
    source: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if not np.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")
        if self.velocity is not None:
            self.velocity = as_vector(self.velocity, "velocity")
        if self.feature is not None:
            self.feature = np.asarray(self.feature, dtype=np.float32)


@dataclass
class FrameDetections:
    """All detections associated with one frame."""

    frame_id: int
    detections: list[Detection3D]
    timestamp: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Track3D:
    """Minimal tracker state for one active object."""

    track_id: int
    box: Box3D
    label: str
    score: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    prototype: np.ndarray | None = None
    hits: int = 1
    missed: int = 0
    motion_state: dict[str, np.ndarray] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.velocity = as_vector(self.velocity, "velocity")
        if self.prototype is not None:
            self.prototype = np.asarray(self.prototype, dtype=np.float32)

    @property
    def is_confirmed(self) -> bool:
        return self.hits > 1


class DetectionSource(ABC):
    """Frame-indexed source of 3D detections."""

    @abstractmethod
    def frame_ids(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def get_frame_detections(self, frame_id: int) -> FrameDetections:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.frame_ids())

    def iter_frames(self):
        for frame_id in self.frame_ids():
            yield self.get_frame_detections(frame_id)
