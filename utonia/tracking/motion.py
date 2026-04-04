from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MotionModelConfig
from .types import Box3D, Track3D


@dataclass
class PredictedMotion:
    center: np.ndarray
    velocity: np.ndarray


class VelocityMotionModel:
    def initialize(self, track: Track3D) -> None:
        track.motion_state = None

    def predict(self, track: Track3D) -> PredictedMotion:
        return PredictedMotion(
            center=track.box.center + track.velocity,
            velocity=track.velocity.copy(),
        )

    def update(self, track: Track3D, measurement: np.ndarray) -> PredictedMotion:
        velocity = measurement - track.box.center
        track.motion_state = None
        return PredictedMotion(center=measurement, velocity=velocity)


class KalmanMotionModel:
    def __init__(self, process_var: float = 1.0, measurement_var: float = 1.0) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var
        self._transition = np.eye(6, dtype=np.float32)
        self._transition[:3, 3:] = np.eye(3, dtype=np.float32)
        self._observation = np.zeros((3, 6), dtype=np.float32)
        self._observation[:3, :3] = np.eye(3, dtype=np.float32)

    def initialize(self, track: Track3D) -> None:
        mean = np.concatenate([track.box.center, track.velocity]).astype(np.float32)
        covariance = np.eye(6, dtype=np.float32)
        covariance[:3, :3] *= 10.0
        covariance[3:, 3:] *= 25.0
        track.motion_state = {"mean": mean, "covariance": covariance}

    def predict(self, track: Track3D) -> PredictedMotion:
        state = track.motion_state
        if state is None:
            self.initialize(track)
            state = track.motion_state
        mean = state["mean"]
        covariance = state["covariance"]
        process_noise = np.eye(6, dtype=np.float32) * self.process_var
        mean = self._transition @ mean
        covariance = self._transition @ covariance @ self._transition.T + process_noise
        state["mean"] = mean
        state["covariance"] = covariance
        return PredictedMotion(center=mean[:3].copy(), velocity=mean[3:].copy())

    def update(self, track: Track3D, measurement: np.ndarray) -> PredictedMotion:
        state = track.motion_state
        if state is None:
            self.initialize(track)
            state = track.motion_state
        mean = state["mean"]
        covariance = state["covariance"]
        measurement = np.asarray(measurement, dtype=np.float32)
        residual = measurement - self._observation @ mean
        measurement_noise = np.eye(3, dtype=np.float32) * self.measurement_var
        innovation = self._observation @ covariance @ self._observation.T + measurement_noise
        kalman_gain = covariance @ self._observation.T @ np.linalg.inv(innovation)
        mean = mean + kalman_gain @ residual
        covariance = (np.eye(6, dtype=np.float32) - kalman_gain @ self._observation) @ covariance
        state["mean"] = mean
        state["covariance"] = covariance
        return PredictedMotion(center=mean[:3].copy(), velocity=mean[3:].copy())


def build_motion_model(config: MotionModelConfig):
    if config.kind == "velocity":
        return VelocityMotionModel()
    if config.kind == "kalman":
        return KalmanMotionModel(
            process_var=config.process_var,
            measurement_var=config.measurement_var,
        )
    raise ValueError(f"Unknown motion model kind: {config.kind}")
