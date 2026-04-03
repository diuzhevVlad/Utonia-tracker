from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .encoder import UtoniaFrameEncoder
from .types import Detection3D, FrameDetections, Track3D, Box3D, normalize_np


def _new_track(next_track_id: int, detection: Detection3D) -> Track3D:
    return Track3D(
        track_id=next_track_id,
        box=detection.box,
        label=detection.label,
        score=detection.score,
        prototype=detection.feature,
    )


def _age_track(track: Track3D) -> None:
    track.box = Box3D(
        center=track.box.center + track.velocity,
        size=track.box.size,
        yaw=track.box.yaw,
        coordinate_frame=track.box.coordinate_frame,
    )
    track.missed += 1


def _update_track(track: Track3D, detection: Detection3D, proto_momentum: float = 1.0) -> None:
    track.velocity = detection.box.center - track.box.center
    track.box = detection.box
    track.score = detection.score
    if detection.feature is not None:
        if track.prototype is None or proto_momentum >= 1.0:
            track.prototype = detection.feature
        else:
            track.prototype = normalize_np(
                proto_momentum * track.prototype
                + (1.0 - proto_momentum) * detection.feature
            )
    track.hits += 1
    track.missed = 0


class MOTracker:
    """Simple online multi-object tracker based on box center distance."""

    def __init__(
        self,
        max_match_distance: float = 5.0,
        max_missed: int = 2,
    ) -> None:
        self.max_match_distance = max_match_distance
        self.max_missed = max_missed
        self._next_track_id = 1
        self._tracks: list[Track3D] = []

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks = []

    def update(self, frame: FrameDetections) -> list[Track3D]:
        detections = frame.detections
        if not self._tracks:
            self._tracks = [self._spawn_track(detection) for detection in detections]
            return self.tracks()

        matches, unmatched_track_ids, unmatched_detection_ids = self._match_detections(
            detections
        )

        for track_id, detection_id in matches:
            _update_track(self._tracks[track_id], detections[detection_id])

        for track_id in unmatched_track_ids:
            _age_track(self._tracks[track_id])

        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        for detection_id in unmatched_detection_ids:
            self._tracks.append(self._spawn_track(detections[detection_id]))

        return self.tracks()

    def tracks(self) -> list[Track3D]:
        return list(self._tracks)

    def _spawn_track(self, detection: Detection3D) -> Track3D:
        track = _new_track(self._next_track_id, detection)
        self._next_track_id += 1
        return track

    def _match_detections(
        self,
        detections: list[Detection3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not detections:
            return [], list(range(len(self._tracks))), []

        invalid_cost = self.max_match_distance + 1.0
        cost = np.full(
            (len(self._tracks), len(detections)),
            fill_value=invalid_cost,
            dtype=np.float32,
        )
        for track_id, track in enumerate(self._tracks):
            predicted_center = track.box.center + track.velocity
            for detection_id, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                distance = np.linalg.norm(predicted_center - detection.box.center)
                if distance <= self.max_match_distance:
                    cost[track_id, detection_id] = distance

        return self._solve_assignment(cost, invalid_cost, len(detections))

    def _solve_assignment(
        self,
        cost: np.ndarray,
        invalid_cost: float,
        num_detections: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        row_ids, col_ids = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        matched_track_ids: set[int] = set()
        matched_detection_ids: set[int] = set()
        for track_id, detection_id in zip(row_ids.tolist(), col_ids.tolist()):
            if cost[track_id, detection_id] >= invalid_cost:
                continue
            matches.append((track_id, detection_id))
            matched_track_ids.add(track_id)
            matched_detection_ids.add(detection_id)

        unmatched_track_ids = [
            track_id for track_id in range(len(self._tracks)) if track_id not in matched_track_ids
        ]
        unmatched_detection_ids = [
            detection_id
            for detection_id in range(num_detections)
            if detection_id not in matched_detection_ids
        ]
        return matches, unmatched_track_ids, unmatched_detection_ids


class UtoniaMOTracker(MOTracker):
    """Online MOT that matches boxes with motion and Utonia box features."""

    def __init__(
        self,
        model=None,
        device: str | None = None,
        scale: float = 0.2,
        max_match_distance: float = 3.0,
        max_missed: int = 2,
        appearance_weight: float = 1.0,
        motion_weight: float = 0.7,
        min_points: int = 64,
        box_margin: float = 1.1,
        proto_momentum: float = 0.9,
    ) -> None:
        super().__init__(
            max_match_distance=max_match_distance,
            max_missed=max_missed,
        )
        self.encoder = UtoniaFrameEncoder(model=model, device=device, scale=scale)
        self.device = self.encoder.device
        self.scale = scale
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.min_points = min_points
        self.box_margin = box_margin
        self.proto_momentum = proto_momentum

    @property
    def model(self):
        return self.encoder.model

    @property
    def transform(self):
        return self.encoder.transform

    def build_model(self) -> None:
        self.encoder.build_model()

    def build_transform(self) -> None:
        self.encoder.build_transform()

    def preprocess(self, coord: np.ndarray):
        return self.encoder.preprocess(coord)

    def encode_frame(self, coord: np.ndarray):
        return self.encoder.encode_frame(coord)

    def update(self, coord: np.ndarray, frame: FrameDetections) -> list[Track3D]:
        coord_t, feat_t = self.encode_frame(coord)
        detections = self._attach_detection_features(frame.detections, coord_t, feat_t)
        if not self._tracks:
            self._tracks = [self._spawn_track(detection) for detection in detections]
            return self.tracks()

        matches, unmatched_track_ids, unmatched_detection_ids = self._match_detections(
            detections
        )

        for track_id, detection_id in matches:
            _update_track(
                self._tracks[track_id],
                detections[detection_id],
                proto_momentum=self.proto_momentum,
            )

        for track_id in unmatched_track_ids:
            _age_track(self._tracks[track_id])

        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        for detection_id in unmatched_detection_ids:
            self._tracks.append(self._spawn_track(detections[detection_id]))

        return self.tracks()

    def _attach_detection_features(
        self,
        detections: list[Detection3D],
        coord_t,
        feat_t,
    ) -> list[Detection3D]:
        for detection in detections:
            detection.feature = self.encoder.box_feature(
                coord_t,
                feat_t,
                detection.box,
                min_points=self.min_points,
                box_margin=self.box_margin,
            )
        return detections

    def _match_detections(
        self,
        detections: list[Detection3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not detections:
            return [], list(range(len(self._tracks))), []

        invalid_cost = self.motion_weight + self.appearance_weight + 1.0
        cost = np.full(
            (len(self._tracks), len(detections)),
            fill_value=invalid_cost,
            dtype=np.float32,
        )
        for track_id, track in enumerate(self._tracks):
            predicted_center = track.box.center + track.velocity
            for detection_id, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                distance = np.linalg.norm(predicted_center - detection.box.center)
                if distance > self.max_match_distance:
                    continue
                similarity = 0.0
                if track.prototype is not None and detection.feature is not None:
                    similarity = float(track.prototype @ detection.feature)
                motion_cost = distance / self.max_match_distance
                appearance_cost = 1.0 - max(similarity, -1.0)
                cost[track_id, detection_id] = (
                    self.motion_weight * motion_cost
                    + self.appearance_weight * appearance_cost
                )

        return self._solve_assignment(cost, invalid_cost, len(detections))
