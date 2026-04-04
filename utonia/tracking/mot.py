from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import AssociationConfig, FeatureCropConfig, MotionModelConfig, build_class_configs
from .encoder import UtoniaFrameEncoder
from .geometry import bev_iou
from .motion import PredictedMotion, build_motion_model
from .types import Detection3D, FrameDetections, Track3D, Box3D, normalize_np


def _new_track(next_track_id: int, detection: Detection3D) -> Track3D:
    return Track3D(
        track_id=next_track_id,
        box=detection.box,
        label=detection.label,
        score=detection.score,
        prototype=detection.feature,
        metadata=dict(detection.metadata),
    )


def _age_track(track: Track3D, predicted: PredictedMotion) -> None:
    track.box = Box3D(
        center=predicted.center,
        size=track.box.size,
        yaw=track.box.yaw,
        coordinate_frame=track.box.coordinate_frame,
    )
    track.velocity = predicted.velocity
    track.missed += 1


def _update_track(
    track: Track3D,
    detection: Detection3D,
    predicted: PredictedMotion,
    proto_momentum: float = 1.0,
) -> None:
    track.velocity = predicted.velocity
    track.box = Box3D(
        center=predicted.center,
        size=detection.box.size,
        yaw=detection.box.yaw,
        coordinate_frame=detection.box.coordinate_frame,
    )
    track.score = detection.score
    track.metadata = dict(detection.metadata)
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
        use_class_thresholds: bool = True,
        enable_bev_iou: bool = True,
        motion_model: MotionModelConfig | None = None,
        association_config: AssociationConfig | None = None,
        class_configs: dict[str, AssociationConfig] | None = None,
    ) -> None:
        base_config = association_config or AssociationConfig(
            max_match_distance=max_match_distance,
            max_missed=max_missed,
            motion_weight=1.0,
            bev_iou_weight=1.0 if enable_bev_iou else 0.0,
            appearance_weight=0.0,
        )
        self.default_config = base_config
        self.class_configs = build_class_configs(base_config, enabled=use_class_thresholds)
        if class_configs is not None:
            self.class_configs.update(class_configs)
        self._next_track_id = 1
        self._tracks: list[Track3D] = []
        self.motion_model_config = motion_model or MotionModelConfig(kind="kalman")
        self.motion_model = build_motion_model(self.motion_model_config)
        self.last_profile: dict[str, float | int] = {}
        self._profile_totals: dict[str, float] = {}
        self._profile_frames = 0

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks = []
        self.last_profile = {}
        self._profile_totals = {}
        self._profile_frames = 0

    def update(self, frame: FrameDetections) -> list[Track3D]:
        start_total = time.perf_counter()
        detections = frame.detections
        profile: dict[str, float | int] = {
            "num_detections": len(detections),
            "num_tracks_before": len(self._tracks),
        }
        start_predict = time.perf_counter()
        predicted_motions = [self.motion_model.predict(track) for track in self._tracks]
        predicted_boxes = [self._predicted_box(track, predicted) for track, predicted in zip(self._tracks, predicted_motions)]
        profile["predict_s"] = time.perf_counter() - start_predict
        if not self._tracks:
            start_spawn = time.perf_counter()
            self._tracks = [self._spawn_track(detection) for detection in detections]
            profile["spawn_s"] = time.perf_counter() - start_spawn
            profile["num_tracks_after"] = len(self._tracks)
            profile["total_s"] = time.perf_counter() - start_total
            self._commit_profile(profile)
            return self.tracks()

        start_match = time.perf_counter()
        matches, unmatched_track_ids, unmatched_detection_ids = self._match_detections(
            detections,
            predicted_boxes,
        )
        profile["matching_s"] = time.perf_counter() - start_match
        profile["num_matches"] = len(matches)
        profile["num_unmatched_tracks"] = len(unmatched_track_ids)
        profile["num_unmatched_detections"] = len(unmatched_detection_ids)

        start_update = time.perf_counter()
        for track_id, detection_id in matches:
            updated_motion = self.motion_model.update(
                self._tracks[track_id],
                detections[detection_id].box.center,
            )
            _update_track(
                self._tracks[track_id],
                detections[detection_id],
                predicted=updated_motion,
            )
        profile["matched_update_s"] = time.perf_counter() - start_update

        start_age = time.perf_counter()
        for track_id in unmatched_track_ids:
            _age_track(self._tracks[track_id], predicted_motions[track_id])
        profile["age_s"] = time.perf_counter() - start_age

        start_filter = time.perf_counter()
        self._tracks = [
            track
            for track in self._tracks
            if track.missed <= self._resolve_config(track.label).max_missed
        ]
        profile["filter_s"] = time.perf_counter() - start_filter

        start_spawn = time.perf_counter()
        for detection_id in unmatched_detection_ids:
            self._tracks.append(self._spawn_track(detections[detection_id]))
        profile["spawn_s"] = time.perf_counter() - start_spawn

        profile["num_tracks_after"] = len(self._tracks)
        profile["total_s"] = time.perf_counter() - start_total
        self._commit_profile(profile)
        return self.tracks()

    def tracks(self) -> list[Track3D]:
        return list(self._tracks)

    def profile_summary(self) -> dict[str, float]:
        if self._profile_frames == 0:
            return {}
        summary = {"frames": float(self._profile_frames)}
        for key, value in self._profile_totals.items():
            summary[key] = value / self._profile_frames
        return summary

    def _commit_profile(self, profile: dict[str, float | int]) -> None:
        self.last_profile = profile
        self._profile_frames += 1
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                self._profile_totals[key] = self._profile_totals.get(key, 0.0) + float(value)

    def _spawn_track(self, detection: Detection3D) -> Track3D:
        track = _new_track(self._next_track_id, detection)
        self.motion_model.initialize(track)
        self._next_track_id += 1
        return track

    def _resolve_config(self, label: str) -> AssociationConfig:
        return self.class_configs.get(label, self.default_config)

    def _predicted_box(self, track: Track3D, predicted: PredictedMotion) -> Box3D:
        return Box3D(
            center=predicted.center,
            size=track.box.size,
            yaw=track.box.yaw,
            coordinate_frame=track.box.coordinate_frame,
        )

    def _match_detections(
        self,
        detections: list[Detection3D],
        predicted_boxes: list[Box3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not detections:
            return [], list(range(len(self._tracks))), []

        invalid_cost = 1e6
        cost = np.full(
            (len(self._tracks), len(detections)),
            fill_value=invalid_cost,
            dtype=np.float32,
        )
        for track_id, track in enumerate(self._tracks):
            predicted_box = predicted_boxes[track_id]
            for detection_id, detection in enumerate(detections):
                candidate_cost = self._association_cost(track, predicted_box, detection)
                if candidate_cost is not None:
                    cost[track_id, detection_id] = candidate_cost

        return self._solve_assignment(cost, invalid_cost, len(detections))

    def _association_cost(
        self,
        track: Track3D,
        predicted_box: Box3D,
        detection: Detection3D,
    ) -> float | None:
        if track.label != detection.label:
            return None
        config = self._resolve_config(track.label)
        distance = np.linalg.norm(predicted_box.center - detection.box.center)
        if distance > config.max_match_distance:
            return None
        iou = bev_iou(predicted_box, detection.box) if config.bev_iou_weight > 0.0 else 0.0
        if iou < config.min_bev_iou:
            return None
        motion_cost = (
            distance / max(config.max_match_distance, 1e-6)
            if config.motion_weight > 0.0
            else 0.0
        )
        iou_cost = (1.0 - iou) if config.bev_iou_weight > 0.0 else 0.0
        return float(config.motion_weight * motion_cost + config.bev_iou_weight * iou_cost)

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
        bev_iou_weight: float = 1.0,
        min_points: int = 64,
        box_margin: float = 1.1,
        proto_momentum: float = 0.9,
        feature_crop: FeatureCropConfig | None = None,
        motion_model: MotionModelConfig | None = None,
        use_class_thresholds: bool = True,
        enable_bev_iou: bool = True,
        class_configs: dict[str, AssociationConfig] | None = None,
    ) -> None:
        super().__init__(
            max_match_distance=max_match_distance,
            max_missed=max_missed,
            use_class_thresholds=use_class_thresholds,
            enable_bev_iou=enable_bev_iou,
            motion_model=motion_model,
            association_config=AssociationConfig(
                max_match_distance=max_match_distance,
                max_missed=max_missed,
                motion_weight=motion_weight,
                bev_iou_weight=bev_iou_weight if enable_bev_iou else 0.0,
                appearance_weight=appearance_weight,
            ),
            class_configs=class_configs,
        )
        self.encoder = UtoniaFrameEncoder(model=model, device=device, scale=scale)
        self.device = self.encoder.device
        self.scale = scale
        self.min_points = min_points
        self.box_margin = box_margin
        self.proto_momentum = proto_momentum
        self.feature_crop = feature_crop or FeatureCropConfig()

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
        start_total = time.perf_counter()
        profile: dict[str, float | int] = {
            "num_detections": len(frame.detections),
            "num_tracks_before": len(self._tracks),
        }
        start_predict = time.perf_counter()
        predicted_motions = [self.motion_model.predict(track) for track in self._tracks]
        predicted_boxes = [self._predicted_box(track, predicted) for track, predicted in zip(self._tracks, predicted_motions)]
        profile["predict_s"] = time.perf_counter() - start_predict
        start_roi = time.perf_counter()
        roi_boxes = self._feature_crop_boxes(frame.detections, predicted_boxes)
        coord_roi, roi_stats = self.encoder.select_roi_coord(
            coord,
            roi_boxes,
            mode=self.feature_crop.mode,
            crop_margin=self.feature_crop.crop_margin,
            min_points=self.feature_crop.min_points,
        )
        profile["feature_crop_s"] = time.perf_counter() - start_roi
        for key, value in roi_stats.items():
            profile[key] = value
        if coord_roi is None:
            profile["encode_frame_s"] = 0.0
            profile["box_features_s"] = 0.0
            profile["appearance_skipped"] = 1
            detections = frame.detections
        else:
            start_encode = time.perf_counter()
            coord_t, feat_t = self.encode_frame(coord_roi)
            profile["encode_frame_s"] = time.perf_counter() - start_encode
            start_feature = time.perf_counter()
            detections = self._attach_detection_features(frame.detections, coord_t, feat_t)
            profile["box_features_s"] = time.perf_counter() - start_feature
            profile["appearance_skipped"] = 0
        if not self._tracks:
            start_spawn = time.perf_counter()
            self._tracks = [self._spawn_track(detection) for detection in detections]
            profile["spawn_s"] = time.perf_counter() - start_spawn
            profile["num_tracks_after"] = len(self._tracks)
            profile["total_s"] = time.perf_counter() - start_total
            self._commit_profile(profile)
            return self.tracks()

        start_match = time.perf_counter()
        matches, unmatched_track_ids, unmatched_detection_ids = self._match_detections(
            detections,
            predicted_boxes,
        )
        profile["matching_s"] = time.perf_counter() - start_match
        profile["num_matches"] = len(matches)
        profile["num_unmatched_tracks"] = len(unmatched_track_ids)
        profile["num_unmatched_detections"] = len(unmatched_detection_ids)

        start_update = time.perf_counter()
        for track_id, detection_id in matches:
            updated_motion = self.motion_model.update(
                self._tracks[track_id],
                detections[detection_id].box.center,
            )
            _update_track(
                self._tracks[track_id],
                detections[detection_id],
                predicted=updated_motion,
                proto_momentum=self.proto_momentum,
            )
        profile["matched_update_s"] = time.perf_counter() - start_update

        start_age = time.perf_counter()
        for track_id in unmatched_track_ids:
            _age_track(self._tracks[track_id], predicted_motions[track_id])
        profile["age_s"] = time.perf_counter() - start_age

        start_filter = time.perf_counter()
        self._tracks = [
            track
            for track in self._tracks
            if track.missed <= self._resolve_config(track.label).max_missed
        ]
        profile["filter_s"] = time.perf_counter() - start_filter

        start_spawn = time.perf_counter()
        for detection_id in unmatched_detection_ids:
            self._tracks.append(self._spawn_track(detections[detection_id]))
        profile["spawn_s"] = time.perf_counter() - start_spawn

        profile["num_tracks_after"] = len(self._tracks)
        profile["total_s"] = time.perf_counter() - start_total
        self._commit_profile(profile)
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

    def _feature_crop_boxes(
        self,
        detections: list[Detection3D],
        predicted_boxes: list[Box3D],
    ) -> list[Box3D]:
        if self.feature_crop.mode == "full":
            return []
        boxes = [detection.box for detection in detections]
        if self.feature_crop.mode == "detections":
            return boxes
        return boxes + predicted_boxes

    def _match_detections(
        self,
        detections: list[Detection3D],
        predicted_boxes: list[Box3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        return super()._match_detections(detections, predicted_boxes)

    def _association_cost(
        self,
        track: Track3D,
        predicted_box: Box3D,
        detection: Detection3D,
    ) -> float | None:
        base_cost = super()._association_cost(track, predicted_box, detection)
        if base_cost is None:
            return None
        config = self._resolve_config(track.label)
        if (
            config.appearance_weight <= 0.0
            or track.prototype is None
            or detection.feature is None
        ):
            return base_cost
        similarity = float(track.prototype @ detection.feature)
        appearance_cost = 1.0 - max(similarity, -1.0)
        return float(base_cost + config.appearance_weight * appearance_cost)
