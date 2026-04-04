from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from .config import (
    AssociationConfig,
    FeatureCropConfig,
    MotionModelConfig,
    RecoveryConfig,
    SpawnConfig,
    build_class_configs,
)
from .encoder import UtoniaFrameEncoder
from .geometry import bev_iou
from .motion import PredictedMotion, build_motion_model
from .types import Detection3D, FrameDetections, Track3D, Box3D, normalize_np


def _new_track(next_track_id: int, detection: Detection3D) -> Track3D:
    """Create a new track from a detection."""

    return Track3D(
        track_id=next_track_id,
        box=detection.box,
        label=detection.label,
        score=detection.score,
        prototype=detection.feature,
        metadata=dict(detection.metadata),
    )


def _age_track(track: Track3D, predicted: PredictedMotion) -> None:
    """Advance an unmatched track using only the motion prediction."""

    track.box = Box3D(
        center=predicted.center,
        size=track.box.size,
        yaw=track.box.yaw,
        coordinate_frame=track.box.coordinate_frame,
    )
    track.velocity = predicted.velocity
    track.missed += 1
    track.state = "lost"


def _update_track(
    track: Track3D,
    detection: Detection3D,
    predicted: PredictedMotion,
    proto_momentum: float = 1.0,
) -> None:
    """Update one track from a matched detection or recovered pseudo-detection."""

    track.velocity = predicted.velocity
    track.box = Box3D(
        center=predicted.center,
        size=detection.box.size,
        yaw=detection.box.yaw,
        coordinate_frame=detection.box.coordinate_frame,
    )
    track.score = detection.score
    track.metadata = dict(detection.metadata)
    track.state = str(detection.metadata.get("track_state", "detected"))
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
    """Configurable online MOT baseline with motion and BEV overlap."""

    def __init__(
        self,
        max_match_distance: float = 5.0,
        max_missed: int = 2,
        use_class_thresholds: bool = True,
        enable_bev_iou: bool = True,
        motion_model: MotionModelConfig | None = None,
        association_config: AssociationConfig | None = None,
        class_configs: dict[str, AssociationConfig] | None = None,
        spawn: SpawnConfig | None = None,
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
        self.spawn = spawn or SpawnConfig()
        self.last_profile: dict[str, float | int] = {}
        self._profile_totals: dict[str, float] = {}
        self._profile_frames = 0

    def reset(self) -> None:
        """Clear all tracks and accumulated profiling state."""

        self._next_track_id = 1
        self._tracks = []
        self.last_profile = {}
        self._profile_totals = {}
        self._profile_frames = 0

    def update(self, frame: FrameDetections) -> list[Track3D]:
        """Predict, associate, age, and spawn tracks for one frame."""

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

        # Unmatched detections can start new tracks unless they overlap a strong one.
        start_spawn = time.perf_counter()
        suppressed_spawns = 0
        for detection_id in unmatched_detection_ids:
            detection = detections[detection_id]
            if self._should_suppress_spawn(detection):
                suppressed_spawns += 1
                continue
            self._tracks.append(self._spawn_track(detection))
        profile["spawn_s"] = time.perf_counter() - start_spawn
        profile["num_suppressed_spawns"] = suppressed_spawns

        profile["num_tracks_after"] = len(self._tracks)
        profile["total_s"] = time.perf_counter() - start_total
        self._commit_profile(profile)
        return self.tracks()

    def tracks(self) -> list[Track3D]:
        """Return a shallow copy of active tracks."""

        return list(self._tracks)

    def profile_summary(self) -> dict[str, float]:
        """Return average profiling values accumulated across frames."""

        if self._profile_frames == 0:
            return {}
        summary = {"frames": float(self._profile_frames)}
        for key, value in self._profile_totals.items():
            summary[key] = value / self._profile_frames
        return summary

    def _commit_profile(self, profile: dict[str, float | int]) -> None:
        """Store per-frame profiling and accumulate numeric totals."""

        self.last_profile = profile
        self._profile_frames += 1
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                self._profile_totals[key] = self._profile_totals.get(key, 0.0) + float(value)

    def _spawn_track(self, detection: Detection3D) -> Track3D:
        """Create and initialize one new track."""

        track = _new_track(self._next_track_id, detection)
        self.motion_model.initialize(track)
        self._next_track_id += 1
        return track

    def _resolve_config(self, label: str) -> AssociationConfig:
        """Return class-specific association settings when present."""

        return self.class_configs.get(label, self.default_config)

    def _predicted_box(self, track: Track3D, predicted: PredictedMotion) -> Box3D:
        """Convert predicted motion back into a box-shaped state."""

        return Box3D(
            center=predicted.center,
            size=track.box.size,
            yaw=track.box.yaw,
            coordinate_frame=track.box.coordinate_frame,
        )

    def _should_suppress_spawn(self, detection: Detection3D) -> bool:
        """Reject duplicate births that overlap strong existing tracks."""

        if not self.spawn.enabled:
            return False
        for track in self._tracks:
            if not track.is_confirmed:
                continue
            if track.missed > self.spawn.max_track_missed:
                continue
            if track.state not in {"detected", "recovered"}:
                continue
            center_distance = np.linalg.norm(track.box.center - detection.box.center)
            if center_distance > self.spawn.max_center_distance:
                continue
            iou = bev_iou(track.box, detection.box)
            min_iou = (
                self.spawn.same_class_min_bev_iou
                if track.label == detection.label
                else self.spawn.cross_class_min_bev_iou
            )
            if iou >= min_iou:
                return True
        return False

    def _match_detections(
        self,
        detections: list[Detection3D],
        predicted_boxes: list[Box3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Build the assignment cost matrix and solve Hungarian matching."""

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
        """Return one association cost or `None` when the pair is gated out."""

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
        """Run Hungarian assignment and keep only valid pairs."""

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
    """MOT tracker that adds Utonia appearance and lost-track recovery."""

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
        recovery: RecoveryConfig | None = None,
        spawn: SpawnConfig | None = None,
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
            spawn=spawn,
        )
        self.encoder = UtoniaFrameEncoder(model=model, device=device, scale=scale)
        self.device = self.encoder.device
        self.scale = scale
        self.min_points = min_points
        self.box_margin = box_margin
        self.proto_momentum = proto_momentum
        self.feature_crop = feature_crop or FeatureCropConfig()
        self.recovery = recovery or RecoveryConfig()

    @property
    def model(self):
        return self.encoder.model

    @property
    def transform(self):
        return self.encoder.transform

    def build_model(self) -> None:
        """Expose model loading through the tracker API."""

        self.encoder.build_model()

    def build_transform(self) -> None:
        """Expose transform creation through the tracker API."""

        self.encoder.build_transform()

    def preprocess(self, coord: np.ndarray):
        """Forward preprocessing to the shared encoder."""

        return self.encoder.preprocess(coord)

    def encode_frame(self, coord: np.ndarray):
        """Forward frame encoding to the shared encoder."""

        return self.encoder.encode_frame(coord)

    def update(self, coord: np.ndarray, frame: FrameDetections) -> list[Track3D]:
        """Run detector matching first, then point-level recovery for lost tracks."""

        start_total = time.perf_counter()
        profile: dict[str, float | int] = {
            "num_detections": len(frame.detections),
            "num_tracks_before": len(self._tracks),
        }
        coord_t = None
        feat_t = None
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
        # Sparse crops skip appearance instead of falling back to full-frame encoding.
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

        # Recovery is a second pass only for unmatched confirmed tracks.
        start_recovery = time.perf_counter()
        recovered_track_ids: list[int] = []
        if coord_t is not None and feat_t is not None and self.recovery.enabled:
            recovered_track_ids = self._recover_tracks(
                track_ids=unmatched_track_ids,
                predicted_boxes=predicted_boxes,
                coord_t=coord_t,
                feat_t=feat_t,
            )
        profile["recovery_s"] = time.perf_counter() - start_recovery
        profile["num_recovered_tracks"] = len(recovered_track_ids)
        recovered_track_id_set = set(recovered_track_ids)
        unmatched_track_ids = [
            track_id for track_id in unmatched_track_ids if track_id not in recovered_track_id_set
        ]

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
        """Attach one pooled Utonia feature to each detection box."""

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
        """Choose which boxes define the ROI used for appearance encoding."""

        if self.feature_crop.mode == "full":
            return []
        boxes = [detection.box for detection in detections]
        if self.feature_crop.mode == "detections":
            return boxes
        return boxes + predicted_boxes

    def _recover_tracks(
        self,
        track_ids: list[int],
        predicted_boxes: list[Box3D],
        coord_t: torch.Tensor,
        feat_t: torch.Tensor,
    ) -> list[int]:
        """Try point-level recovery for unmatched confirmed tracks."""

        recovered_track_ids: list[int] = []
        available = torch.ones(coord_t.shape[0], dtype=torch.bool, device=coord_t.device)
        for track_id in track_ids:
            track = self._tracks[track_id]
            if not track.is_confirmed:
                continue
            if track.missed > self.recovery.max_missed:
                continue
            if track.prototype is None:
                continue

            # Recovery reuses the already-encoded crop; there is no extra forward pass.
            recovery = self._recover_track(
                track=track,
                predicted_box=predicted_boxes[track_id],
                coord_t=coord_t,
                feat_t=feat_t,
                available=available,
            )
            if recovery is None:
                continue

            detection, mask = recovery
            updated_motion = self.motion_model.update(track, detection.box.center)
            _update_track(
                track,
                detection,
                predicted=updated_motion,
                proto_momentum=self.recovery.proto_momentum,
            )
            available[mask] = False
            recovered_track_ids.append(track_id)
        return recovered_track_ids

    def _recover_track(
        self,
        track: Track3D,
        predicted_box: Box3D,
        coord_t: torch.Tensor,
        feat_t: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[Detection3D, torch.Tensor] | None:
        """Recover one lost track from point scores inside a local motion gate."""

        prototype = torch.as_tensor(track.prototype, dtype=torch.float32, device=coord_t.device)
        predicted_center = torch.as_tensor(
            predicted_box.center,
            dtype=torch.float32,
            device=coord_t.device,
        )
        dist = torch.linalg.norm(coord_t - predicted_center, dim=1)
        local_mask = available & (dist <= self.recovery.gate_radius)
        if int(local_mask.sum()) < self.recovery.min_points:
            return None

        local_indices = torch.nonzero(local_mask, as_tuple=False).flatten()
        local_coord = coord_t[local_mask]
        local_feat = feat_t[local_mask]
        sim = local_feat @ prototype
        local_dist = dist[local_mask]
        score = sim - self.recovery.score_alpha * (local_dist / self.recovery.gate_radius)
        anchor_local = torch.argmax(score)
        anchor_coord = local_coord[anchor_local]
        # Anchor on the best point, then keep a compact local cluster.
        support_mask = (
            torch.linalg.norm(local_coord - anchor_coord, dim=1) <= self.recovery.cluster_radius
        ) & (sim >= self.recovery.sim_threshold)
        if int(support_mask.sum()) < self.recovery.min_points:
            topk = torch.topk(score, k=min(self.recovery.min_points, score.numel())).indices
            support_mask = torch.zeros_like(score, dtype=torch.bool)
            support_mask[topk] = True

        if int(support_mask.sum()) < self.recovery.min_points:
            return None

        selected_coord = local_coord[support_mask]
        selected_feat = local_feat[support_mask]
        selected_sim = sim[support_mask]
        mean_similarity = float(selected_sim.mean().item())
        if mean_similarity < self.recovery.min_mean_similarity:
            return None

        recovered_center = selected_coord.mean(0)
        center_distance = float(torch.linalg.norm(recovered_center - predicted_center).item())
        if center_distance > self.recovery.max_center_distance:
            return None

        extent = selected_coord.max(0).values - selected_coord.min(0).values
        max_extent = torch.as_tensor(
            track.box.size * self.recovery.max_extent_scale,
            dtype=torch.float32,
            device=coord_t.device,
        )
        # Recovery keeps the previous box shape, so reject very spread-out support points.
        if torch.any(extent > max_extent):
            return None

        feature = F.normalize(selected_feat.mean(0), dim=0).detach().cpu().numpy()
        detection = Detection3D(
            box=Box3D(
                center=recovered_center.detach().cpu().numpy(),
                size=track.box.size.copy(),
                yaw=track.box.yaw,
                coordinate_frame=track.box.coordinate_frame,
            ),
            label=track.label,
            score=mean_similarity,
            feature=feature,
            source="utonia_recovery",
            metadata={
                **track.metadata,
                "track_state": "recovered",
                "recovered": True,
                "recovery_points": int(support_mask.sum().item()),
                "recovery_mean_similarity": mean_similarity,
            },
        )
        mask = torch.zeros_like(available)
        mask[local_indices[support_mask]] = True
        return detection, mask

    def _match_detections(
        self,
        detections: list[Detection3D],
        predicted_boxes: list[Box3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Reuse the base matcher; only the cost function changes."""

        return super()._match_detections(detections, predicted_boxes)

    def _association_cost(
        self,
        track: Track3D,
        predicted_box: Box3D,
        detection: Detection3D,
    ) -> float | None:
        """Add appearance cost on top of the geometry-only base cost."""

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
