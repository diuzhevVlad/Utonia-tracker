from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .precomputed import NpzDetectionSource
from ..types import Box3D, Detection3D, DetectionSource, FrameDetections, wrap_angle


KITTI_TRACKING_CLASSES = ("Car", "Pedestrian", "Cyclist", "Van")


@dataclass
class _KittiTrackingLabel:
    frame_id: int
    track_id: int
    label: str
    truncated: float
    occluded: int
    alpha: float
    bbox_2d: np.ndarray
    height: float
    width: float
    length: float
    location_camera: np.ndarray
    rotation_y: float


class KittiCalibration:
    def __init__(self, calib_path: str | Path) -> None:
        self.calib_path = Path(calib_path)
        if not self.calib_path.is_file():
            raise FileNotFoundError(f"KITTI calibration file not found: {self.calib_path}")
        values = self._parse_file(self.calib_path)
        projection = values.get("P2")
        if projection is None:
            raise KeyError(f"Missing P2 in {self.calib_path}")
        rect = values.get("R_rect")
        if rect is None:
            rect = values.get("R0_rect")
        if rect is None:
            raise KeyError(f"Missing R_rect/R0_rect in {self.calib_path}")
        tr_velo = values.get("Tr_velo_cam")
        if tr_velo is None:
            tr_velo = values.get("Tr_velo_to_cam")
        if tr_velo is None:
            raise KeyError(f"Missing Tr_velo_cam/Tr_velo_to_cam in {self.calib_path}")

        rect_4x4 = np.eye(4, dtype=np.float32)
        rect_4x4[:3, :3] = rect.reshape(3, 3)

        tr_velo_4x4 = np.eye(4, dtype=np.float32)
        tr_velo_4x4[:3, :] = tr_velo.reshape(3, 4)

        self.projection = projection.reshape(3, 4)
        self.rectified_cam_from_velodyne = rect_4x4 @ tr_velo_4x4
        self.velodyne_from_rectified_cam = np.linalg.inv(self.rectified_cam_from_velodyne)

    @staticmethod
    def _parse_file(calib_path: Path) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for line in calib_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, raw = line.split(":", maxsplit=1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid KITTI calibration line in {calib_path}: {line}"
                    )
                key, raw = parts
            values[key] = np.fromstring(raw, sep=" ", dtype=np.float32)
        return values

    def rect_to_velodyne(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_velo = points_h @ self.velodyne_from_rectified_cam.T
        return points_velo[:, :3]

    def velodyne_to_rect(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_rect = points_h @ self.rectified_cam_from_velodyne.T
        return points_rect[:, :3]

    def rect_to_image(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        proj = points_h @ self.projection.T
        depth = proj[:, 2]
        image = proj[:, :2] / np.clip(depth[:, None], 1e-6, None)
        return image, depth

    def velodyne_to_image(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.rect_to_image(self.velodyne_to_rect(points))

    def box_to_image_bbox(self, box: Box3D) -> np.ndarray | None:
        corners = self._box_corners_lidar(box)
        image, depth = self.velodyne_to_image(corners)
        valid = depth > 0.1
        if not np.any(valid):
            return None
        image = image[valid]
        return np.array(
            [
                np.min(image[:, 0]),
                np.min(image[:, 1]),
                np.max(image[:, 0]),
                np.max(image[:, 1]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _box_corners_lidar(box: Box3D) -> np.ndarray:
        length, width, height = box.size
        half_length = length / 2.0
        half_width = width / 2.0
        half_height = height / 2.0
        corners = np.array(
            [
                [half_length, half_width, half_height],
                [half_length, -half_width, half_height],
                [-half_length, -half_width, half_height],
                [-half_length, half_width, half_height],
                [half_length, half_width, -half_height],
                [half_length, -half_width, -half_height],
                [-half_length, -half_width, -half_height],
                [-half_length, half_width, -half_height],
            ],
            dtype=np.float32,
        )
        cos_yaw = np.cos(box.yaw)
        sin_yaw = np.sin(box.yaw)
        rotation = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return corners @ rotation.T + box.center[None, :]


class KittiGtDetectionSource(DetectionSource):
    """Load KITTI tracking ground-truth boxes as neutral 3D detections."""

    def __init__(
        self,
        sequence_dir: str | Path,
        label_path: str | Path | None = None,
        calib_path: str | Path | None = None,
        allowed_labels: tuple[str, ...] | None = KITTI_TRACKING_CLASSES,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {self.sequence_dir}")

        self.sequence_id = self.sequence_dir.name
        if label_path is None:
            label_path = self.sequence_dir.parents[1] / "label_02" / f"{self.sequence_id}.txt"
        if calib_path is None:
            calib_path = self.sequence_dir.parents[1] / "calib" / f"{self.sequence_id}.txt"

        self.label_path = Path(label_path)
        self.allowed_labels = None if allowed_labels is None else set(allowed_labels)
        self.calibration = KittiCalibration(calib_path)
        self._labels = self._parse_labels()
        self._frame_ids = self._discover_frame_ids()
        self._detections_by_frame = self._load_detections_by_frame()

    def frame_ids(self) -> list[int]:
        return list(self._frame_ids)

    def get_frame_detections(self, frame_id: int) -> FrameDetections:
        if frame_id not in self._detections_by_frame:
            raise KeyError(f"Unknown frame_id {frame_id} for sequence {self.sequence_id}")
        return self._detections_by_frame[frame_id]

    def _discover_frame_ids(self) -> list[int]:
        frame_ids = {int(path.stem) for path in self.sequence_dir.glob("*.bin")}
        frame_ids.update(label.frame_id for label in self._labels)
        if not frame_ids:
            raise ValueError(f"No KITTI frames found in {self.sequence_dir} or {self.label_path}")
        return sorted(frame_ids)

    def _load_detections_by_frame(self) -> dict[int, FrameDetections]:
        frames = {
            frame_id: FrameDetections(
                frame_id=frame_id,
                detections=[],
                metadata={"sequence_id": self.sequence_id},
            )
            for frame_id in self._frame_ids
        }
        for label in self._labels:
            if self.allowed_labels is not None and label.label not in self.allowed_labels:
                continue
            frames.setdefault(
                label.frame_id,
                FrameDetections(
                    frame_id=label.frame_id,
                    detections=[],
                    metadata={"sequence_id": self.sequence_id},
                ),
            )
            frames[label.frame_id].detections.append(self._label_to_detection(label))
        return frames

    def _parse_labels(self) -> list[_KittiTrackingLabel]:
        if not self.label_path.is_file():
            raise FileNotFoundError(f"KITTI label file not found: {self.label_path}")

        labels: list[_KittiTrackingLabel] = []
        for line in self.label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 17:
                raise ValueError(f"Invalid KITTI tracking label line: {line}")
            labels.append(
                _KittiTrackingLabel(
                    frame_id=int(parts[0]),
                    track_id=int(parts[1]),
                    label=parts[2],
                    truncated=float(parts[3]),
                    occluded=int(parts[4]),
                    alpha=float(parts[5]),
                    bbox_2d=np.asarray(parts[6:10], dtype=np.float32),
                    height=float(parts[10]),
                    width=float(parts[11]),
                    length=float(parts[12]),
                    location_camera=np.asarray(parts[13:16], dtype=np.float32),
                    rotation_y=float(parts[16]),
                )
            )
        return labels

    def _label_to_detection(self, label: _KittiTrackingLabel) -> Detection3D:
        center_camera = label.location_camera.copy()
        center_camera[1] -= label.height / 2.0
        center_lidar = self.calibration.rect_to_velodyne(center_camera)[0]
        yaw_lidar = wrap_angle(-(np.pi / 2.0 + label.rotation_y))
        return Detection3D(
            box=Box3D(
                center=center_lidar,
                size=np.array([label.length, label.width, label.height], dtype=np.float32),
                yaw=yaw_lidar,
                coordinate_frame="lidar",
            ),
            label=label.label,
            score=1.0,
            source="kitti_gt",
            metadata={
                "dataset": "KITTI",
                "sequence_id": self.sequence_id,
                "frame_id": label.frame_id,
                "track_id": label.track_id,
                "truncated": label.truncated,
                "occluded": label.occluded,
                "alpha": label.alpha,
                "bbox_2d": label.bbox_2d,
                "rotation_y": label.rotation_y,
            },
        )


class KittiPrecomputedDetectionSource(NpzDetectionSource):
    """Read precomputed detector outputs for one KITTI tracking sequence."""

    def __init__(
        self,
        sequence_dir: str | Path,
        detections_root: str | Path,
        calib_path: str | Path | None = None,
        score_threshold: float = 0.0,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {self.sequence_dir}")
        self.sequence_id = self.sequence_dir.name
        if calib_path is None:
            calib_path = self.sequence_dir.parents[1] / "calib" / f"{self.sequence_id}.txt"
        self.calibration = KittiCalibration(calib_path)
        detections_dir = Path(detections_root) / self.sequence_id
        frame_ids = sorted(int(path.stem) for path in self.sequence_dir.glob("*.bin"))
        super().__init__(
            detections_dir=detections_dir,
            frame_ids=frame_ids,
            score_threshold=score_threshold,
            coordinate_frame="lidar",
            source_name="kitti_precomputed",
            metadata={"dataset": "KITTI", "sequence_id": self.sequence_id},
        )

    def enrich_detection(self, frame_id: int, detection: Detection3D) -> Detection3D:
        detection.metadata.update(
            {
                "dataset": "KITTI",
                "sequence_id": self.sequence_id,
                "frame_id": frame_id,
            }
        )
        bbox_2d = self.calibration.box_to_image_bbox(detection.box)
        if bbox_2d is not None:
            detection.metadata["bbox_2d"] = bbox_2d
        return detection
