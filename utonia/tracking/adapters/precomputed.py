from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import Box3D, Detection3D, DetectionSource, FrameDetections


class NpzDetectionSource(DetectionSource):
    """Read frame-indexed detections from a directory of `.npz` files."""

    def __init__(
        self,
        detections_dir: str | Path,
        frame_ids: list[int] | None = None,
        score_threshold: float = 0.0,
        coordinate_frame: str | None = None,
        source_name: str = "precomputed_npz",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.detections_dir = Path(detections_dir)
        if not self.detections_dir.is_dir():
            raise FileNotFoundError(f"Detection directory not found: {self.detections_dir}")

        self.score_threshold = float(score_threshold)
        self.source_name = source_name
        self.shared_metadata = dict(metadata or {})
        self.meta = self._load_meta()
        self.coordinate_frame = coordinate_frame or self._meta_scalar("coordinate_frame") or "lidar"

        available_frame_ids = sorted(
            int(path.stem)
            for path in self.detections_dir.glob("*.npz")
            if path.name != "meta.npz"
        )
        self._frame_ids = sorted(frame_ids if frame_ids is not None else available_frame_ids)
        self._available_frame_ids = set(available_frame_ids)

    def frame_ids(self) -> list[int]:
        return list(self._frame_ids)

    def get_frame_detections(self, frame_id: int) -> FrameDetections:
        if frame_id not in self._frame_ids:
            raise KeyError(f"Unknown frame_id {frame_id} for {self.detections_dir}")
        detections: list[Detection3D] = []
        if frame_id in self._available_frame_ids:
            detections = self._load_frame_detections(frame_id)
        return FrameDetections(
            frame_id=frame_id,
            detections=detections,
            metadata=dict(self.shared_metadata),
        )

    def _load_meta(self) -> dict[str, np.ndarray]:
        meta_path = self.detections_dir / "meta.npz"
        if not meta_path.is_file():
            return {}
        with np.load(meta_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def _meta_scalar(self, key: str) -> str | None:
        value = self.meta.get(key)
        if value is None or value.size == 0:
            return None
        return str(value.reshape(-1)[0])

    def _load_frame_detections(self, frame_id: int) -> list[Detection3D]:
        frame_path = self.detections_dir / f"{frame_id:06d}.npz"
        with np.load(frame_path, allow_pickle=False) as data:
            boxes = self._get_boxes(data)
            scores = np.asarray(data["scores"], dtype=np.float32)
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            label_ids = self._get_label_ids(data, keep)
            label_names = self._get_label_names(data, label_ids, keep)

        detections: list[Detection3D] = []
        for index, (box, score, label_id, label_name) in enumerate(
            zip(boxes, scores, label_ids, label_names)
        ):
            detection = Detection3D(
                box=Box3D(
                    center=box[:3],
                    size=box[3:6],
                    yaw=float(box[6]),
                    coordinate_frame=self.coordinate_frame,
                ),
                label=str(label_name),
                score=float(score),
                source=self.source_name,
                metadata={
                    **self.shared_metadata,
                    "frame_id": frame_id,
                    "detection_index": index,
                    "label_id": int(label_id),
                },
            )
            detections.append(self.enrich_detection(frame_id, detection))
        return detections

    def _get_boxes(self, data) -> np.ndarray:
        if "boxes_lidar" in data:
            return np.asarray(data["boxes_lidar"], dtype=np.float32)
        if "boxes" in data:
            return np.asarray(data["boxes"], dtype=np.float32)
        raise KeyError(f"Missing boxes_lidar/boxes in {data.files}")

    def _get_label_ids(self, data, keep: np.ndarray) -> np.ndarray:
        if "label_ids" in data:
            return np.asarray(data["label_ids"], dtype=np.int32)[keep]
        return np.zeros(int(np.count_nonzero(keep)), dtype=np.int32)

    def _get_label_names(self, data, label_ids: np.ndarray, keep: np.ndarray) -> np.ndarray:
        if "label_names" in data:
            return np.asarray(data["label_names"], dtype=str)[keep]
        class_names = self.meta.get("class_names")
        if class_names is None:
            return np.asarray(["Unknown"] * label_ids.shape[0], dtype=str)
        class_names = np.asarray(class_names, dtype=str)
        if label_ids.size == 0:
            return np.asarray([], dtype=str)
        indices = np.clip(label_ids - 1, 0, len(class_names) - 1)
        return class_names[indices]

    def enrich_detection(self, frame_id: int, detection: Detection3D) -> Detection3D:
        return detection
