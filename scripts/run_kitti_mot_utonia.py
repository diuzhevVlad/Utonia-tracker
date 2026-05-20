import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.utils.box_utils import (  # noqa: E402
    boxes3d_kitti_camera_to_imageboxes,
    boxes3d_lidar_to_kitti_camera,
)
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from scripts.kitti_tracker_test import load_xyz  # noqa: E402
from utonia import UtoniaTracker  # noqa: E402


KITTI_TEST_SEQUENCES = ["0019", "0020"]
LABEL_TO_CLASS = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
CLASS_TO_LABEL = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}


@dataclass
class Detection:
    det_index: int
    class_name: str
    box_lidar: np.ndarray
    score: float
    box_camera: np.ndarray
    bbox_2d: np.ndarray
    embedding: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    class_name: str
    box_lidar: np.ndarray
    score: float
    embedding: np.ndarray | None
    velocity: np.ndarray
    lost_age: int = 0
    age: int = 1
    hits: int = 1
    last_detector_frame: int | None = None
    utonia_tracker: UtoniaTracker | None = None
    utonia_center_offset: np.ndarray | None = None
    fallback_age: int = 0

    def predict_box(self) -> np.ndarray:
        box = self.box_lidar.copy()
        box[:3] += self.velocity
        return box


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with velodyne/, calib/, label_02/, and optionally image_02/.",
    )
    parser.add_argument("--sequences", nargs="+", default=KITTI_TEST_SEQUENCES)
    parser.add_argument("--classes", nargs="+", default=["Car", "Pedestrian"])
    parser.add_argument(
        "--detections-root",
        default=str(REPO_ROOT / "data" / "detections" / "pointpillar" / "npz"),
        help="Detection root containing <sequence>/<frame>.npz.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument(
        "--low-score-thresh",
        type=float,
        default=None,
        help="Optional low-score detector threshold used only to continue existing tracks, ByteTrack-style.",
    )
    parser.add_argument(
        "--car-low-score-thresh",
        type=float,
        default=None,
        help="Optional Car-specific low-score threshold; defaults to --low-score-thresh.",
    )
    parser.add_argument(
        "--pedestrian-low-score-thresh",
        type=float,
        default=None,
        help="Optional Pedestrian-specific low-score threshold; defaults to --low-score-thresh.",
    )
    parser.add_argument(
        "--use-low-score-continuation",
        action="store_true",
        help="Run a second association pass with low-score detections, but do not start new tracks from them.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--car-match-distance", type=float, default=5.0)
    parser.add_argument("--pedestrian-match-distance", type=float, default=2.0)
    parser.add_argument("--low-score-max-cost", type=float, default=None)
    parser.add_argument("--low-score-min-iou", type=float, default=None)
    parser.add_argument("--low-score-max-distance-car", type=float, default=None)
    parser.add_argument("--low-score-max-distance-pedestrian", type=float, default=None)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument(
        "--appearance-mode",
        choices=["none", "utonia"],
        default="none",
        help="Use Utonia box embeddings in association when set to utonia.",
    )
    parser.add_argument("--embedding-momentum", type=float, default=0.8)
    parser.add_argument("--local-crop-radius", type=float, default=8.0)
    parser.add_argument("--local-crop-min-points", type=int, default=2048)
    parser.add_argument("--box-embedding-min-points", type=int, default=8)
    parser.add_argument(
        "--validate-low-score-utonia",
        action="store_true",
        help="Use Utonia to validate weak low-score detector updates before accepting them.",
    )
    parser.add_argument(
        "--validate-low-score-classes",
        nargs="+",
        default=["Pedestrian"],
        help="Classes whose low-score matches may be checked by Utonia validation.",
    )
    parser.add_argument(
        "--validate-low-score-below",
        type=float,
        default=None,
        help="Only validate low-score detections below this score; validate all allowed classes when omitted.",
    )
    parser.add_argument("--validate-low-score-sim-thresh", type=float, default=0.6)
    parser.add_argument("--validate-low-score-min-points-car", type=int, default=32)
    parser.add_argument("--validate-low-score-min-points-pedestrian", type=int, default=8)
    parser.add_argument(
        "--utonia-fallback",
        action="store_true",
        help="Use Utonia to continue unmatched existing tracks when detector association fails.",
    )
    parser.add_argument(
        "--utonia-fallback-classes",
        nargs="+",
        default=["Pedestrian"],
        help="Classes allowed to initialize or continue Utonia detector-gap fallback.",
    )
    parser.add_argument("--fallback-max-frames", type=int, default=5)
    parser.add_argument(
        "--fallback-max-candidates-per-frame",
        type=int,
        default=8,
        help="Maximum unmatched active tracks that may run Utonia fallback on one frame.",
    )
    parser.add_argument("--fallback-sim-thresh", type=float, default=0.55)
    parser.add_argument("--fallback-min-points-car", type=int, default=32)
    parser.add_argument("--fallback-min-points-pedestrian", type=int, default=8)
    parser.add_argument("--fallback-score", type=float, default=0.15)
    parser.add_argument(
        "--fallback-max-bbox-width",
        type=float,
        default=1200.0,
        help="Reject Utonia fallback projections wider than this many image pixels.",
    )
    parser.add_argument(
        "--fallback-max-bbox-height",
        type=float,
        default=1000.0,
        help="Reject Utonia fallback projections taller than this many image pixels.",
    )
    parser.add_argument(
        "--fallback-max-bbox-area-ratio",
        type=float,
        default=1.5,
        help="When image size is available, reject fallback 2D boxes larger than this image-area ratio.",
    )
    parser.add_argument(
        "--fallback-start-score-thresh",
        type=float,
        default=None,
        help="Only initialize Utonia fallback from tracks whose last detector score is at least this value.",
    )
    parser.add_argument(
        "--fallback-refine-box",
        choices=["none", "support_z", "support_3d"],
        default="none",
        help="Optionally refine fallback box geometry from Utonia support points.",
    )
    parser.add_argument(
        "--fallback-refine-padding",
        type=float,
        default=0.15,
        help="Meters of padding added to support-point extents during fallback box refinement.",
    )
    parser.add_argument(
        "--fallback-refine-min-size-ratio",
        type=float,
        default=0.6,
        help="Lower clamp for refined fallback dimensions relative to the previous box size.",
    )
    parser.add_argument(
        "--fallback-refine-max-size-ratio",
        type=float,
        default=1.4,
        help="Upper clamp for refined fallback dimensions relative to the previous box size.",
    )
    parser.add_argument(
        "--fallback-refine-center-alpha",
        type=float,
        default=0.5,
        help="Blend factor for support-point center shift during support_3d refinement.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--run-name", default="utonia_mot_pointpillar_motion_iou")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data" / "mot_kitti"))
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def alpha_from_camera_box(box_camera: np.ndarray) -> float:
    x, _, z, _, _, _, ry = box_camera
    return wrap_angle(float(ry - math.atan2(x, z)))


def axis_aligned_bev(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    half = boxes[:, 3:5] * 0.5
    return np.column_stack(
        [
            boxes[:, 0] - half[:, 0],
            boxes[:, 1] - half[:, 1],
            boxes[:, 0] + half[:, 0],
            boxes[:, 1] + half[:, 1],
        ]
    ).astype(np.float32)


def pairwise_iou_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def nms_by_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thresh: float) -> np.ndarray:
    keep_all = []
    bev = axis_aligned_bev(boxes)
    for label in sorted(set(int(label) for label in labels.tolist())):
        indices = np.flatnonzero(labels == label)
        order = indices[np.argsort(scores[indices])[::-1]]
        while len(order) > 0:
            current = order[0]
            keep_all.append(current)
            if len(order) == 1:
                break
            ious = pairwise_iou_2d(bev[[current]], bev[order[1:]])[0]
            order = order[1:][ious <= iou_thresh]
    return np.array(sorted(keep_all), dtype=np.int64)


def load_detections(path: Path, classes: set[str], score_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    data = np.load(path)
    boxes = data["pred_boxes"].astype(np.float32)
    scores = data["pred_scores"].astype(np.float32)
    labels = data["pred_labels"].astype(np.int32)
    keep = np.array(
        [
            score >= score_thresh and LABEL_TO_CLASS.get(int(label)) in classes
            for score, label in zip(scores, labels, strict=False)
        ],
        dtype=bool,
    )
    return boxes[keep], scores[keep], labels[keep]


def class_low_score_thresh(args, class_name: str) -> float | None:
    if class_name == "Car" and args.car_low_score_thresh is not None:
        return args.car_low_score_thresh
    if class_name == "Pedestrian" and args.pedestrian_low_score_thresh is not None:
        return args.pedestrian_low_score_thresh
    return args.low_score_thresh


def image_shape(data_root: Path, sequence: str, frame_id: int) -> tuple[int, int] | None:
    image_path = data_root / "image_02" / sequence / f"{frame_id:06d}.png"
    if not image_path.exists():
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image.shape[:2]


def valid_projected_box(box_camera: np.ndarray, bbox_2d: np.ndarray) -> bool:
    if not np.isfinite(box_camera).all() or not np.isfinite(bbox_2d).all():
        return False
    if box_camera[2] <= 0:
        return False
    return bool(bbox_2d[2] > bbox_2d[0] and bbox_2d[3] > bbox_2d[1])


def box_mask(coord: np.ndarray, box: np.ndarray) -> np.ndarray:
    local = coord - box[:3][None]
    c = math.cos(float(box[6]))
    s = math.sin(float(box[6]))
    rot_x = local[:, 0] * c + local[:, 1] * s
    rot_y = -local[:, 0] * s + local[:, 1] * c
    rot_z = local[:, 2]
    half = box[3:6] * 0.5
    return (
        (np.abs(rot_x) <= half[0])
        & (np.abs(rot_y) <= half[1])
        & (np.abs(rot_z) <= half[2])
    )


def detection_embeddings(
    tracker: UtoniaTracker | None,
    coord: np.ndarray,
    detections: list[Detection],
    min_points: int,
) -> None:
    if tracker is None or not detections:
        return
    for det in detections:
        mask = box_mask(coord, det.box_lidar)
        if int(mask.sum()) < min_points:
            det.embedding = None
            continue
        coord_t, feat, _ = tracker.encode_frame(coord, center=det.box_lidar[:3])
        local_mask = box_mask(coord_t.detach().cpu().numpy(), det.box_lidar)
        if int(local_mask.sum()) < min_points:
            det.embedding = None
            continue
        emb = feat[local_mask].mean(0)
        emb = emb / emb.norm().clamp_min(1e-6)
        det.embedding = emb.detach().cpu().numpy().astype(np.float32)


def build_detections(
    data_root: Path,
    detections_root: Path,
    sequence: str,
    frame_id: int,
    calib: Calibration,
    classes: set[str],
    score_thresh: float,
    nms_iou: float,
) -> list[Detection]:
    boxes, scores, labels = load_detections(
        detections_root / sequence / f"{frame_id:06d}.npz",
        classes,
        score_thresh,
    )
    if len(boxes) == 0:
        return []
    if nms_iou >= 0:
        keep = nms_by_class(boxes, scores, labels, nms_iou)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    cam_boxes = boxes3d_lidar_to_kitti_camera(boxes, calib)
    shape = image_shape(data_root, sequence, frame_id)
    image_boxes = boxes3d_kitti_camera_to_imageboxes(cam_boxes, calib, image_shape=shape)
    detections = []
    for det_index, (box, score, label, cam_box, image_box) in enumerate(
        zip(boxes, scores, labels, cam_boxes, image_boxes, strict=False)
    ):
        class_name = LABEL_TO_CLASS.get(int(label))
        if class_name not in classes or not valid_projected_box(cam_box, image_box):
            continue
        detections.append(
            Detection(
                det_index=det_index,
                class_name=class_name,
                box_lidar=box.astype(np.float32),
                score=float(score),
                box_camera=cam_box.astype(np.float32),
                bbox_2d=image_box.astype(np.float32),
            )
        )
    return detections


def split_high_low_detections(
    data_root: Path,
    detections_root: Path,
    sequence: str,
    frame_id: int,
    calib: Calibration,
    classes: set[str],
    high_score_thresh: float,
    low_score_thresh: float | None,
    car_low_score_thresh: float | None,
    pedestrian_low_score_thresh: float | None,
    nms_iou: float,
) -> tuple[list[Detection], list[Detection]]:
    high = build_detections(
        data_root=data_root,
        detections_root=detections_root,
        sequence=sequence,
        frame_id=frame_id,
        calib=calib,
        classes=classes,
        score_thresh=high_score_thresh,
        nms_iou=nms_iou,
    )
    def threshold_for(class_name: str) -> float | None:
        if class_name == "Car" and car_low_score_thresh is not None:
            return car_low_score_thresh
        if class_name == "Pedestrian" and pedestrian_low_score_thresh is not None:
            return pedestrian_low_score_thresh
        return low_score_thresh

    class_low_thresholds = [threshold for class_name in classes if (threshold := threshold_for(class_name)) is not None]
    if not class_low_thresholds:
        return high, []
    min_low_score_thresh = min(class_low_thresholds)
    if min_low_score_thresh >= high_score_thresh:
        return high, []
    all_low = build_detections(
        data_root=data_root,
        detections_root=detections_root,
        sequence=sequence,
        frame_id=frame_id,
        calib=calib,
        classes=classes,
        score_thresh=min_low_score_thresh,
        nms_iou=nms_iou,
    )
    low = [
        det
        for det in all_low
        if det.score < high_score_thresh
        and (threshold := threshold_for(det.class_name)) is not None
        and det.score >= threshold
    ]
    return high, low


def match_distance(class_name: str, args) -> float:
    if class_name == "Pedestrian":
        return args.pedestrian_match_distance
    return args.car_match_distance


def fallback_min_points(args, class_name: str) -> int:
    if class_name == "Pedestrian":
        return args.fallback_min_points_pedestrian
    return args.fallback_min_points_car


def validation_min_points(args, class_name: str) -> int:
    if class_name == "Pedestrian":
        return args.validate_low_score_min_points_pedestrian
    return args.validate_low_score_min_points_car


def pair_distance_iou(track: Track, det: Detection) -> tuple[float, float]:
    pred_box = track.predict_box()
    distance = float(np.linalg.norm(pred_box[:2] - det.box_lidar[:2]))
    iou = float(pairwise_iou_2d(axis_aligned_bev(pred_box[None]), axis_aligned_bev(det.box_lidar[None]))[0, 0])
    return distance, iou


def low_score_gate_reason(
    track: Track,
    det: Detection,
    cost: float,
    distance: float,
    iou: float,
    args,
) -> str:
    if args.low_score_max_cost is not None and cost > args.low_score_max_cost:
        return "cost"
    if args.low_score_min_iou is not None and iou < args.low_score_min_iou:
        return "iou"
    max_distance = (
        args.low_score_max_distance_pedestrian
        if track.class_name == "Pedestrian"
        else args.low_score_max_distance_car
    )
    if max_distance is not None and distance > max_distance:
        return "distance"
    return ""


def should_validate_low_score(track: Track, det: Detection, args) -> bool:
    if not args.validate_low_score_utonia:
        return False
    if track.class_name not in set(args.validate_low_score_classes):
        return False
    if args.validate_low_score_below is not None and det.score >= args.validate_low_score_below:
        return False
    return True


def validate_low_score_with_utonia(
    track: Track,
    det: Detection,
    coord: np.ndarray,
    shared_model: UtoniaTracker,
    args,
) -> tuple[bool, dict]:
    center = (track.predict_box()[:3] + det.box_lidar[:3]) * 0.5
    coord_t, feat, _ = shared_model.encode_frame(coord, center=center)
    coord_np = coord_t.detach().cpu().numpy()
    pred_mask = box_mask(coord_np, track.predict_box())
    det_mask = box_mask(coord_np, det.box_lidar)
    min_points = validation_min_points(args, track.class_name)
    pred_support = int(pred_mask.sum())
    det_support = int(det_mask.sum())
    if pred_support < min_points or det_support < min_points:
        return False, {
            "low_score_reject_reason": "utonia_sparse",
            "validation_similarity": None,
            "validation_pred_support": pred_support,
            "validation_det_support": det_support,
        }

    pred_mask_t = torch_bool_mask(pred_mask, feat)
    det_mask_t = torch_bool_mask(det_mask, feat)
    pred_emb = feat[pred_mask_t].mean(0)
    det_emb = feat[det_mask_t].mean(0)
    pred_emb = pred_emb / pred_emb.norm().clamp_min(1e-6)
    det_emb = det_emb / det_emb.norm().clamp_min(1e-6)
    similarity = float((pred_emb @ det_emb).detach().cpu())
    accepted = similarity >= args.validate_low_score_sim_thresh
    return accepted, {
        "low_score_reject_reason": "" if accepted else "utonia_similarity",
        "validation_similarity": similarity,
        "validation_pred_support": pred_support,
        "validation_det_support": det_support,
    }


def torch_bool_mask(mask: np.ndarray, feat) -> object:
    import torch

    return torch.as_tensor(mask, dtype=torch.bool, device=feat.device)


def association_cost(
    tracks: list[Track],
    detections: list[Detection],
    args,
) -> tuple[np.ndarray, np.ndarray]:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32), np.zeros((len(tracks), len(detections)), dtype=bool)

    pred_boxes = np.stack([track.predict_box() for track in tracks], axis=0)
    det_boxes = np.stack([det.box_lidar for det in detections], axis=0)
    distances = np.linalg.norm(pred_boxes[:, None, :2] - det_boxes[None, :, :2], axis=2)
    ious = pairwise_iou_2d(axis_aligned_bev(pred_boxes), axis_aligned_bev(det_boxes))
    cost = np.full_like(distances, 1e6, dtype=np.float32)
    allowed = np.zeros_like(distances, dtype=bool)
    for row, track in enumerate(tracks):
        gate = match_distance(track.class_name, args)
        for col, det in enumerate(detections):
            if track.class_name != det.class_name or distances[row, col] > gate:
                continue
            appearance_cost = 0.0
            if args.appearance_weight > 0 and track.embedding is not None and det.embedding is not None:
                appearance_cost = 1.0 - float(np.dot(track.embedding, det.embedding))
            cost[row, col] = (
                args.distance_weight * distances[row, col] / max(gate, 1e-6)
                + args.iou_weight * (1.0 - ious[row, col])
                + args.appearance_weight * appearance_cost
            )
            allowed[row, col] = True
    return cost, allowed


def update_track(track: Track, det: Detection, args) -> None:
    old_center = track.box_lidar[:3].copy()
    track.box_lidar = det.box_lidar.copy()
    track.velocity = track.box_lidar[:3] - old_center
    track.score = det.score
    track.lost_age = 0
    track.age += 1
    track.hits += 1
    track.fallback_age = 0
    track.utonia_tracker = None
    track.utonia_center_offset = None
    if det.embedding is not None:
        if track.embedding is None:
            track.embedding = det.embedding.copy()
        else:
            emb = args.embedding_momentum * track.embedding + (1.0 - args.embedding_momentum) * det.embedding
            norm = np.linalg.norm(emb)
            track.embedding = emb / max(norm, 1e-6)


def kitti_line_from_boxes(
    frame_id: int,
    track_id: int,
    class_name: str,
    box_camera: np.ndarray,
    bbox_2d: np.ndarray,
    score: float,
) -> str:
    box = box_camera
    bbox = bbox_2d
    alpha = alpha_from_camera_box(box)
    return (
        f"{frame_id:d} {track_id:d} {class_name} 0 0 {alpha:.6f} "
        f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
        f"{box[4]:.6f} {box[5]:.6f} {box[3]:.6f} "
        f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[6]:.6f} {score:.6f}"
    )


def kitti_line(frame_id: int, track: Track, det: Detection) -> str:
    return kitti_line_from_boxes(
        frame_id=frame_id,
        track_id=track.track_id,
        class_name=det.class_name,
        box_camera=det.box_camera,
        bbox_2d=det.bbox_2d,
        score=det.score,
    )


def mean_similarity(state: dict) -> float | None:
    sim = np.asarray(state.get("sim", []), dtype=np.float32)
    mask = np.asarray(state.get("mask", []), dtype=bool)
    if sim.size == 0 or mask.size == 0 or int(mask.sum()) == 0:
        return None
    return float(sim[mask].mean())


def project_lidar_box(
    data_root: Path,
    sequence: str,
    frame_id: int,
    calib: Calibration,
    box_lidar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    box_camera = boxes3d_lidar_to_kitti_camera(box_lidar[None].astype(np.float32), calib)[0]
    bbox_2d = boxes3d_kitti_camera_to_imageboxes(
        box_camera[None],
        calib,
        image_shape=image_shape(data_root, sequence, frame_id),
    )[0]
    if not valid_projected_box(box_camera, bbox_2d):
        return None
    return box_camera.astype(np.float32), bbox_2d.astype(np.float32)


def fallback_projection_rejection_reason(
    data_root: Path,
    sequence: str,
    frame_id: int,
    bbox_2d: np.ndarray,
    args,
) -> str | None:
    width = float(bbox_2d[2] - bbox_2d[0])
    height = float(bbox_2d[3] - bbox_2d[1])
    if width <= 0.0 or height <= 0.0:
        return "projection_failed"
    if width > args.fallback_max_bbox_width:
        return "bbox_too_wide"
    if height > args.fallback_max_bbox_height:
        return "bbox_too_tall"
    shape = image_shape(data_root, sequence, frame_id)
    if shape is not None:
        image_area = max(float(shape[0] * shape[1]), 1.0)
        if width * height / image_area > args.fallback_max_bbox_area_ratio:
            return "bbox_area_too_large"
    return None


def support_points_from_state(state: dict) -> np.ndarray:
    coord = np.asarray(state.get("coord", []), dtype=np.float32)
    mask = np.asarray(state.get("mask", []), dtype=bool)
    if coord.ndim != 2 or coord.shape[1] != 3 or mask.ndim != 1 or len(coord) != len(mask):
        return np.zeros((0, 3), dtype=np.float32)
    return coord[mask]


def refine_box_from_support(
    box_lidar: np.ndarray,
    state: dict,
    args,
) -> tuple[np.ndarray, dict]:
    if args.fallback_refine_box == "none":
        return box_lidar, {
            "fallback_refined": False,
            "fallback_refine_reason": "disabled",
        }

    support = support_points_from_state(state)
    if len(support) < 3:
        return box_lidar, {
            "fallback_refined": False,
            "fallback_refine_reason": "too_few_support_points",
        }

    refined = box_lidar.copy()
    heading = float(refined[6])
    c = math.cos(heading)
    s = math.sin(heading)
    local = support - refined[:3][None]
    local_x = local[:, 0] * c + local[:, 1] * s
    local_y = -local[:, 0] * s + local[:, 1] * c
    local_z = local[:, 2]
    local_points = np.column_stack([local_x, local_y, local_z])
    lower = np.percentile(local_points, 5, axis=0)
    upper = np.percentile(local_points, 95, axis=0)
    observed_size = np.maximum(
        upper - lower + 2.0 * args.fallback_refine_padding,
        1e-3,
    )
    min_size = refined[3:6] * args.fallback_refine_min_size_ratio
    max_size = refined[3:6] * args.fallback_refine_max_size_ratio
    new_size = np.clip(observed_size, min_size, max_size)
    local_center = (lower + upper) * 0.5

    if args.fallback_refine_box == "support_z":
        refined[2] = float(refined[2] + local_center[2])
        refined[5] = float(new_size[2])
    elif args.fallback_refine_box == "support_3d":
        alpha = float(np.clip(args.fallback_refine_center_alpha, 0.0, 1.0))
        shift_local = local_center * alpha
        shift_world = np.array(
            [
                shift_local[0] * c - shift_local[1] * s,
                shift_local[0] * s + shift_local[1] * c,
                shift_local[2],
            ],
            dtype=np.float32,
        )
        refined[:3] += shift_world
        refined[3:6] = new_size.astype(np.float32)

    return refined.astype(np.float32), {
        "fallback_refined": True,
        "fallback_refine_reason": args.fallback_refine_box,
        "refined_dx": float(refined[3]),
        "refined_dy": float(refined[4]),
        "refined_dz": float(refined[5]),
    }


def initialize_track_utonia(
    track: Track,
    coord: np.ndarray,
    shared_model: UtoniaTracker,
    args,
) -> bool:
    tracker = UtoniaTracker(
        model=shared_model.model,
        mode="local_crop",
        init_radius=1.6,
        cluster_radius=1.6 if track.class_name == "Car" else 0.7,
        gate_radius=match_distance(track.class_name, args),
        local_crop_radius=args.local_crop_radius,
        local_crop_min_points=args.local_crop_min_points,
        init_points=64 if track.class_name == "Car" else 16,
        min_points=64 if track.class_name == "Car" else 16,
        box_height_filter_ratio=0.15 if track.class_name == "Car" else 0.10,
        lost_height_ratio=0.3,
        lost_bad_frames=4,
    )
    state = tracker.initialize_from_box(coord, track.box_lidar)
    track.utonia_tracker = tracker
    track.utonia_center_offset = track.box_lidar[:3] - np.asarray(state["centroid"], dtype=np.float32)
    return True


def try_utonia_fallback(
    track: Track,
    frame_id: int,
    coord: np.ndarray,
    data_root: Path,
    sequence: str,
    calib: Calibration,
    shared_model: UtoniaTracker,
    args,
) -> tuple[bool, str | None, dict]:
    if track.fallback_age >= args.fallback_max_frames:
        return False, None, {"fallback_reason": "max_fallback_age"}
    if track.utonia_tracker is None:
        initialize_track_utonia(track, coord, shared_model, args)
    state = track.utonia_tracker.step(coord)
    support_count = int(state.get("support_count", 0))
    mean_sim = mean_similarity(state)
    jump_xy = state.get("jump_xy")
    accepted = (
        state.get("status") == "active"
        and support_count >= fallback_min_points(args, track.class_name)
        and mean_sim is not None
        and mean_sim >= args.fallback_sim_thresh
        and (jump_xy is None or float(jump_xy) <= match_distance(track.class_name, args))
    )
    if not accepted:
        track.fallback_age += 1
        return False, None, {
            "fallback_reason": state.get("bad_update_reason") or "gate_failed",
            "support_count": support_count,
            "mean_similarity": mean_sim,
            "jump_xy": jump_xy,
        }
    old_center = track.box_lidar[:3].copy()
    center_offset = track.utonia_center_offset if track.utonia_center_offset is not None else 0.0
    fallback_box = track.box_lidar.copy()
    fallback_box[:3] = np.asarray(state["centroid"], dtype=np.float32) + center_offset
    fallback_box, refine_meta = refine_box_from_support(fallback_box, state, args)
    projected = project_lidar_box(data_root, sequence, frame_id, calib, fallback_box)
    if projected is None:
        track.fallback_age += 1
        return False, None, {
            "fallback_reason": "projection_failed",
            "support_count": support_count,
            "mean_similarity": mean_sim,
            "jump_xy": jump_xy,
            **refine_meta,
        }
    box_camera, bbox_2d = projected
    projection_reason = fallback_projection_rejection_reason(
        data_root=data_root,
        sequence=sequence,
        frame_id=frame_id,
        bbox_2d=bbox_2d,
        args=args,
    )
    if projection_reason is not None:
        track.fallback_age += 1
        return False, None, {
            "fallback_reason": projection_reason,
            "support_count": support_count,
            "mean_similarity": mean_sim,
            "jump_xy": jump_xy,
            "projected_bbox_width": float(bbox_2d[2] - bbox_2d[0]),
            "projected_bbox_height": float(bbox_2d[3] - bbox_2d[1]),
            **refine_meta,
        }
    track.box_lidar = fallback_box
    track.velocity = track.box_lidar[:3] - old_center
    track.lost_age = 0
    track.age += 1
    track.hits += 1
    track.fallback_age += 1
    line = kitti_line_from_boxes(
        frame_id=frame_id,
        track_id=track.track_id,
        class_name=track.class_name,
        box_camera=box_camera,
        bbox_2d=bbox_2d,
        score=min(track.score, args.fallback_score),
    )
    return True, line, {
        "fallback_reason": "",
        "support_count": support_count,
        "mean_similarity": mean_sim,
        "jump_xy": jump_xy,
        "projected_bbox_width": float(bbox_2d[2] - bbox_2d[0]),
        "projected_bbox_height": float(bbox_2d[3] - bbox_2d[1]),
        **refine_meta,
    }


def should_try_utonia_fallback(track: Track, frame_id: int, args) -> bool:
    if track.class_name not in set(args.utonia_fallback_classes):
        return False
    if track.last_detector_frame is None:
        return False
    if frame_id - track.last_detector_frame > args.fallback_max_frames:
        return False
    if track.lost_age > 0 and track.utonia_tracker is None:
        return False
    if (
        track.utonia_tracker is None
        and args.fallback_start_score_thresh is not None
        and track.score < args.fallback_start_score_thresh
    ):
        return False
    return True


def run_sequence(args, sequence: str, output_dir: Path, tracker_model: UtoniaTracker | None) -> dict:
    data_root = Path(args.data_root)
    detections_root = Path(args.detections_root)
    velodyne_dir = data_root / "velodyne" / sequence
    calib = Calibration(data_root / "calib" / f"{sequence}.txt")
    frame_paths = sorted(velodyne_dir.glob("*.bin"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    tracks: list[Track] = []
    next_track_id = 0
    result_lines: list[str] = []
    diagnostics = []
    started = time.monotonic()
    num_matches = 0
    num_low_score_matches = 0
    num_low_score_gate_rejects = 0
    num_low_score_utonia_rejects = 0
    num_low_score_utonia_validations = 0
    num_utonia_fallbacks = 0
    num_new_tracks = 0

    for frame_index, frame_path in enumerate(frame_paths, start=1):
        frame_id = int(frame_path.stem)
        detections, low_detections = split_high_low_detections(
            data_root=data_root,
            detections_root=detections_root,
            sequence=sequence,
            frame_id=frame_id,
            calib=calib,
            classes=set(args.classes),
            high_score_thresh=args.score_thresh,
            low_score_thresh=args.low_score_thresh if args.use_low_score_continuation else None,
            car_low_score_thresh=args.car_low_score_thresh if args.use_low_score_continuation else None,
            pedestrian_low_score_thresh=args.pedestrian_low_score_thresh if args.use_low_score_continuation else None,
            nms_iou=args.nms_iou,
        )
        coord = None
        if args.appearance_mode == "utonia" and tracker_model is not None and (detections or low_detections):
            coord = load_xyz(frame_path)
        if args.appearance_mode == "utonia" and tracker_model is not None and detections:
            detection_embeddings(
                tracker_model,
                coord,
                detections,
                min_points=args.box_embedding_min_points,
            )
        if args.appearance_mode == "utonia" and tracker_model is not None and low_detections:
            detection_embeddings(
                tracker_model,
                coord,
                low_detections,
                min_points=args.box_embedding_min_points,
            )

        active_tracks = [track for track in tracks if track.lost_age <= args.max_age]
        cost, allowed = association_cost(active_tracks, detections, args)
        matched_track_indices = set()
        matched_det_indices = set()
        if len(active_tracks) and len(detections):
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols, strict=False):
                if not allowed[row, col]:
                    continue
                track = active_tracks[row]
                det = detections[col]
                update_track(track, det, args)
                track.last_detector_frame = frame_id
                matched_track_indices.add(row)
                matched_det_indices.add(col)
                num_matches += 1
                if track.hits >= args.min_hits:
                    result_lines.append(kitti_line(frame_id, track, det))
                diagnostics.append(
                    {
                        "sequence": sequence,
                        "frame": frame_id,
                        "event": "match",
                        "track_id": track.track_id,
                        "class": track.class_name,
                        "det_index": det.det_index,
                        "score": det.score,
                        "cost": float(cost[row, col]),
                        "lost_age": track.lost_age,
                        "support_count": None,
                        "mean_similarity": None,
                        "jump_xy": None,
                        "projected_bbox_width": None,
                        "projected_bbox_height": None,
                        "fallback_refined": None,
                        "fallback_refine_reason": "",
                        "refined_dx": None,
                        "refined_dy": None,
                        "refined_dz": None,
                        "fallback_reason": "",
                    }
                )

        low_matched_det_indices = set()
        if args.use_low_score_continuation and low_detections:
            active_track_to_row = {id(track): row for row, track in enumerate(active_tracks)}
            unmatched_tracks = [
                track
                for row, track in enumerate(active_tracks)
                if row not in matched_track_indices
            ]
            low_cost, low_allowed = association_cost(unmatched_tracks, low_detections, args)
            if len(unmatched_tracks) and len(low_detections):
                rows, cols = linear_sum_assignment(low_cost)
                for low_row, col in zip(rows, cols, strict=False):
                    if not low_allowed[low_row, col] or col in low_matched_det_indices:
                        continue
                    track = unmatched_tracks[low_row]
                    active_row = active_track_to_row[id(track)]
                    if active_row in matched_track_indices:
                        continue
                    det = low_detections[col]
                    low_distance, low_iou = pair_distance_iou(track, det)
                    gate_reason = low_score_gate_reason(
                        track=track,
                        det=det,
                        cost=float(low_cost[low_row, col]),
                        distance=low_distance,
                        iou=low_iou,
                        args=args,
                    )
                    if gate_reason:
                        num_low_score_gate_rejects += 1
                        diagnostics.append(
                            {
                                "sequence": sequence,
                                "frame": frame_id,
                                "event": "low_score_rejected_gate",
                                "track_id": track.track_id,
                                "class": track.class_name,
                                "det_index": det.det_index,
                                "score": det.score,
                                "cost": float(low_cost[low_row, col]),
                                "lost_age": track.lost_age,
                                "low_score_distance": low_distance,
                                "low_score_iou": low_iou,
                                "low_score_reject_reason": gate_reason,
                            }
                        )
                        continue
                    validation_meta = {}
                    if should_validate_low_score(track, det, args):
                        num_low_score_utonia_validations += 1
                        if coord is None:
                            coord = load_xyz(frame_path)
                        accepted, validation_meta = validate_low_score_with_utonia(
                            track=track,
                            det=det,
                            coord=coord,
                            shared_model=tracker_model,
                            args=args,
                        )
                        if not accepted:
                            num_low_score_utonia_rejects += 1
                            diagnostics.append(
                                {
                                    "sequence": sequence,
                                    "frame": frame_id,
                                    "event": "low_score_rejected_utonia",
                                    "track_id": track.track_id,
                                    "class": track.class_name,
                                    "det_index": det.det_index,
                                    "score": det.score,
                                    "cost": float(low_cost[low_row, col]),
                                    "lost_age": track.lost_age,
                                    "low_score_distance": low_distance,
                                    "low_score_iou": low_iou,
                                    **validation_meta,
                                }
                            )
                            continue
                    update_track(track, det, args)
                    track.last_detector_frame = frame_id
                    matched_track_indices.add(active_row)
                    low_matched_det_indices.add(col)
                    num_low_score_matches += 1
                    if track.hits >= args.min_hits:
                        result_lines.append(kitti_line(frame_id, track, det))
                    diagnostics.append(
                        {
                            "sequence": sequence,
                            "frame": frame_id,
                            "event": "low_score_match",
                            "track_id": track.track_id,
                            "class": track.class_name,
                            "det_index": det.det_index,
                            "score": det.score,
                            "cost": float(low_cost[low_row, col]),
                            "lost_age": track.lost_age,
                            "low_score_distance": low_distance,
                            "low_score_iou": low_iou,
                            **validation_meta,
                            "support_count": None,
                            "mean_similarity": None,
                            "jump_xy": None,
                            "projected_bbox_width": None,
                            "projected_bbox_height": None,
                            "fallback_refined": None,
                            "fallback_refine_reason": "",
                            "refined_dx": None,
                            "refined_dy": None,
                            "refined_dz": None,
                            "fallback_reason": "",
                        }
                    )

        active_id_set = {id(track): row for row, track in enumerate(active_tracks)}
        fallback_attempts = 0
        for track in tracks:
            row = active_id_set.get(id(track))
            if row is not None and row in matched_track_indices:
                continue
            can_try_fallback = (
                args.utonia_fallback
                and row is not None
                and tracker_model is not None
                and should_try_utonia_fallback(track, frame_id, args)
                and fallback_attempts < args.fallback_max_candidates_per_frame
            )
            if can_try_fallback:
                fallback_attempts += 1
                if coord is None:
                    coord = load_xyz(frame_path)
                accepted, line, fallback_meta = try_utonia_fallback(
                    track=track,
                    frame_id=frame_id,
                    coord=coord,
                    data_root=data_root,
                    sequence=sequence,
                    calib=calib,
                    shared_model=tracker_model,
                    args=args,
                )
                if accepted:
                    num_utonia_fallbacks += 1
                    if track.hits >= args.min_hits and line is not None:
                        result_lines.append(line)
                    diagnostics.append(
                        {
                            "sequence": sequence,
                            "frame": frame_id,
                            "event": "utonia_fallback",
                            "track_id": track.track_id,
                            "class": track.class_name,
                            "det_index": None,
                            "score": args.fallback_score,
                            "cost": None,
                            "lost_age": track.lost_age,
                            "support_count": fallback_meta.get("support_count"),
                            "mean_similarity": fallback_meta.get("mean_similarity"),
                            "jump_xy": fallback_meta.get("jump_xy"),
                            "projected_bbox_width": fallback_meta.get("projected_bbox_width"),
                            "projected_bbox_height": fallback_meta.get("projected_bbox_height"),
                            "fallback_refined": fallback_meta.get("fallback_refined"),
                            "fallback_refine_reason": fallback_meta.get("fallback_refine_reason", ""),
                            "refined_dx": fallback_meta.get("refined_dx"),
                            "refined_dy": fallback_meta.get("refined_dy"),
                            "refined_dz": fallback_meta.get("refined_dz"),
                            "fallback_reason": "",
                        }
                    )
                    continue
                diagnostics.append(
                    {
                        "sequence": sequence,
                        "frame": frame_id,
                        "event": "utonia_rejected",
                        "track_id": track.track_id,
                        "class": track.class_name,
                        "det_index": None,
                        "score": None,
                        "cost": None,
                        "lost_age": track.lost_age,
                        "support_count": fallback_meta.get("support_count"),
                        "mean_similarity": fallback_meta.get("mean_similarity"),
                        "jump_xy": fallback_meta.get("jump_xy"),
                        "projected_bbox_width": fallback_meta.get("projected_bbox_width"),
                        "projected_bbox_height": fallback_meta.get("projected_bbox_height"),
                        "fallback_refined": fallback_meta.get("fallback_refined"),
                        "fallback_refine_reason": fallback_meta.get("fallback_refine_reason", ""),
                        "refined_dx": fallback_meta.get("refined_dx"),
                        "refined_dy": fallback_meta.get("refined_dy"),
                        "refined_dz": fallback_meta.get("refined_dz"),
                        "fallback_reason": fallback_meta.get("fallback_reason", ""),
                    }
                )
            track.lost_age += 1
            track.age += 1

        for det_index, det in enumerate(detections):
            if det_index in matched_det_indices:
                continue
            track = Track(
                track_id=next_track_id,
                class_name=det.class_name,
                box_lidar=det.box_lidar.copy(),
                score=det.score,
                embedding=det.embedding.copy() if det.embedding is not None else None,
                velocity=np.zeros(3, dtype=np.float32),
                last_detector_frame=frame_id,
            )
            next_track_id += 1
            tracks.append(track)
            num_new_tracks += 1
            if track.hits >= args.min_hits:
                result_lines.append(kitti_line(frame_id, track, det))
            diagnostics.append(
                {
                    "sequence": sequence,
                    "frame": frame_id,
                    "event": "new",
                    "track_id": track.track_id,
                    "class": track.class_name,
                    "det_index": det.det_index,
                    "score": det.score,
                    "cost": None,
                    "lost_age": track.lost_age,
                    "support_count": None,
                    "mean_similarity": None,
                    "jump_xy": None,
                    "projected_bbox_width": None,
                    "projected_bbox_height": None,
                    "fallback_refined": None,
                    "fallback_refine_reason": "",
                    "refined_dx": None,
                    "refined_dy": None,
                    "refined_dz": None,
                    "fallback_reason": "",
                }
            )

        tracks = [track for track in tracks if track.lost_age <= args.max_age]
        if frame_index == 1 or frame_index % 50 == 0 or frame_index == len(frame_paths):
            elapsed = time.monotonic() - started
            fps = frame_index / max(elapsed, 1e-6)
            remaining = (len(frame_paths) - frame_index) / max(fps, 1e-6)
            print(
                f"{sequence} [{frame_index}/{len(frame_paths)}] "
                f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)} "
                f"dets={len(detections)} low={len(low_detections)} active={len(tracks)}",
                flush=True,
            )

    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / f"{sequence}.txt").write_text("\n".join(result_lines) + ("\n" if result_lines else ""))
    return {
        "sequence": sequence,
        "frames": len(frame_paths),
        "result_rows": len(result_lines),
        "tracks_created": next_track_id,
        "matches": num_matches,
        "low_score_matches": num_low_score_matches,
        "low_score_gate_rejects": num_low_score_gate_rejects,
        "low_score_utonia_validations": num_low_score_utonia_validations,
        "low_score_utonia_rejects": num_low_score_utonia_rejects,
        "utonia_fallbacks": num_utonia_fallbacks,
        "new_tracks": num_new_tracks,
        "diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_model = None
    if args.appearance_mode == "utonia" or args.utonia_fallback or args.validate_low_score_utonia:
        tracker_model = UtoniaTracker(
            mode="local_crop",
            local_crop_radius=args.local_crop_radius,
            local_crop_min_points=args.local_crop_min_points,
        )
        tracker_model.build_model()

    summaries = []
    diagnostics = []
    for sequence in args.sequences:
        summary = run_sequence(args, sequence, output_dir, tracker_model)
        diagnostics.extend(summary.pop("diagnostics"))
        summaries.append(summary)

    with (output_dir / "diagnostics.csv").open("w", newline="") as handle:
        fieldnames = [
            "sequence",
            "frame",
            "event",
            "track_id",
            "class",
            "det_index",
            "score",
            "cost",
            "lost_age",
            "low_score_distance",
            "low_score_iou",
            "low_score_reject_reason",
            "validation_similarity",
            "validation_pred_support",
            "validation_det_support",
            "support_count",
            "mean_similarity",
            "jump_xy",
            "projected_bbox_width",
            "projected_bbox_height",
            "fallback_refined",
            "fallback_refine_reason",
            "refined_dx",
            "refined_dy",
            "refined_dz",
            "fallback_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    summary = {
        "config": vars(args),
        "sequences": summaries,
        "total_frames": sum(row["frames"] for row in summaries),
        "total_result_rows": sum(row["result_rows"] for row in summaries),
        "total_tracks_created": sum(row["tracks_created"] for row in summaries),
        "total_matches": sum(row["matches"] for row in summaries),
        "total_low_score_matches": sum(row["low_score_matches"] for row in summaries),
        "total_low_score_gate_rejects": sum(row["low_score_gate_rejects"] for row in summaries),
        "total_low_score_utonia_validations": sum(row["low_score_utonia_validations"] for row in summaries),
        "total_low_score_utonia_rejects": sum(row["low_score_utonia_rejects"] for row in summaries),
        "total_utonia_fallbacks": sum(row["utonia_fallbacks"] for row in summaries),
        "total_new_tracks": sum(row["new_tracks"] for row in summaries),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved KITTI MOT results to {output_dir / 'data'}", flush=True)


if __name__ == "__main__":
    main()
