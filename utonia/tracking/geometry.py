from __future__ import annotations

import numpy as np

from .types import Box3D


def box_bev_corners(box: Box3D) -> np.ndarray:
    length, width = box.size[:2]
    half_length = length / 2.0
    half_width = width / 2.0
    corners = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float32,
    )
    cos_yaw = np.cos(box.yaw)
    sin_yaw = np.sin(box.yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
        dtype=np.float32,
    )
    return corners @ rotation.T + box.center[:2][None, :]


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    rel = point - edge_start
    return float(edge[0] * rel[1] - edge[1] * rel[0]) >= 0.0


def _segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(float(denom)) < 1e-6:
        return p2
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    return p1 + t * r


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject
    for index in range(len(clipper)):
        edge_start = clipper[index]
        edge_end = clipper[(index + 1) % len(clipper)]
        input_points = output
        if len(input_points) == 0:
            break
        output_points = []
        start = input_points[-1]
        for end in input_points:
            end_inside = _inside(end, edge_start, edge_end)
            start_inside = _inside(start, edge_start, edge_end)
            if end_inside:
                if not start_inside:
                    output_points.append(
                        _segment_intersection(start, end, edge_start, edge_end)
                    )
                output_points.append(end)
            elif start_inside:
                output_points.append(
                    _segment_intersection(start, end, edge_start, edge_end)
                )
            start = end
        output = np.asarray(output_points, dtype=np.float32)
    return output


def bev_iou(box_a: Box3D, box_b: Box3D) -> float:
    corners_a = box_bev_corners(box_a)
    corners_b = box_bev_corners(box_b)
    inter_poly = polygon_clip(corners_a, corners_b)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0
    area_a = polygon_area(corners_a)
    area_b = polygon_area(corners_b)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
