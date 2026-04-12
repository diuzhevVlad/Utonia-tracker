import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = REPO_ROOT / "third_party" / "OpenPCDet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from scripts.kitti_tracker_test import (  # noqa: E402
    TRACKER_PRESETS,
    canonical_label,
    load_xyz,
    match_update_box,
    parse_labels,
    record_to_lidar_box,
    resolve_tracker_param,
    select_init_box,
)
from utonia import UtoniaTracker  # noqa: E402


STATUS_COLORS = {
    "active": "red",
    "lost": "orange",
    "not_started": "gray",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", help="KITTI tracking sequence id, e.g. 0000")
    parser.add_argument(
        "--data-root",
        default="/media/vladislav/KINGSTON/trackkitti/training",
        help="KITTI tracking training root with velodyne/, calib/, and label_02/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of longest tracklets to visualize.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Car", "Cyclist", "Pedestrian"],
        choices=sorted(TRACKER_PRESETS.keys()),
        help="Track classes eligible for the top-k selection.",
    )
    parser.add_argument(
        "--init-source",
        choices=["gt", "pointpillar", "pointrcnn"],
        default="pointpillar",
        help="Initialization source for all tracks in the video.",
    )
    parser.add_argument(
        "--update-source",
        choices=["none", "gt", "pointpillar", "pointrcnn"],
        default="pointpillar",
        help="Per-frame update source for all tracks in the video.",
    )
    parser.add_argument(
        "--detections-base-root",
        default=str(REPO_ROOT / "data" / "detections"),
        help="Base directory containing <model>/npz/<sequence>/<frame>.npz.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--local-crop-radius", type=float, default=8.0)
    parser.add_argument("--local-crop-min-points", type=int, default=2048)
    parser.add_argument("--detection-overlap-thresh", type=float, default=0.3)
    parser.add_argument("--gate-radius", type=float, default=None)
    parser.add_argument("--cluster-radius", type=float, default=None)
    parser.add_argument("--init-points", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--box-height-filter-ratio", type=float, default=None)
    parser.add_argument("--lost-height-ratio", type=float, default=0.5)
    parser.add_argument("--lost-bad-frames", type=int, default=2)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit from the earliest selected track start.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--view-radius",
        type=float,
        default=15.0,
        help="BEV half-width around each track center in meters.",
    )
    parser.add_argument(
        "--max-bg-points",
        type=int,
        default=12000,
        help="Maximum number of gray background points rendered per frame.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Defaults to data/videos_sot/<sequence>_topk.mp4.",
    )
    return parser.parse_args()


def resolve_detection_root(base_root: Path, source: str) -> Path | None:
    if source in {"none", "gt"}:
        return None
    return base_root / source / "npz"


def build_tracklets(label_path: Path, classes: set[str]):
    tracks = defaultdict(list)
    for record in parse_labels(label_path):
        if record["track_id"] < 0:
            continue
        cls = canonical_label(record["type"])
        if cls not in classes:
            continue
        tracks[record["track_id"]].append(record)
    return [sorted(records, key=lambda record: record["frame"]) for records in tracks.values()]


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def box_bev_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, dx, dy, _, heading = box
    half_dx = dx * 0.5
    half_dy = dy * 0.5
    corners = np.array(
        [
            [half_dx, half_dy],
            [half_dx, -half_dy],
            [-half_dx, -half_dy],
            [-half_dx, half_dy],
            [half_dx, half_dy],
        ],
        dtype=np.float32,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def tracker_params_for_class(args, target_class: str) -> dict:
    return {
        "gate_radius": resolve_tracker_param(args.gate_radius, target_class, "gate_radius"),
        "cluster_radius": resolve_tracker_param(args.cluster_radius, target_class, "cluster_radius"),
        "init_points": resolve_tracker_param(args.init_points, target_class, "init_points"),
        "min_points": resolve_tracker_param(args.min_points, target_class, "min_points"),
        "box_height_filter_ratio": resolve_tracker_param(
            args.box_height_filter_ratio,
            target_class,
            "box_height_filter_ratio",
        ),
    }


def make_video(frame_dir: Path, fps: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    sequence = args.sequence
    velodyne_dir = data_root / "velodyne" / sequence
    label_path = data_root / "label_02" / f"{sequence}.txt"
    calib = Calibration(data_root / "calib" / f"{sequence}.txt")
    classes = {canonical_label(name) for name in args.classes}

    tracklets = build_tracklets(label_path, classes)
    tracklets = sorted(tracklets, key=len, reverse=True)[: args.top_k]
    if not tracklets:
        raise RuntimeError(f"No eligible tracklets found in sequence {sequence}")

    init_det_root = resolve_detection_root(Path(args.detections_base_root), args.init_source)
    update_det_root = resolve_detection_root(Path(args.detections_base_root), args.update_source)

    shared_model_tracker = UtoniaTracker(mode="local_crop")
    shared_model_tracker.build_model()
    shared_model = shared_model_tracker.model

    track_views = []
    earliest_frame = min(records[0]["frame"] for records in tracklets)
    last_frame = max(records[-1]["frame"] for records in tracklets)
    if args.max_frames is not None:
        last_frame = min(last_frame, earliest_frame + args.max_frames - 1)

    for records in tracklets:
        first_record = records[0]
        target_class = canonical_label(first_record["type"])
        params = tracker_params_for_class(args, target_class)
        tracker = UtoniaTracker(
            model=shared_model,
            mode="local_crop",
            init_radius=1.6,
            cluster_radius=params["cluster_radius"],
            gate_radius=params["gate_radius"],
            local_crop_radius=args.local_crop_radius,
            local_crop_min_points=args.local_crop_min_points,
            init_points=params["init_points"],
            min_points=params["min_points"],
            detection_overlap_threshold=args.detection_overlap_thresh,
            box_height_filter_ratio=params["box_height_filter_ratio"],
            lost_height_ratio=args.lost_height_ratio,
            lost_bad_frames=args.lost_bad_frames,
        )
        gt = gt_by_frame(records, first_record["track_id"])
        track_views.append(
            {
                "track_id": first_record["track_id"],
                "target_class": target_class,
                "records": records,
                "gt": gt,
                "tracker": tracker,
                "state": None,
                "started": False,
            }
        )

    output_path = Path(args.output) if args.output else REPO_ROOT / "data" / "videos_sot" / f"{sequence}_top{len(track_views)}.mp4"

    with tempfile.TemporaryDirectory(prefix="kitti_sot_video_") as tmpdir:
        frame_dir = Path(tmpdir)
        frame_index = 0
        for frame_id in range(earliest_frame, last_frame + 1):
            coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
            bg_points = sample_points(coord[:, :2], args.max_bg_points)
            fig, axes = plt.subplots(1, len(track_views), figsize=(6 * len(track_views), 6), squeeze=False)
            axes = axes[0]
            for ax, view in zip(axes, track_views, strict=False):
                gt_record = view["gt"].get(frame_id)
                tracker = view["tracker"]
                if not view["started"] and frame_id >= view["records"][0]["frame"]:
                    init_box, _ = select_init_box(
                        source=args.init_source,
                        first_frame=view["records"][0]["frame"],
                        gt_record=view["records"][0],
                        calib=calib,
                        detections_root=(init_det_root / sequence) if init_det_root is not None else Path("."),
                        score_thresh=args.score_thresh,
                    )
                    view["state"] = tracker.initialize_from_box(
                        load_xyz(velodyne_dir / f"{view['records'][0]['frame']:06d}.bin"),
                        init_box,
                    )
                    view["started"] = True
                elif view["started"] and frame_id > view["records"][0]["frame"]:
                    matched_box = None
                    if args.update_source != "none":
                        matched_box, _, _ = match_update_box(
                            source=args.update_source,
                            frame_id=frame_id,
                            gt_record=gt_record,
                            target_class=view["target_class"],
                            calib=calib,
                            detections_root=(update_det_root / sequence) if update_det_root is not None else Path("."),
                            score_thresh=args.score_thresh,
                            pred_centroid=tracker.predict_position().detach().cpu().numpy(),
                            match_radius=tracker.gate_radius,
                        )
                    view["state"] = tracker.step(coord, detection_box=matched_box)

                ax.scatter(bg_points[:, 0], bg_points[:, 1], s=1, c="lightgray", alpha=0.6)

                if view["started"]:
                    state = view["state"]
                    obj_points = state["coord"][state["mask"]] if len(state["coord"]) else np.zeros((0, 3), dtype=np.float32)
                    if len(obj_points):
                        ax.scatter(obj_points[:, 0], obj_points[:, 1], s=5, c="limegreen", alpha=0.9)
                    pred = state["centroid"]
                    ax.scatter([pred[0]], [pred[1]], s=40, c=STATUS_COLORS.get(state["status"], "red"), marker="x")
                    if state.get("used_box") is not None:
                        corners = box_bev_corners(state["used_box"])
                        ax.plot(corners[:, 0], corners[:, 1], c="gold", linewidth=2)
                    center = pred[:2]
                    status = state.get("status", "active")
                else:
                    init_center = record_to_lidar_box(view["records"][0], calib)[:2]
                    center = init_center
                    status = "not_started"

                if gt_record is not None:
                    gt_box = record_to_lidar_box(gt_record, calib)
                    corners = box_bev_corners(gt_box)
                    ax.plot(corners[:, 0], corners[:, 1], c="cyan", linewidth=2)

                ax.set_xlim(center[0] - args.view_radius, center[0] + args.view_radius)
                ax.set_ylim(center[1] - args.view_radius, center[1] + args.view_radius)
                ax.set_aspect("equal")
                ax.grid(True, linewidth=0.3, alpha=0.4)
                ax.set_title(
                    f"frame={frame_id} track={view['track_id']} {view['target_class']}\n"
                    f"status={status} init={args.init_source} update={args.update_source}"
                )
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")

            fig.tight_layout()
            fig.savefig(frame_dir / f"{frame_index:06d}.png", dpi=120)
            plt.close(fig)
            frame_index += 1

        make_video(frame_dir, args.fps, output_path)

    print(f"Saved video to {output_path}")


def gt_by_frame(records, track_id):
    return {record["frame"]: record for record in records if record["track_id"] == track_id}


if __name__ == "__main__":
    main()
