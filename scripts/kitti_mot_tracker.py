import argparse
from pathlib import Path
import sys

import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.adapters import (
    KittiGtDetectionSource,
    KittiPrecomputedDetectionSource,
)
from utonia.tracking.config import AssociationConfig, FeatureCropConfig, MotionModelConfig
from utonia.tracking.mot import MOTracker, UtoniaMOTracker
from utonia.tracking.visualization import detection_boxes, load_xyz, track_boxes


def format_profile(profile: dict[str, float | int]) -> str:
    parts = []
    for key, value in profile.items():
        if key.endswith("_s"):
            parts.append(f"{key}={value * 1000.0:.1f}ms")
        elif key in {"input_points", "roi_points", "roi_boxes"}:
            parts.append(f"{key}={int(value)}")
        elif key in {"roi_skip_appearance", "appearance_skipped"}:
            parts.append(f"{key}={bool(value)}")
    return ", ".join(parts)


def print_profile_summary(summary: dict[str, float]) -> None:
    if not summary:
        return
    frames = int(summary.get("frames", 0.0))
    stage_parts = []
    count_parts = []
    for key, value in summary.items():
        if key == "frames":
            continue
        if key.endswith("_s"):
            stage_parts.append(f"{key}={value * 1000.0:.1f}ms")
        elif key.startswith("num_") or key in {"input_points", "roi_points", "roi_boxes"}:
            count_parts.append(f"{key}={value:.2f}")
        elif key in {"roi_skip_appearance", "appearance_skipped"}:
            count_parts.append(f"{key}={value:.2f}")
    print(f"profile summary over {frames} frames")
    if stage_parts:
        print("  timings: " + ", ".join(stage_parts))
    if count_parts:
        print("  counts: " + ", ".join(count_parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--detections-root",
        help="Optional root of precomputed detections, e.g. data/detections/openpcdet_pointpillar",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Minimum detection score when using precomputed detections",
    )
    parser.add_argument("--max-match-distance", type=float, default=5.0)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--bev-iou-weight", type=float, default=1.0)
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument(
        "--motion-model",
        choices=["kalman", "velocity"],
        default="kalman",
        help="Motion model used for track prediction",
    )
    parser.add_argument("--process-var", type=float, default=1.0)
    parser.add_argument("--measurement-var", type=float, default=1.0)
    parser.add_argument(
        "--disable-class-thresholds",
        action="store_true",
        help="Use flat thresholds for all classes",
    )
    parser.add_argument(
        "--disable-bev-iou",
        action="store_true",
        help="Disable BEV IoU in association cost",
    )
    parser.add_argument(
        "--feature-crop-mode",
        choices=["full", "detections", "detections_and_tracks"],
        default="detections_and_tracks",
        help="Point-cloud region used for Utonia feature extraction",
    )
    parser.add_argument("--feature-crop-margin", type=float, default=2.0)
    parser.add_argument("--feature-crop-min-points", type=int, default=2048)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-frame and average runtime breakdown",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use the basic motion-only tracker instead of Utonia matching",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.detections_root is not None:
        source = KittiPrecomputedDetectionSource(
            args.sequence_dir,
            args.detections_root,
            score_threshold=args.score_thresh,
        )
    else:
        source = KittiGtDetectionSource(args.sequence_dir)
    basic_tracker = None
    utonia_tracker = None
    association_config = AssociationConfig(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        motion_weight=args.motion_weight,
        bev_iou_weight=args.bev_iou_weight if not args.disable_bev_iou else 0.0,
        appearance_weight=0.0,
    )
    motion_config = MotionModelConfig(
        kind=args.motion_model,
        process_var=args.process_var,
        measurement_var=args.measurement_var,
    )
    if args.basic:
        basic_tracker = MOTracker(
            max_match_distance=args.max_match_distance,
            max_missed=args.max_missed,
            use_class_thresholds=not args.disable_class_thresholds,
            enable_bev_iou=not args.disable_bev_iou,
            motion_model=motion_config,
            association_config=association_config,
        )
        tracker_name = "basic"
    else:
        utonia_tracker = UtoniaMOTracker(
            max_match_distance=args.max_match_distance,
            max_missed=args.max_missed,
            motion_weight=args.motion_weight,
            bev_iou_weight=args.bev_iou_weight,
            appearance_weight=args.appearance_weight,
            use_class_thresholds=not args.disable_class_thresholds,
            enable_bev_iou=not args.disable_bev_iou,
            motion_model=motion_config,
            feature_crop=FeatureCropConfig(
                mode=args.feature_crop_mode,
                crop_margin=args.feature_crop_margin,
                min_points=args.feature_crop_min_points,
            ),
        )
        tracker_name = "utonia"

    frame_ids = [frame_id for frame_id in source.frame_ids() if frame_id >= args.start_frame]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    rr.init("utonia_kitti_mot_demo", spawn=True)
    for frame_id in frame_ids:
        point_path = Path(args.sequence_dir) / f"{frame_id:06d}.bin"
        coord = load_xyz(point_path)
        frame = source.get_frame_detections(frame_id)
        if args.basic:
            assert basic_tracker is not None
            tracks = basic_tracker.update(frame)
        else:
            assert utonia_tracker is not None
            tracks = utonia_tracker.update(coord, frame)

        print(
            f"frame {frame_id}: {len(frame.detections)} detections, {len(tracks)} active tracks ({tracker_name})"
        )
        if args.profile:
            tracker_obj = basic_tracker if args.basic else utonia_tracker
            assert tracker_obj is not None
            print("  profile: " + format_profile(tracker_obj.last_profile))

        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(coord))

        det_boxes = detection_boxes(frame)
        if det_boxes is not None:
            rr.log("detections/boxes", det_boxes)

        active_boxes = track_boxes(tracks)
        if active_boxes is not None:
            rr.log("tracks/boxes", active_boxes)

    if args.profile:
        tracker_obj = basic_tracker if args.basic else utonia_tracker
        assert tracker_obj is not None
        print_profile_summary(tracker_obj.profile_summary())


if __name__ == "__main__":
    main()
