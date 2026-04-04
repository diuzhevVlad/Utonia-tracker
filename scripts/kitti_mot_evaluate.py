import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.adapters import (
    KittiGtDetectionSource,
    KittiPrecomputedDetectionSource,
)
from utonia.tracking.config import (
    AssociationConfig,
    FeatureCropConfig,
    MotionModelConfig,
    RecoveryConfig,
    SpawnConfig,
)
from utonia.tracking.mot import MOTracker, UtoniaMOTracker
from utonia.tracking.visualization import load_xyz


KITTI_EVAL_CLASSES = {"Car", "Pedestrian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_dir",
        help="KITTI tracking training dir, e.g. /path/to/trackkitti/training",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Optional sequence ids such as 0000 0001 0020",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use the basic motion-only tracker instead of Utonia matching",
    )
    parser.add_argument(
        "--detections-root",
        help="Optional root of precomputed detections, e.g. data/detections/openpcdet_pointpillar",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
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
        "--disable-birth-suppression",
        action="store_true",
        help="Disable suppression of new tracks overlapping existing confirmed ones",
    )
    parser.add_argument("--spawn-same-class-iou", type=float, default=0.1)
    parser.add_argument("--spawn-cross-class-iou", type=float, default=0.25)
    parser.add_argument("--spawn-max-center-distance", type=float, default=2.0)
    parser.add_argument("--spawn-max-track-missed", type=int, default=1)
    parser.add_argument(
        "--disable-recovery",
        action="store_true",
        help="Disable point-based recovery for unmatched confirmed tracks",
    )
    parser.add_argument("--recovery-max-missed", type=int, default=3)
    parser.add_argument("--recovery-gate-radius", type=float, default=3.0)
    parser.add_argument("--recovery-cluster-radius", type=float, default=1.2)
    parser.add_argument("--recovery-sim-threshold", type=float, default=0.35)
    parser.add_argument("--recovery-min-points", type=int, default=48)
    parser.add_argument("--recovery-min-mean-similarity", type=float, default=0.45)
    parser.add_argument("--recovery-max-center-distance", type=float, default=2.0)
    parser.add_argument("--recovery-max-extent-scale", type=float, default=1.5)
    parser.add_argument(
        "--tracker-name",
        help="Name for exported tracker results",
    )
    parser.add_argument(
        "--output-root",
        default="data/kitti_eval",
        help="Directory for exported results and evaluation workspace",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["car", "pedestrian"],
        choices=["car", "pedestrian"],
        help="Official KITTI classes to evaluate",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export KITTI result files without running TrackEval",
    )
    return parser.parse_args()


def discover_sequence_ids(training_dir: Path) -> list[str]:
    velodyne_root = training_dir / "velodyne"
    if not velodyne_root.is_dir():
        raise FileNotFoundError(f"KITTI velodyne directory not found: {velodyne_root}")
    return sorted(path.name for path in velodyne_root.iterdir() if path.is_dir())


def discover_available_frame_ids(sequence_dir: Path) -> list[int]:
    frame_ids = sorted(int(path.stem) for path in sequence_dir.glob("*.bin"))
    if not frame_ids:
        raise ValueError(f"No point cloud frames found in {sequence_dir}")
    return frame_ids


def build_tracker(args) -> tuple[object, str]:
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
    recovery_config = RecoveryConfig(
        enabled=not args.disable_recovery,
        max_missed=args.recovery_max_missed,
        gate_radius=args.recovery_gate_radius,
        cluster_radius=args.recovery_cluster_radius,
        sim_threshold=args.recovery_sim_threshold,
        min_points=args.recovery_min_points,
        min_mean_similarity=args.recovery_min_mean_similarity,
        max_center_distance=args.recovery_max_center_distance,
        max_extent_scale=args.recovery_max_extent_scale,
    )
    spawn_config = SpawnConfig(
        enabled=not args.disable_birth_suppression,
        same_class_min_bev_iou=args.spawn_same_class_iou,
        cross_class_min_bev_iou=args.spawn_cross_class_iou,
        max_center_distance=args.spawn_max_center_distance,
        max_track_missed=args.spawn_max_track_missed,
    )
    if args.basic:
        return (
            MOTracker(
                max_match_distance=args.max_match_distance,
                max_missed=args.max_missed,
                use_class_thresholds=not args.disable_class_thresholds,
                enable_bev_iou=not args.disable_bev_iou,
                motion_model=motion_config,
                association_config=association_config,
                spawn=spawn_config,
            ),
            args.tracker_name or "utonia_basic",
        )
    return (
        UtoniaMOTracker(
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
            recovery=recovery_config,
            spawn=spawn_config,
        ),
        args.tracker_name or "utonia_appearance",
    )


def build_detection_source(args, sequence_dir: Path):
    if args.detections_root is not None:
        return KittiPrecomputedDetectionSource(
            sequence_dir,
            args.detections_root,
            score_threshold=args.score_thresh,
        )
    return KittiGtDetectionSource(sequence_dir)


def track_sequence(sequence_dir: Path, tracker, use_basic: bool, args) -> tuple[dict[int, list[object]], object]:
    source = build_detection_source(args, sequence_dir)
    tracker.reset()
    results: dict[int, list[object]] = {}
    for frame_id in discover_available_frame_ids(sequence_dir):
        frame = source.get_frame_detections(frame_id)
        if use_basic:
            tracks = tracker.update(frame)
        else:
            coord = load_xyz(sequence_dir / f"{frame_id:06d}.bin")
            tracks = tracker.update(coord, frame)
        results[frame_id] = copy.deepcopy(tracks)
    return results, source


def format_kitti_result_line(frame_id: int, track, calibration=None) -> str | None:
    if track.missed != 0:
        return None
    if track.label not in KITTI_EVAL_CLASSES:
        return None
    bbox_2d = track.metadata.get("bbox_2d")
    if bbox_2d is None and calibration is not None:
        bbox_2d = calibration.box_to_image_bbox(track.box)
    if bbox_2d is None:
        return None
    left, top, right, bottom = [float(value) for value in bbox_2d]
    return (
        f"{frame_id} {track.track_id} {track.label} -1 -1 -10 "
        f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
        f"-1 -1 -1 -1000 -1000 -1000 -10 {float(track.score):.6f}"
    )


def export_sequence_results(
    sequence_id: str,
    tracks_by_frame: dict[int, list[object]],
    out_dir: Path,
    calibration=None,
) -> None:
    lines: list[str] = []
    for frame_id in sorted(tracks_by_frame):
        for track in tracks_by_frame[frame_id]:
            line = format_kitti_result_line(frame_id, track, calibration=calibration)
            if line is not None:
                lines.append(line)
    out_path = out_dir / f"{sequence_id}.txt"
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def ensure_trackeval_available(trackeval_root: Path) -> None:
    runner = trackeval_root / "scripts" / "run_kitti.py"
    if not runner.is_file():
        raise FileNotFoundError(
            f"Official TrackEval runner not found: {runner}. Clone TrackEval into third_party/TrackEval."
        )


def ensure_gt_workspace(training_dir: Path, workspace_root: Path, sequence_ids: list[str]) -> Path:
    gt_root = workspace_root / "gt" / "kitti_2d_box_train"
    gt_root.mkdir(parents=True, exist_ok=True)

    label_target = training_dir / "label_02"
    if not label_target.is_dir():
        raise FileNotFoundError(f"KITTI label directory not found: {label_target}")

    label_link = gt_root / "label_02"
    if label_link.exists() or label_link.is_symlink():
        if label_link.is_symlink() and label_link.resolve() == label_target.resolve():
            pass
        else:
            if label_link.is_dir() and not label_link.is_symlink():
                shutil.rmtree(label_link)
            else:
                label_link.unlink()
            label_link.symlink_to(label_target, target_is_directory=True)
    else:
        label_link.symlink_to(label_target, target_is_directory=True)

    seqmap_path = gt_root / "evaluate_tracking.seqmap.training"
    velodyne_root = training_dir / "velodyne"
    lines = []
    for sequence_id in sequence_ids:
        sequence_dir = velodyne_root / sequence_id
        frame_count = len(list(sequence_dir.glob("*.bin")))
        lines.append(f"{sequence_id} empty 0 {frame_count}")
    seqmap_path.write_text("\n".join(lines) + "\n")
    return gt_root


def run_trackeval(
    trackeval_root: Path,
    gt_root: Path,
    trackers_root: Path,
    tracker_name: str,
    classes_to_eval: list[str],
) -> None:
    command = [
        sys.executable,
        str(trackeval_root / "scripts" / "run_kitti.py"),
        "--GT_FOLDER",
        str(gt_root),
        "--TRACKERS_FOLDER",
        str(trackers_root),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--CLASSES_TO_EVAL",
        *classes_to_eval,
        "--SPLIT_TO_EVAL",
        "training",
        "--USE_PARALLEL",
        "False",
        "--PRINT_CONFIG",
        "False",
        "--PLOT_CURVES",
        "False",
    ]
    subprocess.run(command, check=True, cwd=trackeval_root)


def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    if not training_dir.is_dir():
        raise FileNotFoundError(f"KITTI training directory not found: {training_dir}")

    sequence_ids = args.sequences or discover_sequence_ids(training_dir)
    tracker, tracker_name = build_tracker(args)

    output_root = Path(args.output_root).resolve()
    tracker_data_dir = output_root / "trackers" / "kitti" / "kitti_2d_box_train" / tracker_name / "data"
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    velodyne_root = training_dir / "velodyne"
    for sequence_id in sequence_ids:
        sequence_dir = velodyne_root / sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {sequence_dir}")
        print(f"tracking sequence {sequence_id} with tracker={tracker_name}")
        tracks_by_frame, source = track_sequence(sequence_dir, tracker, args.basic, args)
        export_sequence_results(
            sequence_id,
            tracks_by_frame,
            tracker_data_dir,
            calibration=getattr(source, "calibration", None),
        )

    print(f"exported KITTI results to {tracker_data_dir}")

    if args.export_only:
        return

    trackeval_root = ROOT / "third_party" / "TrackEval"
    ensure_trackeval_available(trackeval_root)
    gt_root = ensure_gt_workspace(training_dir, output_root, sequence_ids)
    run_trackeval(
        trackeval_root=trackeval_root,
        gt_root=gt_root,
        trackers_root=tracker_data_dir.parents[1],
        tracker_name=tracker_name,
        classes_to_eval=args.classes,
    )


if __name__ == "__main__":
    main()
