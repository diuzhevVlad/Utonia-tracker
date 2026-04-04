import argparse
from pathlib import Path
import sys

from matplotlib import cm
import numpy as np
import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.adapters import KittiGtDetectionSource
from utonia.tracking.visualization import load_xyz
from utonia import UtoniaTracker
def collect_gt_tracks(source: KittiGtDetectionSource):
    tracks = {}
    for frame_id in source.frame_ids():
        frame = source.get_frame_detections(frame_id)
        for detection in frame.detections:
            track_id = detection.metadata.get("track_id")
            if track_id is None:
                continue
            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][frame_id] = detection
    return tracks


def choose_track_id(tracks, preferred_labels=None):
    if not tracks:
        raise ValueError("No valid KITTI tracks found in the detection source")

    if preferred_labels is None:
        preferred_labels = ("Van", "Cyclist", "Pedestrian", "Car")

    for cls in preferred_labels:
        class_tracks = {
            track_id: frame_map
            for track_id, frame_map in tracks.items()
            if next(iter(frame_map.values())).label == cls
        }
        if not class_tracks:
            continue

        best_track_id, _ = min(
            class_tracks.items(),
            key=lambda item: (min(item[1].keys()), -len(item[1]), item[0]),
        )
        return best_track_id

    raise ValueError("No supported KITTI track types found")


def plasma(sim):
    sim = ((sim + 1.0) * 0.5).clip(0.0, 1.0)
    return (cm.plasma(sim)[:, :3] * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument("--track-id", type=int, help="Optional KITTI track id override")
    parser.add_argument(
        "--label",
        choices=["Van", "Cyclist", "Pedestrian", "Car"],
        help="Prefer a specific KITTI class when auto-selecting a target",
    )
    args = parser.parse_args()

    velodyne_dir = Path(args.sequence_dir)
    source = KittiGtDetectionSource(velodyne_dir)
    tracks = collect_gt_tracks(source)
    preferred_labels = (args.label,) if args.label is not None else None
    track_id = (
        args.track_id
        if args.track_id is not None
        else choose_track_id(tracks, preferred_labels=preferred_labels)
    )
    gt = tracks.get(track_id, {})
    frame_ids = sorted(gt.keys())
    if not frame_ids:
        raise ValueError(f"Track id {track_id} has no frames in sequence {velodyne_dir.name}")

    first_frame = frame_ids[0]
    sequence_frame_ids = [frame_id for frame_id in source.frame_ids() if frame_id >= first_frame]
    print(
        "tracking KITTI "
        f"track_id={track_id}, class={gt[first_frame].label}, "
        f"init_frame={first_frame}, gt_frames={len(frame_ids)}, "
        f"run_frames={len(sequence_frame_ids)}"
    )

    tracker = UtoniaTracker(init_radius=0.4, cluster_radius=0.4, gate_radius=2.0)
    first_coord = load_xyz(velodyne_dir / f"{first_frame:06d}.bin")
    init_state = tracker.initialize(first_coord, gt[first_frame].box.center)

    rr.init("utonia_kitti_tracker", spawn=True)
    rr.set_time("frame", sequence=first_frame)
    rr.log(
        "points",
        rr.Points3D(init_state["coord"], colors=plasma(init_state["sim"])),
    )
    rr.log(
        "track/object",
        rr.Points3D(
            init_state["coord"][init_state["mask"]],
            colors=np.tile([[0, 255, 0]], (int(init_state["mask"].sum()), 1)),
        ),
    )
    rr.log(
        "track/pred",
        rr.Points3D(init_state["centroid"][None], colors=np.array([[255, 0, 0]], dtype=np.uint8)),
    )
    rr.log(
        "track/gt",
        rr.Points3D(
            gt[first_frame].box.center[None],
            colors=np.array([[0, 255, 255]], dtype=np.uint8),
        ),
    )

    for frame_id in sequence_frame_ids[1:]:
        coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
        state = tracker.step(coord)
        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(state["coord"], colors=plasma(state["sim"])))
        rr.log(
            "track/object",
            rr.Points3D(
                state["coord"][state["mask"]],
                colors=np.tile([[0, 255, 0]], (int(state["mask"].sum()), 1)),
            ),
        )
        rr.log(
            "track/pred",
            rr.Points3D(state["centroid"][None], colors=np.array([[255, 0, 0]], dtype=np.uint8)),
        )
        if frame_id in gt:
            rr.log(
                "track/gt",
                rr.Points3D(
                    gt[frame_id].box.center[None],
                    colors=np.array([[0, 255, 255]], dtype=np.uint8),
                ),
            )


if __name__ == "__main__":
    main()
