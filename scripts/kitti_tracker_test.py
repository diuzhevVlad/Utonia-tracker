import argparse
from pathlib import Path

from matplotlib import cm
import numpy as np
import rerun as rr

from utonia import UtoniaTracker


def load_xyz(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def parse_labels(path: Path):
    records = []
    for line in path.read_text().splitlines():
        parts = line.split()
        records.append(
            {
                "frame": int(parts[0]),
                "track_id": int(parts[1]),
                "type": parts[2],
                "h": float(parts[10]),
                "w": float(parts[11]),
                "l": float(parts[12]),
                "x": float(parts[13]),
                "y": float(parts[14]),
                "z": float(parts[15]),
            }
        )
    return records


def choose_track_id(records):
    frame0 = [r for r in records if r["frame"] == 0]
    for cls in ("Car", "Van"):
        for record in frame0:
            if record["type"] == cls:
                return record["track_id"]


def camera_to_velodyne(record):
    # Minimal KITTI camera->LiDAR axis swap; exact alignment needs sequence calibration.
    return np.array([record["z"], -record["x"], -record["y"]], dtype=np.float32)


def plasma(sim):
    sim = ((sim + 1.0) * 0.5).clip(0.0, 1.0)
    return (cm.plasma(sim)[:, :3] * 255).astype(np.uint8)


def gt_by_frame(records, track_id):
    return {r["frame"]: r for r in records if r["track_id"] == track_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    args = parser.parse_args()

    velodyne_dir = Path(args.sequence_dir)
    seq = velodyne_dir.name
    label_path = velodyne_dir.parents[1] / "label_02" / f"{seq}.txt"

    records = parse_labels(label_path)
    track_id = choose_track_id(records)
    gt = gt_by_frame(records, track_id)
    frame_ids = sorted(gt.keys())

    tracker = UtoniaTracker(init_radius=1.6, cluster_radius=1.6, gate_radius=5.0)
    first_frame = frame_ids[0]
    first_coord = load_xyz(velodyne_dir / f"{first_frame:06d}.bin")
    init_state = tracker.initialize(first_coord, camera_to_velodyne(gt[first_frame]))

    rr.init("utonia_kitti_tracker", spawn=True)
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
        rr.Points3D(camera_to_velodyne(gt[first_frame])[None], colors=np.array([[0, 255, 255]], dtype=np.uint8)),
    )

    for frame_id in frame_ids[1:]:
        coord = load_xyz(velodyne_dir / f"{frame_id:06d}.bin")
        state = tracker.step(coord)
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
        rr.log(
            "track/gt",
            rr.Points3D(camera_to_velodyne(gt[frame_id])[None], colors=np.array([[0, 255, 255]], dtype=np.uint8)),
        )


if __name__ == "__main__":
    main()
