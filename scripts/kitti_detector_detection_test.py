import argparse
from pathlib import Path
import sys

import rerun as rr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utonia.tracking.adapters import KittiPrecomputedDetectionSource
from utonia.tracking.visualization import detection_boxes, load_xyz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="KITTI sequence dir, e.g. /path/to/trackkitti/training/velodyne/0000",
    )
    parser.add_argument(
        "--detections-root",
        default="data/detections/openpcdet_pointpillar",
        help="Root of precomputed detections",
    )
    parser.add_argument("--frame", type=int, help="Optional frame id to inspect")
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Minimum detection score to visualize",
    )
    args = parser.parse_args()

    source = KittiPrecomputedDetectionSource(
        args.sequence_dir,
        args.detections_root,
        score_threshold=args.score_thresh,
    )
    frame_ids = [args.frame] if args.frame is not None else source.frame_ids()

    rr.init("utonia_kitti_detector_detections", spawn=True)
    for frame_id in frame_ids:
        point_path = Path(args.sequence_dir) / f"{frame_id:06d}.bin"
        coord = load_xyz(point_path)
        frame = source.get_frame_detections(frame_id)
        boxes = detection_boxes(frame)
        print(f"frame {frame_id}: {len(frame.detections)} detections")
        rr.set_time("frame", sequence=frame_id)
        rr.log("points", rr.Points3D(coord))
        if boxes is not None:
            rr.log("detections/boxes", boxes)


if __name__ == "__main__":
    main()
