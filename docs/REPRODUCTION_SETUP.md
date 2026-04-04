# Reproduction Setup

This document records the environment, third-party code, local patches, and commands needed to reproduce the current tracking experiments.

## Recommendation On `third_party`

Do **not** commit full third-party repositories directly into the main repo history unless you specifically need a fully offline snapshot.

Recommended options:

1. Keep `third_party/` out of the main Git history and reconstruct it with the commands below.
2. If you want stronger reproducibility, use **Git submodules pinned to exact commits**.
3. If you need a frozen archive for a paper/thesis submission, keep the clone commands plus the local patch notes below.

For this repo, I recommend:

- keep the main repo clean
- pin exact third-party commits in documentation
- document the small local patches

That gives reproducibility without bloating your repository.

## Base Environment

Repository root:

```bash
/home/vladislav/Documents/Utonia-tracker
```

Recommended environment creation:

```bash
conda env create -f environment.yml --verbose
conda activate utonia
```

Current confirmed runtime versions:

- Python: `3.10`
- PyTorch: `2.5.0`
- CUDA runtime in torch: `12.4`
- `rerun-sdk`: `0.30.2`
- `spconv-cu124`: `2.3.8`

Extra Python packages that were required beyond `environment.yml`:

```bash
pip install easydict tensorboardX SharedArray gdown
```

Installed versions in the current environment:

- `easydict==1.13`
- `tensorboardX==2.6.4`
- `SharedArray==3.2.4`
- `gdown==5.2.1`
- `numba==0.65.0`
- `llvmlite==0.47.0`

## Third-Party Repositories

Create the folder if needed:

```bash
mkdir -p third_party
```

### TrackEval

Clone and pin:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
git -C third_party/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

### OpenPCDet

Clone and pin:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
git -C third_party/OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755
```

Install it into the `utonia` environment:

```bash
conda activate utonia
cd third_party/OpenPCDet
pip install -e . --no-build-isolation
```

Important: `--no-build-isolation` was required because normal editable install failed when the build environment could not see `torch`.

## Local Patches Applied To Third-Party Code

Two small local compatibility patches were required.

### 1. `TrackEval` NumPy compatibility patch

File:

```bash
third_party/TrackEval/trackeval/__init__.py
```

Add before the existing imports:

```python
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
```

Reason:

- current NumPy removed deprecated aliases like `np.float`
- this TrackEval revision still uses them

### 2. `OpenPCDet` optional dataset import patch

File:

```bash
third_party/OpenPCDet/pcdet/datasets/__init__.py
```

Instead of importing every dataset backend unconditionally, wrap non-KITTI dataset imports in `try/except ImportError` and set them to `None` on failure.

Example pattern used:

```python
try:
    from .argo2.argo2_dataset import Argo2Dataset
except ImportError:
    Argo2Dataset = None
```

Apply the same pattern for:

- `NuScenesDataset`
- `WaymoDataset`
- `PandasetDataset`
- `LyftDataset`
- `ONCEDataset`
- `Argo2Dataset`

Reason:

- importing `pcdet.datasets` eagerly pulled optional dataset stacks
- the first failure was missing `av2`
- KITTI inference should not require unrelated dataset packages

## Pretrained Detector Checkpoint

The simplest working detector baseline is KITTI `PointPillars` from OpenPCDet.

Download command:

```bash
mkdir -p third_party/OpenPCDet/checkpoints
gdown "https://drive.google.com/uc?id=1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm" -O third_party/OpenPCDet/checkpoints/pointpillar_kitti.pth
```

Current checkpoint path:

```bash
third_party/OpenPCDet/checkpoints/pointpillar_kitti.pth
```

## Detector Precomputation

Script:

```bash
scripts/openpcdet_precompute_detections.py
```

Run on all KITTI tracking training sequences:

```bash
conda activate utonia
python scripts/openpcdet_precompute_detections.py /media/vladislav/KINGSTON/trackkitti/training
```

Run on a subset:

```bash
python scripts/openpcdet_precompute_detections.py /media/vladislav/KINGSTON/trackkitti/training --sequences 0000 0001 0002
```

Default output:

```bash
data/detections/openpcdet_pointpillar/<sequence>/<frame>.npz
```

Metadata file per sequence:

```bash
data/detections/openpcdet_pointpillar/<sequence>/meta.npz
```

Saved frame fields:

- `boxes_lidar`
- `scores`
- `label_ids`
- `label_names`

Internal box convention:

```text
x, y, z, dx, dy, dz, heading
```

Coordinate frame:

```text
lidar
```

## Important Detector Limitation

The pretrained KITTI `PointPillars` detector is **front-view biased**.

Reasons:

- KITTI config uses `FOV_POINTS_ONLY: True`
- point cloud range is front-limited
- pretrained model classes are only:
  - `Car`
  - `Pedestrian`
  - `Cyclist`

So this detector is appropriate as a simple real-detector baseline, but it is **not** a full 360-degree detector.

## Visualization Commands

Ground-truth detections:

```bash
python scripts/kitti_gt_detection_test.py /media/vladislav/KINGSTON/trackkitti/training/velodyne/0000 --frame 0
```

Precomputed detector detections:

```bash
python scripts/kitti_detector_detection_test.py /media/vladislav/KINGSTON/trackkitti/training/velodyne/0000 --frame 0 --score-thresh 0.5
```

MOT demo with GT detections:

```bash
python scripts/kitti_mot_tracker.py /media/vladislav/KINGSTON/trackkitti/training/velodyne/0000 --basic
```

MOT demo with detector detections:

```bash
python scripts/kitti_mot_tracker.py /media/vladislav/KINGSTON/trackkitti/training/velodyne/0000 --basic --detections-root data/detections/openpcdet_pointpillar --score-thresh 0.5
```

MOT demo profiling:

```bash
python scripts/kitti_mot_tracker.py /media/vladislav/KINGSTON/trackkitti/training/velodyne/0000 --max-frames 10 --profile
```

## Evaluation Commands

Official KITTI tracking evaluation wrapper:

```bash
scripts/kitti_mot_evaluate.py
```

Basic tracker with GT detections:

```bash
python scripts/kitti_mot_evaluate.py /media/vladislav/KINGSTON/trackkitti/training --classes car pedestrian --basic
```

Utonia tracker with GT detections:

```bash
python scripts/kitti_mot_evaluate.py /media/vladislav/KINGSTON/trackkitti/training --classes car pedestrian
```

Basic tracker with detector detections:

```bash
python scripts/kitti_mot_evaluate.py /media/vladislav/KINGSTON/trackkitti/training --classes car pedestrian --basic --detections-root data/detections/openpcdet_pointpillar --score-thresh 0.5
```

Export only, no metric computation:

```bash
python scripts/kitti_mot_evaluate.py /media/vladislav/KINGSTON/trackkitti/training --export-only
```

## Generated Outputs

Generated experiment outputs under this repo currently live in:

- `data/detections/`
- `data/kitti_eval/`

These directories are git-ignored and can be regenerated.

## Reproducibility Notes

If you need exact experiment reconstruction later, record at least:

1. main repo commit
2. `third_party/OpenPCDet` commit
3. `third_party/TrackEval` commit
4. the two local third-party patches above
5. checkpoint filename and URL
6. the exact CLI used for precompute / demo / evaluation

For a thesis or paper artifact, using submodules for `TrackEval` and `OpenPCDet` would be a cleaner long-term choice than committing the full cloned directories into the main repo history.
