# OpenPCDet Detections

This repo vendors `third_party/OpenPCDet` and now includes minimal scripts to precompute KITTI tracking detections for research use.

## What Is Included
- `scripts/precompute_kitti_tracking_detections.py`: generic precompute runner.
- `scripts/run_pointpillar_tracking_detections.sh`: thin wrapper for PointPillars.
- `scripts/run_pointrcnn_tracking_detections.sh`: thin wrapper for PointRCNN.

## Dataset Layout
- The scripts expect KITTI tracking training data at a root containing:
```text
training/
  calib/
  label_02/
  velodyne/
```
- The default path used in this workspace is:
```text
/media/vladislav/KINGSTON/trackkitti/training
```

## Third-Party Setup
- The detection scripts expect OpenPCDet at `third_party/OpenPCDet`.
- The Python precompute script changes into `third_party/OpenPCDet/tools` internally before loading configs, because OpenPCDet uses relative config includes.
- `third_party/` is now gitignored in this repo, so the vendored detector checkout stays local by default.

## Bootstrap OpenPCDet
- Clone OpenPCDet into `third_party/`:
```bash
mkdir -p third_party
git clone https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
```
- Install the Python requirements from the vendored repo:
```bash
pip install -r third_party/OpenPCDet/requirements.txt
```
- Install `spconv`.
- In this workspace, `spconv 2.3.8` worked with PyTorch 2.5.0.
- Install OpenPCDet itself:
```bash
python third_party/OpenPCDet/setup.py develop
```
- If CUDA ops are not built correctly, OpenPCDet imports may fail or inference will crash when the model runs.
- Quick import checks:
```bash
python -c "import pcdet; print(pcdet.__file__)"
python -c "import spconv; print(spconv.__version__)"
```

## Expected Third-Party Layout
```text
third_party/
  OpenPCDet/
    checkpoints/
    pcdet/
    tools/
```

## Dependencies
- Required runtime pieces for these scripts:
  - PyTorch with CUDA
  - `spconv`
  - OpenPCDet Python requirements
  - built OpenPCDet CUDA ops
- In this workspace, the following already worked:
  - `import pcdet`
  - `import spconv`
  - PointPillars inference on a sample KITTI frame

## Checkpoint Download
- Create the checkpoints directory if needed:
```bash
mkdir -p third_party/OpenPCDet/checkpoints
```
- PointPillars checkpoint:
```bash
cd third_party/OpenPCDet/checkpoints
gdown --fuzzy "https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing" -O pointpillar_kitti.pth
```
- PointRCNN checkpoint:
```bash
cd third_party/OpenPCDet/checkpoints
gdown --fuzzy "https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing" -O pointrcnn_kitti.pth
```
- If `gdown` is missing:
```bash
pip install gdown
```

## Checkpoints
- PointPillars checkpoint path:
```text
third_party/OpenPCDet/checkpoints/pointpillar_kitti.pth
```
- PointRCNN checkpoint path:
```text
third_party/OpenPCDet/checkpoints/pointrcnn_kitti.pth
```
- The PointRCNN checkpoint can be downloaded with:
```bash
cd third_party/OpenPCDet/checkpoints
gdown --fuzzy "https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing" -O pointrcnn_kitti.pth
```

## Output Layout
- Detections are written under `data/detections/<model>/` by default.
- Two output formats are saved:
  - `npz/<sequence>/<frame>.npz`
  - `txt_lidar/<sequence>/<frame>.txt`
- The `.npz` file stores:
  - `pred_boxes`: `(N, 7)` in OpenPCDet LiDAR format `[x, y, z, dx, dy, dz, heading]`
  - `pred_scores`: `(N,)`
  - `pred_labels`: `(N,)`, 1-based class ids matching OpenPCDet
- The text file stores one detection per line as:
```text
<class_name> <score> <x> <y> <z> <dx> <dy> <dz> <heading>
```

## Usage
- Full first-time setup, then precompute detections:
```bash
mkdir -p third_party
git clone https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
pip install -r third_party/OpenPCDet/requirements.txt
python third_party/OpenPCDet/setup.py develop
mkdir -p third_party/OpenPCDet/checkpoints
cd third_party/OpenPCDet/checkpoints && gdown --fuzzy "https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing" -O pointpillar_kitti.pth
cd /home/vladislav/Documents/Utonia-tracker
bash scripts/run_pointpillar_tracking_detections.sh
```
- Run PointPillars on the full KITTI tracking training set:
```bash
bash scripts/run_pointpillar_tracking_detections.sh
```
- Run PointRCNN on the full KITTI tracking training set:
```bash
bash scripts/run_pointrcnn_tracking_detections.sh
```
- Run only selected sequences:
```bash
bash scripts/run_pointpillar_tracking_detections.sh --sequence 0000 0001
```
- Run a short smoke test:
```bash
bash scripts/run_pointrcnn_tracking_detections.sh --sequence 0000 --max_frames 8
```
- Override the score threshold after model inference:
```bash
bash scripts/run_pointrcnn_tracking_detections.sh --score_thresh 0.3
```
- Override the output directory:
```bash
bash scripts/run_pointpillar_tracking_detections.sh --output_root /tmp/kitti_dets
```
- Recompute existing outputs:
```bash
bash scripts/run_pointrcnn_tracking_detections.sh --overwrite
```

## Notes
- These scripts are intentionally minimal and aimed at research iteration, not a production detection pipeline.
- They process KITTI tracking point clouds directly from `training/velodyne/<sequence>/*.bin`.
- They currently save LiDAR-space detections, which is the most useful format for the tracker work in this repo.
- For tracking experiments, using GT labels and these precomputed detections side by side is recommended to separate association errors from detector errors.
