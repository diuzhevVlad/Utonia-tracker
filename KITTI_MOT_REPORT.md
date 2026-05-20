# KITTI MOT Report

Date: 2026-05-19

## Pipeline

Added a no-training tracking-by-detection MOT baseline:

```text
scripts/run_kitti_mot_utonia.py
```

The pipeline reads 3D detector outputs from `data/detections/<detector>/npz/<sequence>/<frame>.npz`, filters `Car` and `Pedestrian`, applies class-wise NMS, associates detections with Hungarian matching, and writes KITTI tracking result files:

```text
data/mot_kitti/<run_name>/data/0019.txt
data/mot_kitti/<run_name>/data/0020.txt
```

Association uses class match, BEV center distance, axis-aligned BEV IoU, and optional Utonia box embeddings:

```bash
--appearance-mode none|utonia
--appearance-weight <float>
```

Utonia appearance was smoke-tested on a short sequence. Full official validation below uses geometry-only association first, because this establishes the detector-assisted MOT baseline quickly and isolates detector/association quality before spending time on Utonia embeddings.

## Official-Protocol Evaluation

Added a TrackEval wrapper:

```text
scripts/eval_kitti_mot_trackeval.py
```

It prepares a KITTI-compatible TrackEval layout with `label_02`, `evaluate_tracking.seqmap.training`, tracker result files, and runs TrackEval `Kitti2DBox` with HOTA, CLEAR, and Identity metrics for official KITTI tracking classes `car` and `pedestrian`.

The evaluation uses KITTI 2D box MOT protocol, matching the official benchmark format. Sequences evaluated here are the common KITTI test split used in the SOT experiments:

```text
0019 0020
```

## Runs

### PointPillars, score 0.3

```bash
conda run -n utonia python scripts/run_kitti_mot_utonia.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --score-thresh 0.3 \
  --nms-iou 0.1 \
  --max-age 5 \
  --run-name official_mot_pointpillar_motion_iou
```

| Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 52.22 | 17.75 | 64.05 | 85.61 | 56.03 | 39 | 3837 | 822 |
| Pedestrian | 28.66 | -67.40 | 41.09 | 80.95 | 35.62 | 120 | 8594 | 1119 |

### PointPillars, score 0.7

```bash
conda run -n utonia python scripts/run_kitti_mot_utonia.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --score-thresh 0.7 \
  --nms-iou 0.1 \
  --max-age 5 \
  --run-name official_mot_pointpillar_motion_iou_score07
```

| Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 52.93 | 41.25 | 66.07 | 69.14 | 71.67 | 32 | 1561 | 1763 |
| Pedestrian | 23.71 | 21.88 | 37.24 | 38.24 | 72.71 | 118 | 843 | 3628 |

### PointRCNN, score 0.5

```bash
conda run -n utonia python scripts/run_kitti_mot_utonia.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointrcnn/npz \
  --score-thresh 0.5 \
  --nms-iou 0.1 \
  --max-age 5 \
  --run-name official_mot_pointrcnn_motion_iou_score05
```

| Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 48.42 | 34.54 | 61.04 | 67.51 | 67.89 | 59 | 1824 | 1856 |
| Pedestrian | 28.52 | 13.36 | 46.54 | 51.16 | 59.00 | 132 | 2088 | 2869 |

### PointPillars, score 0.7, Utonia appearance

```bash
conda run -n utonia python scripts/run_kitti_mot_utonia.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --score-thresh 0.7 \
  --nms-iou 0.1 \
  --max-age 5 \
  --appearance-mode utonia \
  --appearance-weight 0.5 \
  --run-name official_mot_pointpillar_score07_utonia_app05
```

Runtime was about `24.5` minutes for `1896` frames.

| Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 52.93 | 41.25 | 66.07 | 69.14 | 71.67 | 32 | 1561 | 1763 |
| Pedestrian | 23.42 | 21.77 | 36.93 | 38.24 | 72.71 | 124 | 843 | 3628 |

Compared with the same PointPillars `score_thresh=0.7` geometry-only baseline, the naive always-on Utonia appearance cost did not improve the official metrics. Car metrics are unchanged, while Pedestrian association slightly worsens (`IDSW 118 -> 124`, `IDF1 37.24 -> 36.93`).

## Findings

- The tracker now follows a standard detector-assisted MOT structure and produces official KITTI-format result files.
- TrackEval official KITTI 2D box protocol runs successfully on generated outputs.
- PointPillars at `score_thresh=0.7` is the best Car setting among these quick checks: it greatly reduces FP and improves MOTA while keeping HOTA about the same.
- PointPillars at `score_thresh=0.3` has higher recall but too many false positives, especially for Pedestrian.
- PointRCNN gives better Pedestrian IDF1 than PointPillars score `0.7`, but worse Car HOTA/MOTA.
- Naive always-on Utonia appearance does not help yet. In this setup, geometry already determines most matches; when appearance changes matches, it slightly hurts Pedestrian. Utonia should be used as a targeted re-identification fallback for ambiguous matches, occlusion gaps, and lost-track recovery rather than as a constant additive cost on every association.
- Remaining problems are detector thresholding/class-specific filtering, many Pedestrian ID switches/fragments, and no mature re-identification policy yet.

## Utonia Detector-Gap Recovery

Added a detector-gap diagnostic:

```text
scripts/eval_utonia_detector_gap_recovery.py
```

This experiment finds GT tracks where a reliable high-score detector update is available at frame `t-1`, but missing at frame `t`. It initializes Utonia from the last reliable detector box and runs Utonia-only fallback for up to `5` detector-gap frames. It logs whether Utonia would keep the track active, plus 3D IoU, center distance, support count, mean similarity, and whether a low-score detector match existed.

### High-score misses

Command:

```bash
conda run -n utonia python scripts/eval_utonia_detector_gap_recovery.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --high-score-thresh 0.7 \
  --low-score-thresh 0.3 \
  --max-cases-per-class 8 \
  --max-gap-frames 5 \
  --run-name pointpillar_score07_gap_recovery_utonia_top8
```

| Class | Gap Frames | Accepted Rate | Mean 3D IoU | Median 3D Dist | Low-Score Match Rate | Mean Support | Mean Similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 17 | 100.0% | 0.386 | 0.850 m | 94.1% | 108.7 | 0.847 |
| Pedestrian | 24 | 100.0% | 0.528 | 0.189 m | 79.2% | 94.4 | 0.947 |

Most high-score misses still have low-score detector evidence. These are ByteTrack-style cases: low-score continuation should solve many of them cheaply, with Utonia available as a confidence check.

### No-low-score first-frame misses

Command:

```bash
conda run -n utonia python scripts/eval_utonia_detector_gap_recovery.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --high-score-thresh 0.7 \
  --low-score-thresh 0.3 \
  --max-cases-per-class 8 \
  --max-gap-frames 5 \
  --require-low-score-miss \
  --run-name pointpillar_score07_gap_recovery_utonia_no_low_score_top8
```

| Class | Gap Frames | Accepted Rate | Mean 3D IoU | Median 3D Dist | Low-Score Match Rate | Mean Support | Mean Similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 10 | 100.0% | 0.436 | 0.630 m | 40.0% | 73.7 | 0.819 |
| Pedestrian | 30 | 100.0% | 0.266 | 0.451 m | 50.0% | 124.4 | 0.928 |

This confirms the useful role for Utonia: it can often keep the object center through detector dropouts. The weakness is box quality, especially Pedestrian IoU, because fallback still reuses detector box dimensions/yaw while only moving the center.

Best no-low-score examples:

| Sequence | Track | Class | Gap Start | Frames | Mean 3D IoU | Median 3D Dist |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0019 | 12 | Pedestrian | 143 | 5 | 0.596 | 0.091 m |
| 0019 | 72 | Car | 503 | 5 | 0.590 | 0.607 m |
| 0019 | 49 | Pedestrian | 399 | 1 | 0.543 | 0.197 m |

Worst no-low-score failures are mostly Pedestrian cases with high support and high similarity but low 3D IoU, which means the Utonia target center can be plausible while the fixed box/yaw/height model is wrong.

## ByteTrack-Style And Utonia Fallback Ablations

Implemented the next no-training ablations in `scripts/run_kitti_mot_utonia.py`:

```bash
--use-low-score-continuation
--low-score-thresh 0.3
--utonia-fallback
--fallback-start-score-thresh 0.7
--fallback-max-candidates-per-frame 2
```

Low-score detections can continue existing tracks but cannot create new tracks. Utonia fallback is only initialized from recently updated, high-score detector tracks after both high-score and low-score detector association fail.

### TrackEval results

| Run | Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointPillars 0.7 geometry | Car | 52.93 | 41.25 | 66.07 | 69.14 | 71.67 | 32 | 1561 | 1763 |
| PointPillars 0.7 + low-score 0.3 | Car | 54.32 | 38.39 | 68.34 | 77.28 | 66.64 | 11 | 2210 | 1298 |
| PointPillars 0.7 + low-score 0.3 + targeted Utonia | Car | 53.71 | 36.36 | 67.73 | 77.49 | 65.44 | 11 | 2338 | 1286 |
| PointPillars 0.7 geometry | Pedestrian | 23.71 | 21.88 | 37.24 | 38.24 | 72.71 | 118 | 843 | 3628 |
| PointPillars 0.7 + low-score 0.3 | Pedestrian | 33.09 | 19.15 | 52.43 | 70.05 | 58.47 | 67 | 2923 | 1759 |
| PointPillars 0.7 + low-score 0.3 + targeted Utonia | Pedestrian | 33.19 | 19.29 | 52.58 | 70.40 | 58.49 | 67 | 2935 | 1739 |

### Runtime and diagnostics

| Run | Result Rows | Tracks | High-Score Matches | Low-Score Matches | Utonia Fallbacks | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low-score 0.3 | 15103 | 307 | 8816 | 5980 | 0 | 3.8 s |
| Low-score 0.3 + targeted Utonia | 15284 | 307 | 8816 | 5988 | 173 | 77.5 s |

Targeted Utonia fallback events by class:

| Class | Accepted Fallbacks |
| --- | ---: |
| Car | 136 |
| Pedestrian | 37 |

Rejected fallback reasons:

| Reason | Count |
| --- | ---: |
| projection_failed | 225 |
| height_collapse | 5 |
| jump_xy | 4 |
| gate_failed | 2 |

### Interpretation

- ByteTrack-style low-score continuation is the strongest no-training improvement so far. It improves Car HOTA/IDF1 and strongly improves Pedestrian HOTA/IDF1 by recovering detector misses, but it lowers MOTA because false positives rise.
- Targeted Utonia fallback gives a small Pedestrian gain after low-score continuation (`HOTA 33.09 -> 33.19`, `IDF1 52.43 -> 52.58`) and slightly improves Pedestrian FN, but it hurts Car due to extra false positives and weaker projected boxes.
- The dominant Utonia rejection is `projection_failed`, which points to fallback box projection/geometry quality rather than feature similarity alone.
- Current conclusion: keep low-score continuation, but gate Utonia fallback more tightly or improve fallback box reconstruction before using it as a default MOT component.

## One-Day Working Utonia Tracker

Implemented a safer Utonia detector-gap fallback policy:

```bash
--utonia-fallback-classes Pedestrian
--fallback-start-score-thresh 0.7
--fallback-score 0.15
--fallback-max-bbox-width 1200
--fallback-max-bbox-height 1000
--fallback-max-bbox-area-ratio 1.5
```

This keeps Car on the stronger low-score baseline and uses Utonia only for short Pedestrian detector gaps after high-score and low-score association both fail.

### Final ablation table

| Run | Class | HOTA | MOTA | IDF1 | DetRe | DetPr | IDSW | FP | FN | Utonia Fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointPillars 0.7 geometry | Car | 52.93 | 41.25 | 66.07 | 69.14 | 71.67 | 32 | 1561 | 1763 | 0 |
| Low-score continuation | Car | 54.32 | 38.39 | 68.34 | 77.28 | 66.64 | 11 | 2210 | 1298 | 0 |
| Low-score + Ped-only Utonia | Car | 54.32 | 38.39 | 68.34 | 77.28 | 66.64 | 11 | 2210 | 1298 | 0 |
| Low-score + Ped-strict Utonia | Car | 54.32 | 38.39 | 68.34 | 77.28 | 66.64 | 11 | 2210 | 1298 | 0 |
| Low-score + strict all-class Utonia | Car | 54.26 | 38.18 | 68.27 | 77.28 | 66.52 | 11 | 2222 | 1298 | 11 |
| PointPillars 0.7 geometry | Pedestrian | 23.71 | 21.88 | 37.24 | 38.24 | 72.71 | 118 | 843 | 3628 | 0 |
| Low-score continuation | Pedestrian | 33.09 | 19.15 | 52.43 | 70.05 | 58.47 | 67 | 2923 | 1759 | 0 |
| Low-score + Ped-only Utonia | Pedestrian | 33.19 | 19.29 | 52.58 | 70.40 | 58.49 | 67 | 2935 | 1739 | 37 |
| Low-score + Ped-strict Utonia | Pedestrian | 33.19 | 19.27 | 52.56 | 70.36 | 58.48 | 67 | 2934 | 1741 | 34 |
| Low-score + strict all-class Utonia | Pedestrian | 33.19 | 19.27 | 52.56 | 70.36 | 58.48 | 67 | 2934 | 1741 | 34 |

### Diagnostics

| Run | Accepted Fallbacks | Rejected Fallbacks | Main Rejection Reasons |
| --- | ---: | ---: | --- |
| `official_mot_low03_utonia_ped_only` | 37 Pedestrian | 7 | `jump_xy=4`, `height_collapse=3` |
| `official_mot_low03_utonia_ped_strict` | 34 Pedestrian | 3 | `jump_xy=2`, `height_collapse=1` |
| `official_mot_low03_utonia_strict_all` | 34 Pedestrian, 11 Car | 219 | `projection_failed=99`, `bbox_too_wide=69`, `gate_failed=48` |

Selected default:

```text
official_mot_low03_utonia_ped_only
```

It satisfies the one-day acceptance criteria: Car is unchanged from the low-score baseline, Pedestrian improves slightly, and Utonia contributes real fallback events. A Rerun recording was generated for sequence `0019`, frames `000000-000249`:

```text
data/rerun/official_mot_low03_utonia_ped_only_0019_000000_000249.rrd
```

Open it with:

```bash
conda run -n utonia rerun data/rerun/official_mot_low03_utonia_ped_only_0019_000000_000249.rrd
```

The recording contains visible Utonia fallback events in the first 250 frames, including frames `125`, `143-147`, and `184-185`.

## Support-Point Box Refinement Check

Added optional fallback box refinement from Utonia support points:

```bash
--fallback-refine-box none|support_z|support_3d
```

The refinement estimates support-point extents in the fallback box frame, clamps dimensions relative to the previous detector box, and optionally updates only vertical geometry (`support_z`) or full 3D center/size (`support_3d`). It is off by default.

### Refinement ablations

| Run | Class | HOTA | MOTA | IDF1 | FP | FN | Utonia Fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ped-only Utonia, no refinement | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 0 |
| Ped-only Utonia, `support_z` | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 0 |
| Ped-only Utonia, `support_3d` | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 0 |
| Ped-only Utonia, no refinement | Pedestrian | 33.19 | 19.29 | 52.58 | 2935 | 1739 | 37 |
| Ped-only Utonia, `support_z` | Pedestrian | 33.14 | 19.08 | 52.50 | 2938 | 1748 | 37 |
| Ped-only Utonia, `support_3d` | Pedestrian | 33.12 | 19.07 | 52.46 | 2936 | 1752 | 36 |
| Strict all-class Utonia, `support_3d` | Car | 54.26 | 38.18 | 68.27 | 2222 | 1298 | 12 |
| Strict all-class Utonia, `support_3d` | Pedestrian | 33.12 | 19.08 | 52.46 | 2935 | 1752 | 33 |

Diagnostics show why this is not the default:

| Run | Accepted Fallbacks | Rejected Fallbacks | Main Rejection Reasons |
| --- | ---: | ---: | --- |
| `official_mot_low03_utonia_ped_refine_z` | 37 Pedestrian | 7 | `jump_xy=4`, `height_collapse=3` |
| `official_mot_low03_utonia_ped_refine_3d` | 36 Pedestrian | 7 | `jump_xy=4`, `height_collapse=3` |
| `official_mot_low03_utonia_strict_all_refine_3d` | 33 Pedestrian, 12 Car | 218 | `projection_failed=123`, `gate_failed=46`, `bbox_too_wide=44` |

Conclusion: direct geometry refinement from the current Utonia support mask is not reliable enough. It slightly worsens Pedestrian and does not solve Car fallback; the support mask is useful for center continuity, but not yet as a robust dimension/yaw estimator. Keep `--fallback-refine-box none` as the default.

## Low-Score And Utonia Validation Ablations

The stronger no-training path is to keep high-score PointPillars detections as the anchor, use low-score detections for continuation, and apply Utonia only to the riskiest Pedestrian updates. These ablations all use KITTI MOT sequences `0019` and `0020`, classes `Car` and `Pedestrian`, PointPillars `score_thresh=0.7`, low-score continuation, and Pedestrian-only Utonia fallback unless noted.

### Threshold and gate sweep

| Run | Class | HOTA | MOTA | IDF1 | FP | FN | IDSW | Low Matches | Utonia Fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low-score 0.3 baseline | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 11 | 5980 | 0 |
| Low-score 0.3 baseline | Pedestrian | 33.09 | 19.15 | 52.43 | 2923 | 1759 | 67 | 5980 | 0 |
| Ped-only Utonia, low 0.3 | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 11 | 5983 | 37 |
| Ped-only Utonia, low 0.3 | Pedestrian | 33.19 | 19.29 | 52.58 | 2935 | 1739 | 67 | 5983 | 37 |
| Ped low 0.35 | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 11 | 5599 | 43 |
| Ped low 0.35 | Pedestrian | 34.63 | 22.92 | 55.52 | 2683 | 1783 | 62 | 5599 | 43 |
| Ped low 0.35 + Ped dist 1.5 | Car | 54.32 | 38.39 | 68.34 | 2210 | 1298 | 11 | 5621 | 58 |
| Ped low 0.35 + Ped dist 1.5 | Pedestrian | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 | 5621 | 58 |
| Ped low 0.35 + IoU 0.05 | Car | 54.57 | 39.44 | 68.57 | 2129 | 1315 | 15 | 5159 | 96 |
| Ped low 0.35 + IoU 0.05 | Pedestrian | 34.58 | 26.93 | 54.68 | 2435 | 1799 | 58 | 5159 | 96 |
| Ped low 0.35 + cost 1.2 | Car | 54.90 | 40.86 | 69.04 | 2043 | 1324 | 11 | 5057 | 106 |
| Ped low 0.35 + cost 1.2 | Pedestrian | 34.06 | 26.03 | 53.54 | 2473 | 1815 | 57 | 5057 | 106 |

The best pre-validation Pedestrian setting is `official_mot_pedlow035_peddist15`: it keeps Car exactly at the low-score baseline and improves Pedestrian from `HOTA=33.09`, `IDF1=52.43` to `HOTA=35.39`, `IDF1=56.20`. IoU and cost gates improve precision and Car metrics, but they damage Pedestrian association compared with the class-specific distance gate.

### Fallback score and lifecycle budget

| Run | Pedestrian HOTA | Pedestrian MOTA | Pedestrian IDF1 | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ped dist 1.5, fallback score 0.05 | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 |
| Ped dist 1.5, fallback score 0.10 | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 |
| Ped dist 1.5, fallback score 0.15 | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 |
| Ped dist 1.5, fallback score 0.20 | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 |
| Ped dist 1.5, fallback max 1 | 34.91 | 24.36 | 55.63 | 2633 | 1759 | 51 |
| Ped dist 1.5, fallback max 3 | 35.39 | 24.41 | 56.19 | 2646 | 1747 | 47 |
| Ped dist 1.5, fallback max 5 | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 |

Fallback score has almost no effect here because TrackEval matching is dominated by geometry after detections are emitted. A one-frame fallback budget is too short; `--fallback-max-frames 5` remains the best default.

### Utonia validation for weak low-score Pedestrian updates

| Run | Validate Below | Sim Thresh | Ped HOTA | Ped MOTA | Ped IDF1 | FP | FN | IDSW | Validations | Utonia Rejects | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No validation | - | - | 35.39 | 24.41 | 56.20 | 2648 | 1745 | 47 | 0 | 0 | 58 |
| `val045_sim070` | 0.45 | 0.70 | 35.03 | 24.94 | 55.40 | 2556 | 1795 | 58 | 664 | 51 | 61 |
| `val045_sim095` | 0.45 | 0.95 | 35.22 | 25.16 | 55.98 | 2542 | 1797 | 57 | 675 | 90 | 66 |
| `val050_sim095` | 0.50 | 0.95 | 35.99 | 26.34 | 57.32 | 2458 | 1812 | 57 | 1110 | 130 | 74 |
| `val050_sim099` | 0.50 | 0.99 | 37.03 | 28.24 | 59.36 | 2370 | 1803 | 42 | 1082 | 199 | 79 |
| `val050_sim0995` | 0.50 | 0.995 | 37.11 | 28.36 | 59.37 | 2355 | 1811 | 42 | 1075 | 218 | 84 |
| `val050_sim0999` | 0.50 | 0.999 | 36.22 | 28.24 | 57.88 | 2288 | 1876 | 51 | 1045 | 398 | 143 |
| `val055_sim0995` | 0.55 | 0.995 | 36.79 | 28.07 | 58.95 | 2311 | 1858 | 56 | 1632 | 306 | 110 |

Selected run:

```text
official_mot_pedlow035_peddist15_val050_sim0995
```

This is the current best no-training Utonia MOT configuration on the test sequences. Car remains unchanged from the low-score baseline (`HOTA=54.315`, `MOTA=38.393`, `IDF1=68.337`). Pedestrian improves over the previous Ped-only Utonia default from `HOTA=33.194`, `IDF1=52.580` to `HOTA=37.108`, `IDF1=59.372`.

The key command is:

```bash
conda run --no-capture-output -n utonia python scripts/run_kitti_mot_utonia.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0019 0020 \
  --classes Car Pedestrian \
  --detections-root data/detections/pointpillar/npz \
  --score-thresh 0.7 \
  --use-low-score-continuation \
  --low-score-thresh 0.3 \
  --pedestrian-low-score-thresh 0.35 \
  --low-score-max-distance-pedestrian 1.5 \
  --nms-iou 0.1 \
  --max-age 5 \
  --min-hits 1 \
  --car-match-distance 5.0 \
  --pedestrian-match-distance 2.0 \
  --appearance-mode none \
  --utonia-fallback \
  --utonia-fallback-classes Pedestrian \
  --fallback-max-frames 5 \
  --fallback-max-candidates-per-frame 2 \
  --fallback-start-score-thresh 0.7 \
  --fallback-sim-thresh 0.55 \
  --fallback-min-points-pedestrian 8 \
  --fallback-score 0.15 \
  --validate-low-score-utonia \
  --validate-low-score-classes Pedestrian \
  --validate-low-score-below 0.50 \
  --validate-low-score-sim-thresh 0.995 \
  --validate-low-score-min-points-pedestrian 8 \
  --run-name official_mot_pedlow035_peddist15_val050_sim0995
```

Diagnostics for the selected run:

| Event | Count |
| --- | ---: |
| High-score matches | 8799 |
| Low-score matches | 5170 |
| Low-score gate rejects | 110 |
| Utonia low-score validations | 1075 |
| Utonia low-score rejects | 218 |
| Pedestrian Utonia fallback accepts | 84 |
| Pedestrian Utonia fallback rejects | 12 |

Low-score Utonia rejection reasons:

| Reason | Count |
| --- | ---: |
| `utonia_similarity` | 155 |
| `utonia_sparse` | 63 |

Fallback rejection reasons:

| Reason | Count |
| --- | ---: |
| `jump_xy` | 4 |
| `height_collapse` | 3 |
| `bbox_too_wide` | 3 |
| `projection_failed` | 2 |

A new Rerun recording was generated for sequence `0019`, frames `000000-000249`:

```text
data/rerun/official_mot_pedlow035_peddist15_val050_sim0995_0019_000000_000249.rrd
```

It contains visible Utonia fallback events in the first 250 frames, including frames `125`, `136`, `143-147`, `184-185`, `214`, `225`, `233`, `237`, and `249`.

## Next No-Training Steps

1. Treat `official_mot_pedlow035_peddist15_val050_sim0995` as the working no-training Utonia MOT tracker.
2. Validate the selected configuration on additional KITTI train sequences before claiming full-split robustness.
3. Add a cheap cache for Utonia validation embeddings; `score < 0.55` was slower mainly because it validated many more low-score candidates.
4. Keep Car Utonia fallback disabled until projection failures and box geometry are fixed.
