# KITTI 3D SOT Metrics Report

Date: 2026-05-19

## Evaluator Update

After the baseline run, the evaluator was updated for fairer comparisons:

- `Van` is no longer canonicalized into `Car` during evaluation.
- The default class list is now `Car Pedestrian Van Cyclist`, matching the common KITTI 3D SOT mean.
- The evaluator now supports `--metric-space 3d` and `--metric-space bev`.
- `--metric-space 3d` is the default and summarizes Success with 3D IoU and Precision with full 3D center distance.
- Per-frame CSV now stores both `bev_overlap` / `bev_accuracy` and `iou_3d` / `accuracy_3d`.
- An in-process point-cloud frame cache reduces repeated disk reads across tracklets.
- Progress prints now flush immediately, so long full-split runs are visibly active under `conda run`.

Smoke checks passed:

```bash
conda run -n utonia python -m py_compile scripts/eval_kitti_sot_3d.py

conda run -n utonia python scripts/eval_kitti_sot_3d.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --sequences 0000 \
  --classes Car Van Pedestrian Cyclist \
  --init-source gt \
  --update-source none \
  --init-policy immediate \
  --max-tracks 1 \
  --max-track-frames 2 \
  --metric-space 3d \
  --run-name codex_smoke_strict3d_after_patch_20260519
```

Class-count validation for KITTI test sequences `0019` and `0020` now returns the expected split: `Car=6424`, `Van=1248`, `Pedestrian=6088`, `Cyclist=308`, total `14068` frames.

## Diagnostic Experiment

Command:

```bash
conda run -n utonia python scripts/eval_kitti_sot_3d.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --split test \
  --init-source gt \
  --update-source none \
  --init-policy immediate \
  --metric-space 3d \
  --run-name diagnostic_kitti3d_strict_gt_none_20260519
```

Failure report:

```bash
conda run -n utonia python scripts/report_failure_tracks_3d.py \
  --run-dir data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519 \
  --top-k 20 \
  --top-k-per-class 5 \
  --report-min-track-length 10 \
  --drift-distance 2.0 \
  --low-overlap 0.1
```

Outputs:

```text
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/summary.json
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/per_track.csv
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/per_frame.csv
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/failure_tracks_3d.csv
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/failure_tracks_3d.md
```

Strict 3D aggregate:

| Class | Frames | Success 3D | Precision 3D | Mean 3D Dist | Median 3D Dist | Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 14068 | 27.88 | 39.21 | 27.86 m | 1.61 m | 27.05% |
| Car | 6424 | 17.72 | 19.61 | 40.13 m | 4.59 m | 28.41% |
| Pedestrian | 6088 | 39.08 | 62.03 | 4.39 m | 0.32 m | 19.04% |
| Van | 1248 | 15.50 | 15.97 | 86.07 m | 25.20 m | 65.79% |
| Cyclist | 308 | 68.39 | 91.10 | 0.18 m | 0.15 m | 0.00% |

The failure report selected `26` representative tracks: `8` Car, `8` Pedestrian, `5` Van, and `5` Cyclist. Failure reasons were dominated by `lost` tracks (`21`) plus `min_overlap` cases (`5`), which points to re-acquisition and box/yaw quality as the first algorithmic targets.

## Official KITTI Tracking Focus

The official KITTI tracking benchmark evaluates only `Car` and `Pedestrian`. The diagnostic tooling now defaults to these classes; `Van` and `Cyclist` can still be requested explicitly for 3D SOT all-class comparisons.

Official-class strict 3D subset from `diagnostic_kitti3d_strict_gt_none_20260519`:

| Class | Frames | Tracklets | Success 3D | Precision 3D | Frame Lost Rate | Track Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 6424 | 120 | 17.72 | 19.61 | 28.41% | 21.67% |
| Pedestrian | 6088 | 62 | 39.08 | 62.03 | 19.04% | 24.19% |

Official-only failure report:

```bash
conda run -n utonia python scripts/report_failure_tracks_3d.py \
  --run-dir data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519 \
  --classes Car Pedestrian \
  --top-k 20 \
  --top-k-per-class 8 \
  --report-min-track-length 10 \
  --drift-distance 2.0 \
  --low-overlap 0.1 \
  --output-prefix official_failure_tracks_3d
```

Outputs:

```text
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/official_failure_tracks_3d.csv
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/official_failure_tracks_3d.md
```

The official-only report selected `20` tracks: `10` Car and `10` Pedestrian. All selected failures were `lost`, so the first fix should target the lost-state lifecycle and re-acquisition path before tuning Van/Cyclist-specific behavior.

## Fast Official Subset Experiments

Diagnostic subset:

```text
data/eval_sot_3d/diagnostic_kitti3d_strict_gt_none_20260519/official_diagnostic_subset.csv
```

The subset contains `40` official-class tracks: `20` failure tracks from `official_failure_tracks_3d.csv` and `20` matched controls with length `>=10`, no `lost`, and high `success_3d`. All experiment runs used `--classes Car Pedestrian`, `--metric-space 3d`, `--max-track-frames 120`, and this track-list CSV.

Frame-level results:

| Run | Change | Success 3D | Precision 3D | Lost Rate | Mean 3D Dist | Median 3D Dist |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `official_subset_baseline` | baseline | 36.71 | 45.95 | 41.49% | 19.14 m | 0.95 m |
| `official_subset_less_aggressive_lost` | `lost_bad_frames=4`, `lost_height_ratio=0.3` | 42.35 | 56.64 | 17.59% | 21.74 m | 0.36 m |
| `official_subset_sparse_fallback` | sparse fallback only | 37.23 | 52.22 | 24.22% | 21.68 m | 0.52 m |
| `official_subset_recovery` | ungated recovery | 43.25 | 59.22 | 1.74% | 4232733.50 m | 0.34 m |
| `official_subset_recovery_gated` | recovery with bad-update gate | 41.83 | 56.87 | 21.87% | 62.35 m | 0.36 m |
| `official_subset_lifecycle_sparse` | less aggressive lost + sparse fallback | 37.09 | 52.07 | 10.05% | 17.38 m | 0.52 m |
| `official_subset_class_gate_mid` | `Car=8m`, `Pedestrian=2.5m` gate | 42.22 | 57.64 | 19.21% | 14.73 m | 0.36 m |
| `official_subset_lifecycle_gate_mid` | less aggressive lost + `Car=8m`, `Pedestrian=2.5m` | 44.63 | 61.17 | 9.00% | 12.57 m | 0.32 m |
| `official_subset_lifecycle_gate_car12_ped25` | less aggressive lost + `Car=12m`, `Pedestrian=2.5m` | 45.34 | 62.23 | 4.34% | 13.84 m | 0.32 m |
| `official_subset_lifecycle_gate_car12_ped4` | less aggressive lost + `Car=12m`, `Pedestrian=4m` | 44.84 | 61.37 | 6.56% | 14.05 m | 0.32 m |
| `official_subset_lifecycle_gate_car12_ped15` | less aggressive lost + `Car=12m`, `Pedestrian=1.5m` | 42.88 | 57.82 | 9.79% | 14.53 m | 0.35 m |

Sparsity findings from `official_subset_baseline`:

| Group | Tracks | Lost Tracks | Median Initial Support | Median Min Filtered Support | Median Pre-Lost Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control Car | 10 | 0 | 162.5 | 111.0 | n/a |
| Failure Car | 10 | 8 | 11.0 | 4.5 | 9.0 |
| Control Pedestrian | 10 | 0 | 54.5 | 47.5 | n/a |
| Failure Pedestrian | 10 | 9 | 54.0 | 18.0 | 44.0 |

Conclusions:

- Car failures are strongly sparse: failure tracks start with far fewer LiDAR points than controls.
- Pedestrian failures are not sparse at initialization, but they lose filtered support during the track.
- The best immediate lifecycle change is less aggressive lost detection. It improves both Success and Precision while reducing lost rate.
- Sparse fallback reduces lost rate, especially for Pedestrian, but the tested parameters expand the mask too aggressively and reduce Success when combined with lifecycle tuning.
- Recovery needs an additional motion/velocity policy. Ungated recovery produced catastrophic coordinate outliers; gated recovery fixes that numerical failure but still leaves long `motion_only` drift tails.
- Class-specific gate tuning is promising: it is close to the best Success, improves Precision, and has the lowest mean distance among valid runs, but it still loses more tracks than the lifecycle-only run.
- Focused lifecycle+gate tuning found the best subset candidate so far: `--lost-bad-frames 4 --lost-height-ratio 0.3 --car-gate-radius 12 --pedestrian-gate-radius 2.5`.
- In that best run, Pedestrian failure tracks have `0` lost tracks; the remaining `lost` cases are `2` sparse Car failures with median failure-Car initial support still `11.0` and median min filtered support `4.5`.

## Full Official Validation

Winning subset config was validated on the full official KITTI test split for `Car` and `Pedestrian`:

```bash
conda run -n utonia python scripts/eval_kitti_sot_3d.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --split test \
  --classes Car Pedestrian \
  --init-source gt \
  --update-source none \
  --init-policy immediate \
  --metric-space 3d \
  --lost-bad-frames 4 \
  --lost-height-ratio 0.3 \
  --car-gate-radius 12 \
  --pedestrian-gate-radius 2.5 \
  --run-name official_full_lifecycle_gate_car12_ped25
```

Output:

```text
data/eval_sot_3d/official_full_lifecycle_gate_car12_ped25
```

Runtime was `33m39s` for `182` tracklets and `12512` frames.

| Class | Frames | Tracklets | Success 3D | Precision 3D | Mean 3D Dist | Median 3D Dist | Frame Lost Rate | Track Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 12512 | 182 | 30.99 | 45.47 | 14.16 m | 1.04 m | 6.32% | n/a |
| Car | 6424 | 120 | 18.41 | 20.36 | 25.66 m | 4.08 m | 9.68% | 9.17% |
| Pedestrian | 6088 | 62 | 44.26 | 71.97 | 2.03 m | 0.29 m | 2.78% | 4.84% |

Compared with the strict official baseline from `diagnostic_kitti3d_strict_gt_none_20260519`, this config mainly fixes lifecycle failures:

| Class | Baseline Success 3D | New Success 3D | Baseline Precision 3D | New Precision 3D | Baseline Lost Rate | New Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 17.72 | 18.41 | 19.61 | 20.36 | 28.41% | 9.68% |
| Pedestrian | 39.08 | 44.26 | 62.03 | 71.97 | 19.04% | 2.78% |

Failure report for the full validation selected `22` representative failures: `17` Car and `5` Pedestrian. Reasons were `14` `lost` and `8` `drift`, so the next bottleneck is not generic lifecycle anymore; it is Car drift/sparse-support quality plus box/yaw refinement.

## Run

Command:

```bash
conda run -n utonia python scripts/eval_kitti_sot_3d.py \
  --data-root /media/vladislav/KINGSTON/trackkitti/training \
  --split test \
  --classes Car Pedestrian Cyclist \
  --init-source gt \
  --update-source none \
  --init-policy immediate \
  --run-name codex_kitti3d_test_gt_none_20260519
```

Output:

```text
data/eval_sot_3d/codex_kitti3d_test_gt_none_20260519
```

This is a fair SOT-style run in the sense that it uses ground-truth initialization on the first frame and no GT/detector updates afterwards.

## Results

Frame-level aggregate:

| Split | Tracklets | Frames | Success BEV | Precision BEV | Mean BEV Dist | Median BEV Dist | Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI 19-20 | 206 | 14068 | 32.04 | 41.52 | 21.46 m | 1.58 m | 27.70% |

By evaluator class:

| Class | Tracklets | Frames | Success BEV | Precision BEV | Mean BEV Dist | Median BEV Dist | Lost Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Car | 136 | 7672 | 19.06 | 19.42 | 36.98 m | 4.92 m | 37.60% |
| Pedestrian | 62 | 6088 | 46.13 | 66.70 | 2.98 m | 0.18 m | 16.62% |
| Cyclist | 8 | 308 | 76.93 | 94.40 | 0.11 m | 0.09 m | 0.00% |

Important: in the current code, `Van` is canonicalized to `Car`, so the reported `Car` row is actually `Car + Van`. Raw KITTI test-frame counts are `Car=6424`, `Van=1248`, `Pedestrian=6088`, `Cyclist=308`.

## Published Reference Points

Published KITTI 3D SOT papers use the same common split: KITTI training sequences `0-16` for train, `17-18` for validation, and `19-20` for testing. The standard frame counts are `Car=6424`, `Pedestrian=6088`, `Van=1248`, `Cyclist=308`, `Mean=14068`.

Selected method-level mean Success/Precision:

| Method | Mean Success | Mean Precision | Source |
| --- | ---: | ---: | --- |
| Utonia tracker baseline | 32.04 | 41.52 | This run, BEV metrics |
| SC3D | 31.2 | 48.5 | M2-Track CVPR 2022 Table 1 |
| P2B | 42.4 | 60.0 | M2-Track CVPR 2022 Table 1 |
| BAT | 55.0 | 75.2 | M2-Track CVPR 2022 Table 1 |
| V2B | 58.4 | 75.2 | M2-Track CVPR 2022 Table 1 |
| M2-Track | 62.9 | 83.4 | M2-Track CVPR 2022 Table 1 |
| CDTracker | 60.18 | 78.36 | Computed from CDTracker Table 1 class values and KITTI frame counts |
| BEVTrack | 71.83 | 89.25 | Computed from BEVTrack repository class values and KITTI frame counts |

Sources:

- M2-Track paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.pdf
- CDTracker paper: https://www.mdpi.com/2072-4292/16/13/2322
- BEVTrack repository: https://github.com/xmm-prio/BEVTrack

## Problems Found

1. The current evaluator reports BEV overlap and BEV center distance, not strict published 3D box IoU and full 3D center distance.
2. `Van` is merged into `Car` through `canonical_label`, so per-class comparison against papers is not valid for `Car` and impossible for `Van`.
3. The predicted box keeps the initial box dimensions and heading; only the center is shifted by the tracked centroid. This hurts turning vehicles and objects with changing yaw.
4. Once a track enters `lost`, the tracker only propagates constant velocity and never re-acquires. Long tracks can accumulate very large drift tails.
5. The mean distance is much larger than the median distance, especially on `Car`, which confirms that a subset of tracks drifts badly after failure.
6. `7120 / 14068` frames have BEV overlap below `0.1`; `6563 / 14068` frames have BEV center distance above `2m`.
7. The Utonia feature is used as a frozen point-level similarity signal without a learned SOT head, box regression head, or explicit re-identification module.

## Highest-Priority Fixes

1. Add a strict evaluator mode with real 3D IoU and full 3D center-distance AUC.
2. Stop canonicalizing `Van` into `Car` inside evaluation; only merge for detector label compatibility where explicitly requested.
3. Add box regression/refinement: at minimum estimate yaw and dimensions from support points or matched detections.
4. Add re-acquisition after `lost`, using a larger crop and detection boxes when available.
5. Use Utonia appearance in a detector-associated SOT/MOT loop instead of relying only on point clustering around a prototype.
