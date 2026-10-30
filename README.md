# Stereo visual odometry for a GNSS-denied MAV (EuRoC)

Stereo VO + object detection on the EuRoC MAV dataset (bachelor diploma, KNU).

Pipeline (per frame, `vo/`):

1. **Calibration / rectification** (`euroc.py`, `features.Rectifier`): intrinsics + radial-tangential
   distortion + `T_BS` extrinsics from the EuRoC yaml files; `T_C1C0 = T_BC1⁻¹ · T_BC0`;
   `cv2.stereoRectify` → epipolar lines become image rows.
2. **Feature matching left↔right** (`features.StereoMatcher`): ORB / SIFT / GFTT, grid bucketing,
   matching only among candidates on the same row with a positive disparity, Lowe ratio test,
   mutual check, LK sub-pixel refinement of the right correspondence.
3. **Triangulation**: `Z = f·B/d`, `X = (u−cx)Z/f`, `Y = (v−cy)Z/f`, then into the world frame
   with the current pose (landmarks live in one world frame).
4. **Tracking + PnP** (`odometry.py`): landmarks tracked with pyramidal LK (forward–backward
   check); `solvePnPRansac` (constant-velocity initial guess) + Levenberg–Marquardt refinement;
   a new stereo keyframe is added when tracking gets weak.
5. **Evaluation** (`evaluation.py`): Umeyama SE(3) ATE (and the Sim(3) scale as a diagnostic),
   first-pose-anchored drift, RPE over 1 s, triangulation accuracy vs the Leica laser scan,
   TUM files for `evo`.

## Run

```bash
source ~/py310_vio/bin/activate                    # VO only needs opencv, numpy, scipy, pyyaml, matplotlib
python run_vo.py --seq data/euroc/V1_01_easy/mav0  # -> results/V1_01_easy_orb_{metrics.json,trajectory.png,est.tum,gt.tum}
python run_vo.py --seq data/euroc/V2_01_easy/mav0 --detector sift --live
python run_vo.py --compare                         # every detector on every sequence -> results/comparison.md
python run_vo.py --video data/videos/cam0_vid.mp4 data/videos/cam1_vid.mp4 --seq data/euroc/V2_01_easy/mav0 --live
                                                   # synchronised stereo video; calibration from --seq, no GT

~/py_envs/py_env1/bin/python app.py                # GUI (torch + PySide6): VO window + detection
~/py_envs/py_env1/bin/python -m detection.train --epochs 12 --out models/ssd_v2.pth
~/py_envs/py_env1/bin/python -m detection.train --eval-only models/ssd_v2.pth
```

## Layout

```
vo/            config, EuRoC loading, stereo front-end, odometry, geometry, evaluation
detection/     SSDLite model + time-based train/val split, training, IoU/mAP metrics
run_vo.py      CLI / comparison runner      app.py   GUI
data/euroc/    V1_01_easy, V2_01_easy, MH_02_easy   data/videos/  raw videos
models/        detector weights             results/  outputs      docs/  diploma diagrams
_archive/      tarball of the original code (before the 2026-10-29 rewrite)
```

## Results (whole sequences, `python run_vo.py --compare`)

ATE = RMSE after SE(3) Umeyama alignment (no scale fit). Drift = final position error with only the
first pose aligned, as % of distance flown (the GNSS-denied scenario). Depth error = median distance
of triangulated points to the Leica laser scan (V1/V2 only), using ground-truth poses.

| sequence | detector | ATE [m] | drift [%] | RPE 1 s [m / deg] | Sim3 scale | lost frames | depth err [m] | fps |
|---|---|---|---|---|---|---|---|---|
| V1_01_easy (58 m) | ORB | 0.218 | 0.97 | 0.047 / 0.58 | 1.027 | 2 | 0.070 | 33 |
| | SIFT | 0.156 | 0.39 | 0.047 / 0.55 | 1.028 | 3 | 0.067 | 17 |
| | GFTT+KLT | **0.096** | 0.57 | 0.044 / 0.47 | 1.014 | 2 | 0.069 | 49 |
| V2_01_easy (36 m) | ORB | **0.097** | 0.54 | 0.020 / 0.99 | 1.030 | 94 | 0.051 | 48 |
| | SIFT | 0.115 | 0.62 | 0.016 / 0.40 | 1.009 | **5** | 0.051 | 23 |
| | GFTT+KLT | 0.192 | 1.87 | 0.018 / 0.55 | 1.033 | 96 | 0.051 | 62 |
| MH_02_easy (73 m) | ORB | 0.075 | 0.40 | 0.014 / 0.16 | 0.997 | 0 | – | 61 |
| | SIFT | **0.073** | **0.20** | 0.012 / 0.12 | 0.999 | 0 | – | 28 |
| | GFTT+KLT | 0.109 | 0.57 | 0.013 / 0.49 | 1.001 | 9 | – | 57 |

The ~95 lost frames on V2_01 are the landing on a mattress (closer than 0.15 m, repetitive texture);
only SIFT survives it. Triangulation error grows with depth as expected for stereo (σ_Z ∝ Z²/(f·B)):
~4 cm below 2 m, ~9 cm at 4–6 m.

Old vs new front-end on the same 40 V1_01 frames: median depth error vs laser **2.14 m → 0.063 m**
(1 % → 68 % of points within 10 cm).

Detector (SSDLite, held-out last 20 % of every labelled segment, IoU ≥ 0.5): old checkpoint
mAP50 = 0.01, retrained `models/ssd_v2.pth` mAP50 = 0.31 (board 0.67, mat 0.33, pipe 0.18,
radiator 0.05). The best epoch was picked on the same split, so this is slightly optimistic.

## What was wrong in the original code (fixed)

| # | Bug | Effect |
|---|-----|--------|
| 1 | Stereo extrinsic computed as `T_BC1·T_BC0⁻¹` instead of `T_BC1⁻¹·T_BC0` | baseline expressed in the body frame → OpenCV rectified the rig as a **vertical** stereo pair; every depth was wrong (median 2.1 m error vs laser scan, now 6 cm) |
| 2 | Left/right images paired by sorted index | V2_01 has 8 missing left images → every pair was 0.4 s apart; MH_02 had 111 unpaired frames. Now paired by timestamp |
| 3 | Re-initialisation triangulated in the *current* camera frame | global pose reset at every re-init; the trajectory was a set of unrelated pieces. Now landmarks live in one world frame and poses are chained |
| 4 | Only 14 points kept (`KP_PER_FRAME`), chosen as the *closest* ones; `MIN_MATCHES = 14` | ill-conditioned PnP, re-init almost every frame |
| 5 | Stereo matching: F-matrix RANSAC (sparse) or 8 px row tolerance + "use all matches if filter too strict" (dense), no disparity sign check | wrong correspondences accepted; for a rectified pair the epipolar geometry is known exactly |
| 6 | Keypoint positions from coarse ORB octaves used directly | disparity error of several px → large depth error (∝ Z²) |
| 7 | PnP: 10 px RANSAC threshold, no refinement, no motion prior, no sanity check | noisy poses |
| 8 | Camera pose → body: `T_BS⁻¹` applied to a world point; rectification rotation `R1` ignored | wrong frame for comparison with ground truth |
| 9 | GT: nearest-next sample via `searchsorted`, position only, frames outside GT range clipped | time misalignment; now linear + SLERP interpolation, out-of-range frames skipped |
| 10 | Evaluation only with Sim(3) Umeyama | hides scale errors (stereo is metric) → now SE(3) ATE + scale reported separately + anchored drift + RPE |
| 11 | `sparse_f.py`: `start_frame`, `pts2`, `pts3`, `gray0_l` undefined; `rmse` undefined on short runs; compare mode unpacked 3 of 4 return values; `plot=False` crashed | the "sparse" module never ran |
| 12 | `START_FRAME` from the GUI was ignored by the VO | VO and detection showed different frames |
| 13 | Detector training: boxes left in 752×480 px on a 320×320 tensor | the model learned wrong boxes; the GUI then rescaled them again |
| 14 | Extra ImageNet `Normalize` before a model that normalises internally | double normalisation, weaker transfer from pretrained weights |
| 15 | Detection metric: centre distance / 10 000 px "diagonal", threshold 100 | every prediction matched every GT; precision computed per frame, not cumulatively (tp=236 → precision 0) |
| 16 | Detection scored on its own training frames; unlabelled frames counted as empty GT | leakage, meaningless score. Now IoU≥0.5, AP@0.5, time-based held-out validation split |
| 17 | (dataset) the official V2_01 calibration leaves a constant 1.7–1.9 px vertical offset after rectification (≈0.24° relative pitch; V1_01/MH_02: 0.00 px) | 60 % of correct matches rejected, ATE 0.92 m. Now estimated online from the first second (`Rectifier.self_calibrate`) → ATE 0.10 m |
| 18 | (found while testing) coasting on lost frames recomputed the velocity from itself; rotations never re-orthonormalised; NaN poses passed the jump check | pose overflow to 1e+200 on V2_01's landing |
