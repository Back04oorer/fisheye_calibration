# Replace pinhole stereo calibration with a fisheye ChArUco workflow

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current as work proceeds. The repository does not contain its own `PLANS.md`; this plan follows the ExecPlan methodology loaded by the implementing agent.

## Purpose / Big Picture

After this change, a user can run `python3 stereo_calib.py` from `/home/tsj/softwore/stereo_calib_charuco` and obtain a physically plausible transform between the two fisheye cameras. The program loads the completed fisheye monocular calibrations, uses direct ChArUco correspondences from the 50 paired images, and writes parameters plus rectified previews. A successful run should report subpixel RMS, a baseline close to the nominal 60 mm, and low vertical disparity after rectification.

## Progress

- [x] (2026-07-29 16:45+08) Reproduce the old marker-PnP and pinhole result: 12 retained pairs, 9.802183 px RMS, and a false 96.154358 mm baseline.
- [x] (2026-07-29 16:52+08) Verify that direct multiscale ChArUco detection retains 48 of the current 50 pairs.
- [x] (2026-07-29 16:57+08) Compare direct rational-pinhole and fisheye stereo prototypes and select the fisheye model.
- [x] (2026-07-29 17:09+08) Replace `stereo_calib.py` with the fixed-intrinsic fisheye workflow.
- [x] (2026-07-29 17:10+08) Validate the new script against all 50 real image pairs.
- [x] (2026-07-29 17:11+08) Add fourteen deterministic tests for fisheye discovery, validation, matching, fixed point counts, filtering, serialization, and preview maps.
- [x] (2026-07-29 17:24+08) Run the complete 35-test repository suite and one final default-output calibration.
- [x] (2026-07-29 17:25+08) Inspect the final repository output and complete this plan's retrospective.

## Surprises & Discoveries

- Observation: The existing `calib_charuco.detect_charuco_robust()` now detects direct ChArUco corners reliably on almost every stereo image.
  Evidence: 48 of 50 pairs pass the default 12-corner per-side requirement. Only `stereo_019.jpg` and `stereo_021.jpg` fail because the left images provide 11 and 10 reliable corners.
- Observation: Correct correspondences alone do not make a pinhole stereo model suitable for these lenses.
  Evidence: A direct ChArUco rational-pinhole prototype still returned about 4.228 px RMS and a 92.93 mm baseline.
- Observation: OpenCV 4.9's Python `cv2.fisheye.stereoCalibrate` wrapper asserts that every view has the same point count, although the conceptual API accepts vectors of variable-length observations.
  Evidence: Variable counts triggered `fixedSize() ... Assertion failed`. Selecting 12 points per pair removed the assertion and produced a valid result.
- Observation: Per-pair mean vertical error can hide one or two severe point mismatches.
  Evidence: Scoring by each pair's P95 identifies five current outliers, including `stereo_012.jpg` at 15.8020 px, while the retained set's global P95 falls to 0.437361 px.
- Observation: The current real-data validation exceeds all quality targets.
  Evidence: The `/tmp/stereo_calib_fisheye_validation` run used 43 pairs, returned 0.215534 px RMS, estimated a 60.472006 mm baseline, and measured 0.116331/0.437361 px median/P95 vertical error.

## Decision Log

- Decision: Load `fisheye_params_*.npz` instead of the rational compatibility file `params_*.npz`.
  Rationale: The monocular calibration reports fisheye as the recommended model, and only that model produces valid stereo geometry for these near-180-degree lenses.
  Date/Author: 2026-07-29 / Codex
- Decision: Intersect direct ChArUco corner IDs instead of projecting chessboard corners from marker-based PnP poses.
  Rationale: Direct IDs provide measured left/right correspondences and remove the large projection error that caused the old 9.8 px stereo fit.
  Date/Author: 2026-07-29 / Codex
- Decision: Select exactly 12 spatially distributed common corners from every pair.
  Rationale: OpenCV 4.9's Python fisheye binding requires fixed-size per-view inputs. Farthest-point sampling preserves board coverage while satisfying that binding constraint.
  Date/Author: 2026-07-29 / Codex
- Decision: Keep both fisheye intrinsics fixed with `cv2.fisheye.CALIB_FIX_INTRINSIC`.
  Rationale: The user has already completed left and right monocular calibration. Fixing intrinsics prevents the stereo optimizer from hiding bad correspondences by changing the lens models.
  Date/Author: 2026-07-29 / Codex
- Decision: Use each pair's rectified vertical-error P95 as its outlier score.
  Rationale: P95 catches sparse severe mismatches that a mean score can dilute and directly measures the horizontal-epipolar behavior required after rectification.
  Date/Author: 2026-07-29 / Codex
- Decision: Preserve `F` only as a labeled compatibility approximation.
  Rationale: A single 3 by 3 fundamental matrix does not describe distorted fisheye pixels. `E`, the fisheye intrinsics, and the rectification matrices remain valid outputs.
  Date/Author: 2026-07-29 / Codex

## Outcomes & Retrospective

The implementation is complete. `python3 -m py_compile stereo_calib.py` succeeds, all 35 repository tests pass, and the unmodified command `python3 stereo_calib.py` creates `stereo_calib_20260729_172404/`. The selected fisheye pipeline corrects both failure signals from the former implementation: RMS falls from 9.802183 px to 0.215534 px, and the baseline falls from a false 96.154358 mm to 60.472006 mm. Direct ChArUco detection increases usable input from 13 pre-filter pairs to 48; P95 filtering retains 43 consistent pairs. Six generated previews decode at 2560 by 720, and visual inspection confirms horizontal alignment. No requested work remains in this plan.

## Context and Orientation

`calib_charuco.py` defines the printed board, multiscale ChArUco detection, and the monocular fisheye calibration. Its completed outputs live under `charuco_left_calib_20260729_164020/results/` and `charuco_right_calib_20260729_163950/results/`. The files named `fisheye_params_*.npz` contain a 3 by 3 camera matrix in `mtx`, four distortion coefficients in `dist`, and `model="fisheye"`.

`stereo_calib.py` pairs exact filenames from `charuco_left/` and `charuco_right/`. A ChArUco corner ID identifies one physical chessboard intersection, so matching the same ID between the left and right images creates a stereo correspondence. An extrinsic transform uses OpenCV's convention `X_right = R * X_left + T`; the translation is measured in millimeters because the board geometry is defined in millimeters.

The implemented data and rejection paths are:

    [left/right images] + [left/right fisheye intrinsics]
                        |
                        v
              [direct ChArUco detection] -- weak/unreadable --> [reject]
                        |
                        v
                  [common corner IDs] ------- fewer than 12 --> [reject]
                        |
                        v
             [12 distributed points per pair]
                        |
                        v
              [fixed-intrinsic fisheye fit]
                        |
                        v
              [rectify and score pair P95] -- outlier --> [remove and retry]
                        |
                        v
               [parameters, report, previews]

## Plan of Work

In `stereo_calib.py`, replace imports of marker-only detection with `setup_charuco_detector` and `detect_charuco_robust`. Change automatic parameter discovery to match `fisheye_params_*.npz`. Reject any intrinsic model whose stored name is not `fisheye` or whose distortion vector does not contain exactly four coefficients.

Build each observation by intersecting direct ChArUco IDs. Add deterministic farthest-point sampling over normalized coordinates from both views, and select `--points-per-pair` points, defaulting to 12. Validate every observation before passing it to OpenCV so object, left-image, and right-image arrays all have identical fixed lengths and fisheye-compatible shapes.

Replace `cv2.stereoCalibrate`, `cv2.stereoRectify`, `cv2.undistortPoints`, and `cv2.initUndistortRectifyMap` with their `cv2.fisheye` equivalents. Keep intrinsics fixed. Derive `E=[T]xR`, retain a clearly labeled approximate `F` for file compatibility, and use full-image ROI tuples because fisheye rectification does not return pinhole-style ROIs.

Filter image pairs by rectified vertical-error P95. Serialize `model="fisheye"`, all geometry, quality metrics, and rejection reasons to NPZ, YAML, and a text report. Generate representative previews with fisheye rectification maps and horizontal guide lines.

In `tests/test_stereo_calib.py`, cover exact-name pairing, fisheye-only discovery, four-coefficient validation, spatial point selection, direct ID intersection, fixed input counts, P95 filtering, fisheye model serialization, fisheye preview map construction, quality boundaries, and output overwrite protection.

## Concrete Steps

Work from `/home/tsj/softwore/stereo_calib_charuco`.

Run syntax validation:

    python3 -m py_compile stereo_calib.py

Run all repository tests:

    python3 -m unittest discover -s tests -v

Run the real calibration with defaults:

    python3 stereo_calib.py

Inspect the newest output directory and report:

    find stereo_calib_<timestamp> -maxdepth 3 -type f -print | sort
    sed -n '1,180p' stereo_calib_<timestamp>/results/stereo_calibration_report.txt

The command may be repeated safely because default runs create a new timestamped directory.

## Validation and Acceptance

The script passes when syntax validation succeeds, all repository tests pass, and the real run finds 50 filename pairs, retains at least 40 pairs, and exits with status zero. The expected quality limits are RMS below 1.0 px, median vertical disparity below 0.5 px, P95 vertical disparity below 2.0 px, and baseline deviation within 5% of 60 mm.

The output NPZ must declare `model="fisheye"`. It must contain finite matrices with shapes `K1=(3,3)`, `D1=(4,1)`, `K2=(3,3)`, `D2=(4,1)`, `R=(3,3)`, `T=(3,1)`, `E=(3,3)`, `R1=(3,3)`, `R2=(3,3)`, `P1=(3,4)`, `P2=(3,4)`, and `Q=(4,4)`. Six default previews should decode as 2560 by 720 color images, and corresponding features should follow the same horizontal guide lines.

The test coverage map is:

    [T3] exact filename pairing and missing partners
    [T3] fisheye-only parameter discovery and model validation
    [T3] spatial fixed-count point selection
    [T3] direct ChArUco ID intersection
    [T3] OpenCV fixed-size input validation
    [T3] P95-based outlier removal
    [T3] fisheye model serialization and approximate-F marker
    [T3] fisheye preview map construction
    [E2E] 50-pair detection, fitting, filtering, output, and visual inspection

Eight deterministic paths have T3 coverage. The end-to-end path uses the repository's real calibration images.

## Idempotence and Recovery

Default runs never overwrite prior results. An explicit `--output` path is accepted only when it does not exist. Source images and monocular calibration directories remain unchanged. If detection or optimization fails, read the logged rejection or OpenCV error, correct the input or option, and rerun with a new output directory. Failed historical output directories remain useful diagnostic evidence and do not need deletion.

## Artifacts and Notes

The final default run produced:

    output: stereo_calib_20260729_172404
    tests: 35 passed
    input pairs: 50
    direct detections retained: 48
    final pairs after P95 filtering: 43
    stereo RMS: 0.215534 px
    T: [-60.403137, -0.117387, 2.882830] mm
    estimated baseline: 60.472006 mm
    vertical median / P95: 0.116331 / 0.437361 px
    overall quality: PASS
    rectified previews: 6 at 2560x720

All saved matrices are finite and have the required shapes. Visual inspection of `rectified_stereo_022.jpg` confirms horizontal alignment across the board and room features.

## Interfaces and Dependencies

Use Python's standard library, NumPy, and the installed OpenCV 4.9.0 package. `stereo_calib.py` imports `create_charuco_board`, `cv_imread`, `setup_charuco_detector`, and `detect_charuco_robust` from `calib_charuco.py`.

The module exposes `find_latest_params(side, search_root)`, `load_intrinsics(path)`, `select_spatially_distributed_indices(left_points, right_points, count, image_size)`, `match_common_charuco_points(...)`, `prepare_fisheye_inputs(observations)`, `calibrate_once(...)`, `calibrate_with_filtering(...)`, `save_rectified_previews(...)`, and `run_calibration(args)` for testing and reuse.

Plan revision note: 2026-07-29 17:12+08, replaced the obsolete marker-PnP and pinhole plan with the implemented direct-ChArUco fisheye design, prototype evidence, focused test results, and remaining final validation work.

Plan revision note: 2026-07-29 17:25+08, recorded the successful 35-test regression run, final default calibration output, artifact inspection, and completed outcome.
