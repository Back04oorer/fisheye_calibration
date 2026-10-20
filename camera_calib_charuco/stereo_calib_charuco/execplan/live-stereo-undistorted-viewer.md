# Add fisheye model dispatch to the live stereo viewer

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must stay current as work proceeds. The repository has no local `PLANS.md`; this plan follows the ExecPlan methodology loaded by the implementing agent.

## Purpose / Big Picture

After this change, `python3 view_stereo_undistorted.py` can load the stereo parameters produced by the current fisheye `stereo_calib.py` and display geometrically correct undistorted or rectified video. The same command continues to accept historical pinhole stereo files. The program discovers the saved camera model automatically, prints it at startup, and uses the matching OpenCV API for image maps and calibration-board validation.

## Progress

- [x] (2026-07-27 20:45+08) Implement the original pinhole stereo viewer, full-resolution display, cached maps, saving, and keyboard controls.
- [x] (2026-07-27 22:03+08) Add live marker recognition, rectified vertical-error metrics, transformed outlines, and status overlay.
- [x] (2026-07-29 17:27+08) Confirm that the viewer ignores the new NPZ `model` field and always calls pinhole OpenCV functions.
- [x] (2026-07-29 17:28+08) Compare hard-coded fisheye, manual model selection, and automatic NPZ model dispatch; select automatic dispatch with legacy fallback.
- [x] (2026-07-29 17:29+08) Add model loading, model-specific distortion validation, fisheye map construction, fisheye point transformation, and console model output.
- [x] (2026-07-29 17:30+08) Expand the isolated viewer suite from 14 to 19 tests, including model reporting before camera-open failure, and fix the expected per-marker fisheye point-call count.
- [x] (2026-07-29 17:31+08) Process all 50 saved pairs with the new fisheye branch and inspect a full-size annotated preview.
- [x] (2026-07-29 17:32+08) Pass all 40 repository tests and verify the missing-camera error after the viewer reports the selected fisheye model.
- [ ] Run the live camera and keyboard integration check on a host exposing `/dev/video0`.

## Surprises & Discoveries

- Observation: A four-element distortion vector does not identify a fisheye model by itself.
  Evidence: OpenCV pinhole APIs accept four Brown coefficients, so the viewer must read the explicit NPZ `model` field rather than infer the model from `D1.size`.
- Observation: Treating the new fisheye calibration as pinhole can still produce plausible-looking images and some small errors near the optical center.
  Evidence: Across 50 saved pairs, the incorrect pinhole branch reports a per-pair vertical P95 median of 1.501590 px but a 95th percentile of 6.950689 px and maximum of 8.969370 px.
- Observation: Correct fisheye dispatch sharply improves the full-field validation.
  Evidence: The fisheye branch reports a per-pair P95 median of 0.588459 px, a 95th percentile of 1.214319 px, and a maximum of 2.491318 px. Status totals are 45 PASS, 1 WARNING, 0 FAIL, and 4 INSUFFICIENT.
- Observation: `transform_board_detection` transforms each marker quadrilateral separately.
  Evidence: A synthetic detection containing two marker IDs calls `cv2.fisheye.undistortPoints` twice, not once; the test now documents that contract.
- Observation: The current execution environment still exposes no confirmed live USB camera path for GUI and keyboard validation.
  Evidence: Offline image processing exercises parameter loading, model dispatch, map construction, remapping, detection, annotation, and metrics without claiming live-device behavior.

## Decision Log

- Decision: Read `model` from the stereo NPZ and canonicalize supported names to `fisheye` or `pinhole`.
  Rationale: The saved model is the only reliable way to choose the projection equations. Automatic selection removes a user-error path.
  Date/Author: 2026-07-29 / Codex
- Decision: Treat a missing `model` field as pinhole.
  Rationale: Stereo files created before the fisheye update used `cv2.stereoCalibrate` and have no model field. This fallback preserves those files.
  Date/Author: 2026-07-29 / Codex
- Decision: Accept `pinhole`, `rational`, `brown`, and `brown5` as pinhole aliases; reject all other labels.
  Rationale: These labels use OpenCV's pinhole distortion API. Rejecting unknown names prevents silent processing with the wrong equations.
  Date/Author: 2026-07-29 / Codex
- Decision: Require exactly four distortion coefficients for fisheye and at least four for pinhole.
  Rationale: OpenCV's fisheye model has exactly four coefficients, while pinhole models may store 4, 5, 8, 12, or 14 coefficients.
  Date/Author: 2026-07-29 / Codex
- Decision: Dispatch both remap construction and point transformation through the same model.
  Rationale: Correcting images with one projection model and validating marker coordinates with another would produce misleading overlays and quality metrics.
  Date/Author: 2026-07-29 / Codex
- Decision: Keep the existing runtime controls, dimensions, and validation thresholds unchanged.
  Rationale: The requested change concerns camera-model correctness. The established interaction and quality semantics remain useful.
  Date/Author: 2026-07-29 / Codex

## Outcomes & Retrospective

The fisheye compatibility implementation is complete. The viewer loads `model="fisheye"`, preserves `D1` and `D2` as 4 by 1 arrays, builds all four maps with `cv2.fisheye.initUndistortRectifyMap`, and transforms board points with `cv2.fisheye.undistortPoints`. Historical files without `model` remain on the pinhole path. All 40 repository tests pass. Offline processing of all 50 pairs produces 45 PASS, 1 WARNING, 0 FAIL, and 4 INSUFFICIENT results, and the inspected full-size preview is horizontally aligned. A direct startup attempt prints the chosen parameter file and `相机模型: fisheye`, then exits clearly because this environment has no `/dev/video0`. Live GUI and keyboard validation remains the only hardware-dependent gap.

## Context and Orientation

`stereo_calib.py` writes timestamped `stereo_calib_<timestamp>/results/stereo_params_<timestamp>.npz` files. Current files contain `model="fisheye"`, `image_size`, camera matrices `K1/K2`, four-coefficient distortions `D1/D2`, rectification rotations `R1/R2`, and projection matrices `P1/P2`. Historical files omit `model` and contain pinhole distortion vectors.

`view_stereo_undistorted.py` loads those fields into `StereoParameters`. It precomputes maps for an `undistort` mode and a `rectify` mode. It also detects ArUco markers in raw frames and transforms their centers and outlines into corrected coordinates. Image remapping and point transformation must use the same camera model.

The model branch and its failure paths are:

    [stereo NPZ]
          |
          v
    [read model] -- unknown/malformed --> [exit 1 before camera open]
          |
     +----+----------------------+
     |                           |
  fisheye                  pinhole or absent
     |                           |
 [D size == 4]             [D size >= 4]
     |                           |
     v                           v
 [fisheye maps/points]     [pinhole maps/points]
     |                           |
     +-------------+-------------+
                   v
       [remap + marker validation]
                   |
                   v
       [full-size preview and save]

## Plan of Work

Extend the frozen `StereoParameters` dataclass in `view_stereo_undistorted.py` with a canonical `model` string that defaults to `pinhole` for source compatibility. Add `_read_camera_model(data)` to decode a scalar NPZ model field, map accepted pinhole aliases to `pinhole`, preserve `fisheye`, reject unknown labels, and return `pinhole` when the key is absent.

In `load_stereo_parameters`, read the model before validating distortions. Fisheye arrays must contain exactly four finite coefficients and must be reshaped to `(4, 1)`. Pinhole arrays must contain at least four coefficients and must be reshaped to `(1, N)`.

Add a model argument to `_build_view_maps`. Call `cv2.fisheye.initUndistortRectifyMap` for fisheye and `cv2.initUndistortRectifyMap` for pinhole. Pass `parameters.model` for all left/right undistort and rectify maps.

Add the same model argument to `_transform_points`. Call `cv2.fisheye.undistortPoints` for fisheye and `cv2.undistortPoints` for pinhole. Route transformed marker outlines and rectified marker-center validation through this helper. Print the canonical model after loading parameters.

Update `tests/test_view_stereo_undistorted.py` so its synthetic parameter factory can produce either model. Test explicit pinhole round trips, missing-model legacy fallback, fisheye four-coefficient loading, unsupported labels, bad fisheye coefficient counts, both models in both remap modes, and explicit calls to fisheye map and point functions.

## Concrete Steps

Work from `/home/tsj/softwore/stereo_calib_charuco`.

Run syntax validation and the isolated viewer suite:

    python3 -m py_compile view_stereo_undistorted.py
    python3 -m unittest tests.test_view_stereo_undistorted -v

Run the full repository suite:

    python3 -m unittest discover -s tests -v

Run the offline production-data check with the importable viewer helpers. Load the newest NPZ, assert that `parameters.model == "fisheye"`, build both map sets, process all exact-name pairs from `charuco_left/` and `charuco_right/`, and write one annotated 2560 by 720 preview under `/tmp`.

On a host with the side-by-side USB camera, run:

    python3 view_stereo_undistorted.py

The console should print the selected NPZ, `相机模型: fisheye`, and the negotiated 2560 by 720 stream before the GUI loop begins.

## Validation and Acceptance

The implementation passes when compilation succeeds, all repository tests pass, and the current stereo NPZ loads as fisheye with `D1.shape == D2.shape == (4, 1)`. Both `undistort` and `rectify` map sets must contain 720 by 1280 maps. A corrected combined preview must remain 720 by 2560 without resizing.

Offline validation should produce no `FAIL` frames with the current 50 pairs. Its finite per-pair vertical-error P95 distribution should have a median below 1 px and a 95th percentile below 2 px. The inspected preview must show corresponding board features on the same horizontal guide lines.

The coverage map is:

    [T3] explicit fisheye and pinhole model loading
    [T3] legacy missing-model pinhole fallback
    [T3] unknown model and coefficient-count errors
    [T3] pinhole and fisheye map construction
    [T3] pinhole and fisheye point transformation
    [T3] both correction modes and full-size remapping
    [T3] marker matching, visibility filtering, and status thresholds
    [E2E] current 50-pair fisheye offline validation and preview inspection
    [GAP] live USB capture, GUI refresh, and keyboard controls

Seven deterministic paths and the offline end-to-end path have coverage. The live path remains a hardware-only gap.

## Idempotence and Recovery

Parameter loading and offline checks do not modify calibration files or source images. Map creation can be repeated safely. Viewer saves use timestamped names and do not overwrite existing captures. If a model validation error occurs, select a compatible NPZ; do not relabel coefficients without recalibrating. Every exit after camera creation releases the device and destroys OpenCV windows.

## Artifacts and Notes

Current offline evidence:

    parameters: stereo_calib_20260729_172611/results/stereo_params_20260729_172611.npz
    loaded model: fisheye
    distortion shapes: D1=(4,1), D2=(4,1)
    rectify map shapes: left=(720,1280), right=(720,1280)
    statuses: 45 PASS, 1 WARNING, 0 FAIL, 4 INSUFFICIENT
    pair P95 median / distribution P95 / max: 0.588459 / 1.214319 / 2.491318 px
    inspected pair: stereo_022.jpg, 21 usable markers, 0.718597 px P95, PASS
    preview: /tmp/viewer_fisheye_rectified_stereo_022.jpg, 2560x720
    viewer tests: 19 passed
    all repository tests: 40 passed
    no-camera startup: reports model=fisheye, then exits 1 with 无法打开 USB 相机 0

For comparison, processing the same fisheye coefficients with pinhole functions produces a per-pair P95 distribution of 1.501590 / 6.950689 / 8.969370 px. This difference proves that model dispatch changes user-visible geometry rather than only metadata handling.

## Interfaces and Dependencies

Use Python's standard library, NumPy, the installed OpenCV 4.9.0 package, and `setup_detector()` from `calib_charuco.py`. Do not add dependencies.

`StereoParameters` must expose `model`, `image_size`, `K1`, `D1`, `K2`, `D2`, `R1`, `R2`, `P1`, and `P2`. Keep `load_stereo_parameters(path)`, `build_remap_sets(parameters)`, `transform_board_detection(...)`, `calculate_validation_metrics(...)`, and `run_viewer(args)` importable for tests and offline checks.

Plan revision note: 2026-07-29 17:31+08, replaced the pinhole-only viewer plan with the implemented automatic fisheye/pinhole dispatch, compatibility rules, test coverage, and current 50-pair evidence.

Plan revision note: 2026-07-29 17:32+08, recorded the 40-test regression result, startup model output, clean missing-camera error, and completed software outcome while retaining the live-hardware gap.
