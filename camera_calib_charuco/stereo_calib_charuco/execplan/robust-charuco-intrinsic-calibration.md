# Make ChArUco intrinsic calibration safe and accurate

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. The repository does not contain its own `PLANS.md`; this plan follows the ExecPlan methodology loaded by the implementing agent.

## Purpose / Big Picture

After this change, a user can run `python3 calib_charuco.py --images charuco_left` and obtain stable intrinsic parameters without a GUI or an OpenCV boundary crash. The program will detect actual visible ChArUco corners under strong lens distortion, low contrast, partial board visibility, and small marker scale. It will save a recommended fisheye model and a rational pinhole model compatible with the existing stereo calibration code.

## Progress

- [x] (2026-07-29 16:00+08) Reproduce the current failure and compare Brown, rational, and fisheye models on all 50 left images.
- [x] (2026-07-29 16:00+08) Select a backward-compatible dual-model design.
- [x] (2026-07-29 16:14+08) Refactor `calib_charuco.py` while preserving imported detector interfaces.
- [x] (2026-07-29 16:15+08) Add regression tests for board topology, safe corner refinement, multiscale detection, and output naming.
- [x] (2026-07-29 16:16+08) Run the full unit suite and a real `charuco_left` end-to-end calibration.
- [x] (2026-07-29 16:17+08) Record final artifacts, metrics, and remaining limitations.
- [x] (2026-07-29 16:30+08) Restore complete terminal output for matrices, distortion coefficients, precision statistics, and parameter paths after user feedback.

## Surprises & Discoveries

- Observation: The physical board uses OpenCV's legacy pattern, while `create_charuco_board()` currently creates the modern pattern.
  Evidence: with `setLegacyPattern(True)`, board marker ID 0 matches the custom top-left `marker_3d[0]`, and `CharucoDetector` finds 63 corners in `stereo_049.jpg`; without it, the detector finds zero ChArUco corners.
- Observation: The current script crashes when a projected corner lies outside the image.
  Evidence: `cornerSubPix` raised `Rect(0, 0, src.cols, src.rows).contains(cT)` while processing the thirty-first accepted frame.
- Observation: A five-parameter Brown model cannot represent this lens.
  Evidence: Brown RMS was about 1.205 pixels, while fisheye and rational fits were about 0.187 pixels after filtering seven isolated bad corners.
- Observation: CLAHE plus 2x detection recovers useful observations from distant and backlit frames.
  Evidence: 48 of 50 images supplied at least 12 corners, compared with 42 images using raw 1x detection.
- Observation: The repository's stereo topology test still expected an older 11 by 10 board.
  Evidence: the baseline suite failed because `build_marker_corner_map()` correctly returned 44 markers while the test expected 54; deriving the expected counts from `SQUARES_X` and `SQUARES_Y` restored consistency.
- Observation: Stable point filtering required one more iteration than the initial prototype.
  Evidence: the real script run removed 6, then 1, then 1 corner before the fourth fit removed none and reached 0.185326 pixel fisheye RMS.

## Decision Log

- Decision: Preserve `create_charuco_board`, `setup_detector`, `detect_markers_robust`, and `cv_imread` signatures used by other modules.
  Rationale: Capture, stereo calibration, and rectified-viewer code import these helpers directly.
  Date/Author: 2026-07-29 / Codex
- Decision: Make fisheye the recommended intrinsic model but keep rational parameters under the unqualified `params_` filename.
  Rationale: The lens behaves as a fisheye, but current stereo code uses pinhole APIs such as `stereoCalibrate` and `undistortPoints`.
  Date/Author: 2026-07-29 / Codex
- Decision: Detect visible ChArUco corners directly and remove the projected-all-corners path.
  Rationale: Direct observation prevents out-of-frame crashes and false snapping to marker edges or background texture.
  Date/Author: 2026-07-29 / Codex
- Decision: Use multiscale detection only during intrinsic calibration; improve marker helper fallback without changing its return contract.
  Rationale: Intrinsic calibration benefits from maximum coverage, while live capture and stereo matching need predictable existing interfaces.
  Date/Author: 2026-07-29 / Codex
- Decision: Print a structured result summary by default instead of only RMS values or the entire report file.
  Rationale: Users need immediate access to K, D, precision, model choice, and artifact paths without opening another file; detailed per-image rows remain in CSV.
  Date/Author: 2026-07-29 / Codex

## Outcomes & Retrospective

The implementation is complete. `calib_charuco.py` now runs image mode without a window, detects the physical legacy board at multiple scales and contrast levels, filters only in-bounds observed ChArUco corners, and saves separate fisheye and rational models. All 28 unit tests pass. The real 50-image left run accepted 48 images, retained 1,940 corners, reached 0.185326 pixel fisheye RMS and 0.185187 pixel rational RMS, and wrote loadable NPZ/YAML files plus reports and previews under `/tmp/charuco-left-script-test-XdWtKY/calibration`.

The two weak frames remain `stereo_019.jpg` and `stereo_021.jpg`; both contain fewer than 12 reliable corners after CLAHE and 2x detection. The ordinary Brown model remains unsuitable at 1.204360 pixel RMS. No source image was changed.

After user feedback, the command now prints complete fisheye and rational camera matrices, distortion coefficients, global RMS, per-view median/P95/maximum RMS, accuracy grades, diagnostic Brown RMS, and exact output paths. A regression test verifies that these fields remain visible.

## Context and Orientation

`calib_charuco.py` defines the board geometry and performs monocular calibration. `capture_stereo.py` imports its marker detector to decide whether a stereo frame is saveable. `stereo_calib.py` imports the same detector and board geometry to match jointly visible board regions between the left and right cameras. `view_stereo_undistorted.py` imports detector configuration for live validation. Therefore, helper signatures form a repository-wide compatibility boundary.

A ChArUco corner is an internal chessboard intersection whose identity comes from neighboring ArUco markers. The fisheye model maps the angle of an incoming ray to image radius with four distortion coefficients. The rational pinhole model extends ordinary radial distortion with a numerator and denominator and works with the current pinhole stereo APIs.

The changed data flow is:

    [sorted image paths]
             |
             v
    [read consistent image size] -- unreadable/mismatch --> [named rejection]
             |
             v
    [raw and CLAHE detection at 1x, 1.5x, 2x]
             |
      fewer than 12 corners ---------------------------> [named rejection]
             |
             v
    [rescale, bounds-check, refine on original grayscale]
             |
             v
    [fisheye calibration] -- residual > threshold --> [remove point, retry]
             |
             +--> [fisheye_params_* NPZ/YAML]
             |
             +--> [rational fit and params_* NPZ/YAML]

## Plan of Work

Refactor `calib_charuco.py` into importable helpers plus a command-line entry point. Add argument parsing for mode, image directory, explicit output directory, minimum corner count, point outlier threshold, preview count, and raw/corner image copies. Keep the existing configuration constants as defaults.

Set the board's legacy pattern immediately after construction. Improve `detect_markers_robust()` so it compares raw and CLAHE marker detections instead of trying enhancement only after total failure, rescales multiscale candidates when needed, and applies subpixel refinement only to points whose windows fit inside the image.

Add a ChArUco-specific observation detector that tries the three image variants at three scales. Store each observation's filename, object points, image points, corner IDs, detector variant, scale, marker count, and blur score. Reject weak frames with an explicit reason.

Fit fisheye parameters first. Calculate per-corner residuals, remove isolated points above the configured threshold while retaining at least the configured number of points in each frame, and repeat for a bounded number of iterations. Fit rational and ordinary Brown models on the final observation set. Save model metadata, per-view CSV metrics, rejected images, removed points, a concise report, annotated corner images, and rectilinear fisheye previews with an explicit cropped output camera matrix.

Add `tests/test_calib_charuco.py`. Update any outdated board-topology expectations in `tests/test_stereo_calib.py` to derive counts from the configured 11 by 8 board rather than an older board size.

## Concrete Steps

Work from `/home/tsj/softwore/stereo_calib_charuco`.

First compile and run unit tests:

    python3 -m py_compile calib_charuco.py
    python3 -m unittest discover -s tests -v

Then run a real calibration outside the repository output tree:

    output=$(mktemp -d /tmp/charuco-left-calib-XXXXXX)
    rmdir "$output"
    python3 calib_charuco.py --mode image --images charuco_left --output "$output" --no-raw-copies --preview-count 2

Expect the program to report 48 or more accepted images, fisheye RMS below 0.30 pixels, rational RMS below 0.30 pixels, and a successful output path. Inspect the report and parameter selection:

    sed -n '1,220p' "$output/results/calibration_report.txt"
    python3 -c 'from pathlib import Path; import stereo_calib; print(stereo_calib.find_latest_params("left", Path(".")))'

The repository-level auto-selection must resolve to a file named `params_<timestamp>.npz`, never `fisheye_params_<timestamp>.npz`.

## Validation and Acceptance

The implementation passes when all Python files compile, every unit test passes, and the real calibration exits with status zero without opening a window. The real run must accept at least 45 images, retain at least 1,800 ChArUco corners, and produce finite camera matrices. Fisheye and rational RMS must each remain below 0.30 pixels; Brown RMS should remain substantially higher and appear only as a diagnostic.

The output must contain `params_<timestamp>.npz`, `params_<timestamp>.yaml`, `fisheye_params_<timestamp>.npz`, `fisheye_params_<timestamp>.yaml`, `calibration_report.txt`, `per_view_metrics.csv`, `rejected_images.csv`, and `removed_corner_outliers.csv`. Loading the rational NPZ through `stereo_calib.load_intrinsics()` must succeed with a 3 by 3 camera matrix and at least eight distortion coefficients.

The test coverage map is:

    [T3] legacy board topology and custom marker geometry agree
    [T3] out-of-bounds candidate corners never reach cornerSubPix
    [T3] rendered legacy board is detected through the multiscale path
    [T3] explicit output directories refuse overwrite
    [T3] fisheye filename cannot shadow rational auto-selection
    [E2E] 50-image left calibration, artifact load, and RMS thresholds

Unit coverage targets 5/5 deterministic paths. The real image quality path remains an end-to-end validation because OpenCV detector behavior depends on actual image content.

## Idempotence and Recovery

Default runs create new timestamped directories. An explicit output directory must not already exist. Source images remain unchanged. If a run fails during image collection, the output may contain logs and rejected-image records but must not contain a misleading parameter file. Retry with a fresh output path after correcting the reported input problem.

## Artifacts and Notes

Baseline evidence before implementation:

    current script: crashes on an out-of-frame projected corner
    raw 1x ChArUco observations: 42/50 images
    multiscale and CLAHE observations: 48/50 images
    filtered points: 1,940
    fisheye RMS: 0.186922 px
    rational RMS: 0.186795 px
    Brown RMS: 1.204609 px

Final validation evidence:

    unit tests: 28 passed
    accepted images: 48/50
    retained corners: 1,940
    removed corner outliers: 8
    fisheye RMS: 0.185326 px
    rational RMS: 0.185187 px
    Brown RMS: 1.204360 px
    output: /tmp/charuco-left-script-test-XdWtKY/calibration

## Interfaces and Dependencies

Use Python's standard library, NumPy, and the installed OpenCV 4.9.0 package. Preserve these public functions and return contracts in `calib_charuco.py`:

    create_charuco_board() -> (board, chessboard_corners_3d, marker_3d)
    setup_detector() -> (aruco_detector, clahe, clahe_strong)
    detect_markers_robust(gray, detector, clahe, clahe_strong, marker_3d, board)
        -> (object_points, image_points, marker_corners, marker_ids)
    cv_imread(file_path) -> color image or None

Add importable helpers for argument parsing, safe subpixel refinement, ChArUco observation detection, model calibration, result saving, and orchestration. `main(argv=None)` must return a process status and `raise SystemExit(main())` must remain the command-line boundary.

Plan revision note: 2026-07-29, initial plan written after reproducing the crash and validating the dual-model approach on the current left image set.

Plan revision note: 2026-07-29 16:17+08, recorded the completed refactor, regression tests, real-data metrics, and final artifact paths.

Plan revision note: 2026-07-29 16:30+08, restored detailed console results after the first refactor made the terminal summary too terse.
