# Robust ChArUco Intrinsic Calibration Design

## Purpose

Update `calib_charuco.py` so batch calibration succeeds on partial, backlit, and strongly distorted ChArUco images. The script must preserve the detector helpers imported by `capture_stereo.py`, `stereo_calib.py`, and `view_stereo_undistorted.py`.

## Design

The script will set `CharucoBoard.setLegacyPattern(True)` because the physical 11 by 8 board places marker ID 0 in the top-left square. This makes OpenCV's board topology agree with the existing custom `marker_3d` coordinates.

Image calibration will use observed ChArUco corners instead of projecting all 70 chessboard corners. A new detector will evaluate the raw grayscale image and two CLAHE variants at 1x, 1.5x, and 2x scale. It will select the candidate with the most ChArUco corners, rescale the coordinates, discard non-finite or out-of-bounds points, and refine only valid points on the original grayscale image. Frames with fewer than 12 corners will be recorded as rejected.

The calibration stage will fit OpenCV's four-parameter fisheye model, remove isolated corner residuals above a configurable pixel threshold, and refit until stable. It will also fit the rational pinhole model on the same observations. The fisheye model is the recommended representation for this lens. The rational model remains the default `params_<timestamp>.npz` output because the current stereo pipeline calls pinhole OpenCV APIs. Fisheye files will use the `fisheye_params_` prefix so `stereo_calib.find_latest_params()` cannot select them accidentally.

After saving, the command will print a structured terminal summary. It will show the accepted and rejected image counts, retained and removed corner counts, fisheye and rational global RMS values, per-view median/P95/maximum RMS values, both camera matrices and distortion vectors, a plain-language accuracy grade, the Brown-model diagnostic, and exact artifact paths. The CSV report remains the source for individual image rows; the terminal must not require the user to open a file to see the final intrinsics.

The source flow and its observable failure paths are:

```text
[image files or captured frames]
              |
              v
 [raw/CLAHE x 1x/1.5x/2x detection]
              |
       too few corners ----------> [rejected_images.csv]
              |
              v
 [bounds check + original-image subpixel]
              |
              v
 [fisheye fit] --> [point outlier removal] --> [stable refit]
              |
              +--> [fisheye_params_* recommended]
              |
              +--> [params_* rational compatibility]
```

Image mode will run without OpenCV windows. USB and IP modes will retain interactive save and exit controls, then pass saved frames through the same calibration pipeline. Output directories will remain timestamped and explicit output paths will never be overwritten.

## Verification

Unit tests will verify the legacy marker layout, safe subpixel filtering, multiscale ChArUco detection on a rendered board, output naming, argument validation, and the presence of matrices and precision statistics in the terminal summary. The full suite must continue to pass for modules that import the detector helpers. An end-to-end run against `charuco_left/` must use at least 45 of 50 images and produce fisheye RMS below 0.30 pixels. `stereo_calib.find_latest_params("left")` must select the rational `params_` file rather than the fisheye alternative.
