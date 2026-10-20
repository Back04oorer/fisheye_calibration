# Fisheye Stereo Extrinsic Calibration Design

## Purpose

`stereo_calib.py` calculates the fixed transform from the left fisheye camera
to the right fisheye camera. It consumes paired images from `charuco_left/` and
`charuco_right/`, automatically loads the latest four-parameter fisheye
intrinsics, and produces stereo geometry, quality metrics, and rectified
previews. The nominal 60 mm baseline remains a validation target; the optimizer
does not force the translation to that value.

## Evaluated Approaches

Three approaches were measured against the current 50 stereo pairs.

The former marker-PnP plus pinhole workflow detected too few useful pairs and
fed projected chessboard guesses into `cv2.stereoCalibrate`. It retained 12
pairs, reported 9.802 px RMS, and estimated a false 96.154 mm baseline. This
approach is rejected.

Directly intersecting detected ChArUco IDs removes the projected-point error,
but fitting those points with the rational pinhole stereo model still produced
about 4.228 px RMS and a 92.93 mm baseline. This approach is also rejected
because the near-180-degree lenses require the fisheye projection model.

The selected approach intersects direct ChArUco corner IDs and calls
`cv2.fisheye.stereoCalibrate` with both fisheye intrinsics fixed. The current
implementation retains 43 pairs after quality filtering, reaches 0.2155 px RMS,
and estimates a 60.472 mm baseline.

## Data Flow

This diagram shows the accepted path and the input and quality rejection paths.

```text
[paired images] + [fisheye_params_*.npz]
                 |
                 v
       [multiscale ChArUco detection] -- unreadable/weak --> [reject with reason]
                 |
                 v
          [intersect corner IDs] -------- fewer than 12 --> [reject with reason]
                 |
                 v
 [select 12 spatially distributed points per pair]
                 |
                 v
 [cv2.fisheye.stereoCalibrate, fixed intrinsics]
                 |
                 v
 [fisheye rectify + per-pair vertical P95]
                 |                         |
                 |              P95 outlier, retry <= 2
                 |                         v
                 |                    [reject pair]
                 v
 [NPZ/YAML/report + fisheye rectified previews]
```

OpenCV 4.9's Python binding requires the same point count in every stereo view.
The script therefore uses deterministic farthest-point sampling to select 12
common corners from each pair. Coordinates from both images participate in the
distance calculation, so the selected subset covers the visible board instead
of clustering in one region.

## Calibration and Filtering

The script loads only `fisheye_params_*.npz` by default. Each distortion vector
must contain exactly four finite coefficients, and a stored model name must be
`fisheye`. It passes the saved camera matrices and distortion vectors to
`cv2.fisheye.stereoCalibrate` with `CALIB_FIX_INTRINSIC`, which limits the fit to
the stereo rotation and translation.

After each fit, `cv2.fisheye.stereoRectify` and
`cv2.fisheye.undistortPoints` map all matched observations to rectified image
coordinates. The script scores each pair by the 95th percentile of its absolute
vertical disparity. It removes scores above the larger of 1.0 px and
`median + 2.5 * 1.4826 * MAD`, retries at most twice, and never reduces the data
set below `--min-pairs`.

The result passes when RMS is below 1.0 px, median vertical disparity is below
0.5 px, P95 vertical disparity is below 2.0 px, and the estimated baseline is
within 5% of 60 mm. Warning and failure results are still written for diagnosis.

## Outputs

Each run creates `stereo_calib_<timestamp>/` with a timestamped log, NPZ and
OpenCV YAML parameter files, a text report, and representative rectified image
pairs. The parameter files declare `model="fisheye"` and store `K1`, `D1`, `K2`,
`D2`, `R`, `T`, `E`, `R1`, `R2`, `P1`, `P2`, `Q`, metrics, and accepted and
rejected pair names. Full-image rectangles are stored as `roi1` and `roi2`
because the fisheye rectification API does not return pinhole-style valid ROIs.

For compatibility, the files retain an `F` field computed from `E` and the two
camera matrices. The report and NPZ mark this matrix as approximate: distorted
fisheye pixels do not obey one global 3 by 3 fundamental matrix. Consumers
should use the fisheye model and rectification matrices for image processing.

