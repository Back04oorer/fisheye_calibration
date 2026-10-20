# Stereo Corrected Viewer Design

## Purpose

`view_stereo_undistorted.py` opens the side-by-side stereo camera, loads the
latest stereo calibration, and displays corrected left and right views without
resizing. It supports both the current four-parameter fisheye calibration and
older pinhole-compatible calibration files. The viewer also recognizes the
calibration board and reports rectified vertical disparity as a live external
calibration check.

## Camera Model Selection

Three compatibility approaches were considered. Switching the viewer entirely
to fisheye would correctly handle new results but break historical pinhole
files. Adding a required `--model` option would preserve both paths but allow a
user to select a model that disagrees with the saved coefficients. The selected
approach reads the model from the NPZ file and dispatches automatically.

New calibration files store `model="fisheye"` and must contain exactly four
finite coefficients in each of `D1` and `D2`. Files labeled `pinhole`,
`rational`, `brown`, or `brown5` use the pinhole path and require at least four
coefficients. Historical files without `model` default to pinhole because all
files produced before the fisheye stereo update used that API. Unknown model
names fail before the camera opens.

This diagram shows the model boundary shared by image remapping and validation
point transformation.

```text
[latest stereo NPZ]
          |
          v
 [read model + validate D]
          |
     +----+--------------------+
     |                         |
 model=fisheye          pinhole/legacy file
     |                         |
     v                         v
[cv2.fisheye maps]       [cv2 pinhole maps]
[fisheye points]         [pinhole points]
     |                         |
     +------------+------------+
                  v
       [remap + board validation]
                  |
                  v
       [full-size combined preview]
```

## Runtime Data Flow

The program builds both correction modes before opening the frame loop. Pure
undistortion uses identity rectification and the saved camera matrix as the
output projection. Stereo rectification uses `R1/P1` and `R2/P2`. For fisheye
files, both modes call `cv2.fisheye.initUndistortRectifyMap`; for pinhole files,
they call `cv2.initUndistortRectifyMap`.

Each 2560 by 720 frame splits into two 1280 by 720 views. Marker detection runs
on the raw images because the detector expects original pixels. The viewer
intersects marker IDs, transforms the marker centers and outlines with the same
camera model used for the image maps, removes points outside the corrected
views, and calculates absolute vertical disparity. The metric always uses the
rectified coordinate system, even when the display shows pure undistortion.

The display keeps the left image on the left and the right image on the right.
`m` toggles correction mode, `v` toggles board validation, `g` toggles horizontal
guides, `s` saves a clean corrected pair, and `Esc` or `q` exits. The console
prints the detected camera model at startup.

## Validation and Errors

Fewer than six visible common marker centers produces `INSUFFICIENT`. With six
or more centers, P95 vertical disparity below 2 px produces `PASS`, 2 through 3
px produces `WARNING`, and above 3 px produces `FAIL`.

The current fisheye parameter file improves the offline 50-pair validation
substantially. Treating it incorrectly as pinhole yields a per-pair P95 median
of 1.502 px and P95 of 6.951 px. Correct fisheye dispatch reduces those values
to 0.588 px and 1.214 px. Forty-five pairs pass, one warns, four have too few
visible markers, and none fail. The inspected `stereo_022.jpg` preview has 21
usable markers and 0.719 px P95 vertical error.

The program rejects missing files, malformed matrices, invalid image sizes,
unsupported models, wrong fisheye coefficient counts, wrong frame dimensions,
camera negotiation failures, failed reads, and failed image writes. Camera
creation remains inside a release-safe `try/finally` boundary.

