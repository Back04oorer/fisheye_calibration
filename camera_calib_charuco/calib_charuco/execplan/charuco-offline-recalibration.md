# Fix ChArUco corner refinement and recalibrate saved images

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must stay current as work proceeds. The
methodology comes from the external ExecPlan skill because this repository has no
checked-in `PLANS.md`.

## Purpose / Big Picture

The calibration script currently crashes when a projected chessboard corner falls outside
an image. Its initial calibration is also degraded by incompatible marker recovery and a
second oversized subpixel pass. After this change, the script can safely calibrate a partly
cropped board and can recalibrate an existing capture directory without changing its raw
images. Running `python3 calib_charuco.py --recalibrate-folder
charuco_calib_20260728_213552` will write timestamped logs and parameter files into that
capture directory.

## Progress

- [x] (2026-07-28 21:55 +08) Diagnosed the out-of-bounds projection and measured the
  corner-refinement regressions on all 25 saved images.
- [x] (2026-07-28 22:03 +08) Added the shared calibration pipeline, bounded chessboard
  refinement, timestamped reports, and the offline CLI.
- [x] (2026-07-28 22:04 +08) Compiled the script and exercised CLI help, a missing input
  directory, and the successful offline path.
- [x] (2026-07-28 22:04 +08) Recalibrated `charuco_calib_20260728_213552`, validated both
  parameter formats, and confirmed all 25 raw-image checksums.

## Surprises & Discoveries

- Observation: The saved board uses the hand-built `marker_3d` layout, while the default
  OpenCV `CharucoBoard` passed to `refineDetectedMarkers` uses the opposite square parity.
  Evidence: marker ID 0 spans x=3..27 mm in `marker_3d` but x=33..57 mm in the OpenCV
  board.
- Observation: Direct detector output produces an initial RMS of 0.899515 px; incompatible
  marker recovery raises it to 2.437393 px; the additional `(11,11)` subpixel pass raises
  it to 5.883257 px.
- Observation: Image `raw_001.jpg` projects two corners below the 480-pixel image boundary.
  Keeping points with a 3-pixel margin leaves 86 of 88 corners.
- Observation: The completed offline run matched the diagnostic baseline exactly.
  Evidence: the two chessboard iterations produced 0.396814 px and 0.396274 px; filtering
  retained 21 views and produced a final RMS of 0.327706 px.

## Decision Log

- Decision: Use direct `ArucoDetector.detectMarkers` output and retain its configured
  `CORNER_REFINE_SUBPIX` result.
  Rationale: It matches the physical marker model and measured better than recovery or a
  second marker subpixel pass.
  Date/Author: 2026-07-28 / Codex.
- Decision: Add `--recalibrate-folder CAPTURE_DIR` and write timestamped results into the
  existing `logs` and `results` directories.
  Rationale: The user selected this location, and the command can avoid copying or
  recompressing raw images.
  Date/Author: 2026-07-28 / Codex.
- Decision: Refine projected chessboard corners once with a `(3,3)` window and a matching
  3-pixel boundary margin.
  Rationale: Read-only trials produced 0.396274 px after two refinement passes, better than
  the `(5,5)` and `(5,5)` then `(3,3)` alternatives.
  Date/Author: 2026-07-28 / Codex.

## Outcomes & Retrospective

The script now rejects unsafe projected corners, avoids the two marker-refinement paths that
degraded the initial fit, and exposes a headless offline command. The requested dataset
completed without an OpenCV exception. It saved finite, matching NPZ and YAML parameters,
retained 21 views, and achieved 0.327706 px RMS. All 25 source-image checksums remained
unchanged. The only unexercised path is live USB capture because validation did not control
the user's camera hardware or display session.

## Context and Orientation

`calib_charuco.py` is the only executable source file. Its `main` function collects USB,
IP-camera, or local images, detects ArUco markers, performs an initial camera calibration,
projects all internal chessboard corners, refines those coordinates, filters high-error
views, and writes NPZ, YAML, report, and log files. A subpixel pass adjusts approximate
pixel coordinates to nearby image corners. It requires each starting coordinate to lie
inside the image.

The change preserves one common owner for calibration outputs while giving capture and
offline modes different owners for input images:

    [USB/IP capture] --owns new raw files--+
                                           |
                                           v
                                  [shared calibration]
                                           |
                                           v
    [existing raw_images] --read only----> [timestamped logs/results]
              |
              +-- invalid/missing/<5 images --> clear error, no calibration

## Plan of Work

First, update marker detection so it no longer passes the incompatible OpenCV board to
`refineDetectedMarkers` and no longer repeats marker subpixel refinement. Keep the CLAHE
fallback and the existing minimum marker count.

Second, update `refine_chessboard_corners` to reject failed poses, non-finite projections,
and coordinates within three pixels of an image edge. Apply every mask to both image and
object points, require six remaining corners, run one `(3,3)` subpixel pass, and validate
the returned correspondences again.

Third, extract the current post-capture stages into a shared function. Add an offline
loader that accepts a capture root containing `raw_images`, detects markers without a GUI,
checks consistent image sizes, and passes those correspondences to the shared function.
Add an `argparse` entry point named `--recalibrate-folder`. With no option, retain the
existing USB/IP/image collection behavior.

Finally, timestamp the report as well as NPZ and YAML files. Run the offline command on the
requested directory, inspect the log and saved matrices, and prove that all raw-image
checksums remain unchanged.

## Concrete Steps

Work from `/home/tsj/softwore/calib_charuco`.

    python3 -m py_compile calib_charuco.py
    python3 calib_charuco.py --help
    sha256sum charuco_calib_20260728_213552/raw_images/* > /tmp/charuco-before.sha256
    python3 calib_charuco.py --recalibrate-folder charuco_calib_20260728_213552
    sha256sum -c /tmp/charuco-before.sha256

Load the newest result files with Python and confirm that the camera matrix is 3x3, the
distortion vector contains finite values, and the NPZ and YAML values agree.

## Validation and Acceptance

Compilation and `--help` must exit zero. An invalid capture directory must exit with a
short error that names the missing requirement. The requested offline run must process 25
images without opening a display or raising an OpenCV exception. Its log must show about
86 chessboard corners for image 2, a final retained view count near 21, and RMS below 0.5
px. Each returned object-point array must have the same length as its image-point array.
All 25 raw-image checksums must pass after calibration.

## Idempotence and Recovery

The offline command is repeatable because every log, NPZ, YAML, and report filename uses a
new timestamp. It only reads `raw_images`. If a run stops midway, rerun the same command;
the earlier partial timestamped outputs may remain for diagnosis and do not affect the
next run.

## Artifacts and Notes

The diagnostic baseline is:

    direct marker RMS:       0.899515 px
    chess refinement pass 1: 0.396814 px
    chess refinement pass 2: 0.396274 px
    expected filtered final: 0.327706 px, 21 views

The completed run wrote:

    charuco_calib_20260728_213552/logs/calib_charuco_20260728_220247.log
    charuco_calib_20260728_213552/results/params_20260728_220247.npz
    charuco_calib_20260728_213552/results/params_20260728_220247.yaml
    charuco_calib_20260728_213552/results/calibration_report_20260728_220247.txt

## Interfaces and Dependencies

Keep Python 3, NumPy 1.24, and OpenCV 4.9 as the only runtime dependencies. Add the
standard-library `argparse` module. The public CLI gains
`--recalibrate-folder CAPTURE_DIR`; the capture directory must contain `raw_images`, and
the command writes only under its `logs` and `results` subdirectories.

Revision note (2026-07-28 21:55 +08): Created this ExecPlan from the accepted implementation
plan and the measured behavior of the saved dataset.

Revision note (2026-07-28 22:03 +08): Recorded completion of the code changes before
running validation so the progress list remains restartable.

Revision note (2026-07-28 22:04 +08): Closed the plan with end-to-end results, artifact
paths, checksum evidence, and the remaining live-hardware test gap.
