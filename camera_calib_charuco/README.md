# Guided fisheye stereo calibration

A local browser dashboard for side-by-side stereo cameras. It checks ChArUco detection live, guides board movements, captures steady new poses automatically, fits a fisheye model, and checks that frozen model on fresh images. Original images and results stay on this computer.

## Start

On this computer, run `camera-calibration` from any terminal, in any folder.
The launcher reopens the running GUI, or starts it in the background and opens your
browser. Choose **Open** beside your saved session to resume it, or **New session**.
Closing the browser leaves the GUI running so the command can reopen it.

The same launcher is available from the repository with `./launch_gui.py`.
The command is a symlink at `~/.local/bin/camera-calibration` pointing to that
script; `~/.local/bin` is already on your shell's `PATH`.
Startup logs are saved to `output/gui-8765.log`.

Run from this repository with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run stereo-calibrate gui --camera 0
```

Open **http://127.0.0.1:8765**. The session picker appears before the camera opens. Choose **New session**, **Open** to resume a saved session, or **Replay** to review saved images without the camera. Use **Sessions** in the header to switch later; opening the picker pauses active collection. New sessions begin with the current board/camera settings and receive their own folder when collection starts. **Load another session folder** accepts an existing session directory outside the default location. **Folder** opens a saved session in the desktop file manager without loading it. The page opens automatically unless `--no-browser` is supplied. Python 3.12+ is required; `uv.lock` records the tested dependencies. The old `.calib_venv` is not modified. Do not install multiple OpenCV distributions into the same environment.

`--camera 0` selects the first video device, which supplies **both eyes in one frame**. It does not select the left eye. A stable `/dev/v4l/by-id/...` device path can also be supplied. Close other camera applications before starting this one.

The default mode is **2560 × 720**, or 1280 × 720 per eye. Change width/height in the setup form or CLI before collection. For example, the connected Sunplus camera also advertises:

```bash
uv run stereo-calibrate gui --camera 0 --width 3840 --height 1200
```

Use the camera mode that the downstream application will actually use. Modes can differ in field of view as well as resolution; do not reuse calibration across modes without establishing their pixel mapping.

## Choose and measure the A4 board

Both supplied boards fit landscape A4, use `DICT_5X5_100`, and have legacy layout enabled. Counts below are **squares**, not inner corners.

| Preset | Squares | Square / marker length | Printed pattern | Inner corners |
| --- | --- | --- | --- | --- |
| `existing-20mm` | 11 × 8 | 20 / 15 mm | 220 × 160 mm | 70 |
| `a4-30mm` | 8 × 6 | 30 / 22.5 mm | 240 × 180 mm | 35 |

The first matches the layout of `calib.io_charuco_260x200_8x11_20_15_DICT_5X5.pdf`. The larger-square preset trades fewer corners for larger marker details. It is the next A4 option to try when the 20 mm pattern is too small at the working distance.

Use **Print this board** in the GUI, or generate both PDFs:

```bash
uv run stereo-calibrate board --preset existing-20mm
uv run stereo-calibrate board --preset a4-30mm
```

Files appear in `output/pdf/`. Print at **100% / actual size**, with no fit-to-page scaling. Measure the 100 mm ruler and several squares horizontally and vertically. Enter the measured square length in the GUI; the marker length scales with it. If the printer stretches the two axes differently, correct the printing rather than entering an average. Mount the sheet flat on rigid backing, keep glare off the pattern, and keep camera focus fixed.

Legacy layout describes the arrangement of black squares and marker IDs for boards generated before OpenCV's layout change. It must match the physical print. It is not an accuracy setting or a lens model. AprilGrid is not supported in this workflow.

### Check before collecting

The dashboard shows detection **before anything is saved**. **Live video** shows the newest camera image independently of detection. Counts below it show the latest measurement. Select **Detected frame** to see the exact measured image and its corner overlays; green points are corners with matching IDs in both cameras. Start becomes available with at least 12 shared, non-collinear corners. Watch that recognition stays stable as you move toward the image edges and tilt the board; a brief successful detection at the centre is not enough to establish usability everywhere.

In the live check on this setup, the 20 mm board resting at the approximately 56 cm working distance was not reliably detected. Its squares were about 15–16 pixels wide in the 2560 × 720 mode. The higher-resolution mode alone did not fix detection on the table. Bringing the board closer produced about 43–44 pixels per square and a sample with 60 shared corners at 3840 × 1200. This verifies layout compatibility, **not calibration accuracy**. The 30 mm board at tabletop distance still needs a physical live check. There is no universal pixel-size or sharpness cutoff in the software.

## Collect and know when it is finished

1. Match the printed preset, measured square length, camera mode and left/right order. Select **Apply settings**, then **Start guided capture**.
2. Move the printed-pattern centre anywhere inside the blue area in the **left preview** and follow the illustrated movement cue. Keep corners visible in both cameras. The blue area shows the existing placement tolerance; it is not necessary to align with its exact centre. Around a camera support or other obstruction, position the board beside it or closer to the lenses while keeping the camera rig fixed. Hold each new pose steady for at least 0.7 seconds (and three fresh observations). The app chooses the sharper stereo pair from that hold and rechecks its full-resolution corners before saving. Repeated stationary poses are not saved repeatedly. A detection gap over 0.75 seconds or a repeated/out-of-order timestamp resets the hold. Once **Captured** appears, move to the next pose; there is no need to hold the board during fitting.
3. Cover the nine positions, a closer distance, and four tilt directions. Tilt cues refer to the board's **printed** edges. Keep the stereo rig fixed. There is no image quota. A centre view and tilts about both axes permit an early numerical fit; this improves guidance but does not establish quality. If that seed cannot fit yet, collection continues as coverage grows.
4. Continue following the same movement guide until completion. Fitting and independent accuracy checks happen automatically within **Guided collection**; an area may be revisited with a new tilt or distance. Internally, check images are kept separate from the images used to fit that candidate. There is no separate validation task to start.
5. **Geometric checks passed** means pose coverage, training residuals, fresh per-eye reprojection and stereo alignment all pass. Download NPZ, YAML and the report. A failed check automatically starts a **Retry**: the guide names the failed measurements and requests a new training pose in that region (or another coverage round if the overall training fit failed). Change tilt or distance and hold still. The app captures a new training pose and refits, then re-evaluates **all saved check images** against that updated calibration. Areas that still pass keep their coverage; only failing, missing, or corrected areas needing a fresh confirmation receive new capture cues. A failed saved measurement remains in the report and blocks completion even if a later image passes. Old candidates and original check images are preserved, and check images never enter the fit.

Pause stops saving while the preview stays live. Undo marks the last training capture as excluded and preserves its original image. Camera errors pause capture and expose a reconnect control. Settings are fixed once a session begins; changing the physical board or camera mode requires a new session.

The movement illustration uses the session's actual ChArUco pattern, proportions and marker layout. The position map follows the detected board centre; when a useful pose is ready, it switches to the full board and **Hold still**. A repeated pose prompts an outstanding tilt or a change in distance. After capture, the guide advances automatically. These cues use the same novelty and coverage decision as capture, without imposing extra poses.

The accuracy panel shows how many areas pass, need work, or remain unchecked. **Improve lower left**, for example, requests a new training pose in the failed region; **Verify lower left** confirms the corrected area. The remaining areas are not restarted simply because the model was refitted.

**Saved pairs** shows every recorded stereo pair, including earlier checks and undone captures. Labels distinguish images used in the current fit, images waiting for fitting, and check results. Click a thumbnail to download its original PNG. Measurements and coverage remain under **Measurements** and **Session & diagnostics**.

For a completed session, start `stereo-calibrate gui --session PATH --inspection-log`. This creates a timestamped run inside `PATH/diagnostics/runs/`, with an append-only `readings.jsonl`: the frozen model and board/camera settings, then every processed detection frame with timestamps, IDs, raw/rectified corner coordinates, quality, and alignment measurements. Zero detections and processing errors are retained. It records all computed readings, not every acquired camera frame or raw video. The live verification panel shows the recording counter and **Stop recording**, which closes the log while the preview continues. Calibration changes are blocked until recording stops; this diagnostic run never adds training captures or changes fitted parameters. Each run requires a new directory. An explicit new directory may still follow `--inspection-log`.

Add `--inspection-images` for short diagnostic holds when investigating unstable corners. Each reading then links to its exact, full-resolution side-by-side PNG under the diagnostic run's `frames/`, including frames with no detections. PNG preserves the camera's decoded pixels without further lossy compression; it cannot undo the camera's MJPEG compression. Images are saved before their readings, and an image-write failure visibly stops recording. Saving every processed image uses disk space and may reduce detection throughput; use **Stop recording** after the hold. These images stay separate from calibration training and validation captures. Quality records also identify the detector's selected scale and contrast variant for each eye.

The detector uses OpenCV’s board-based marker recovery to reconsider rejected marker candidates before locating ChArUco corners. It uses the current image and known board layout, with no temporal smoothing, cached corners or fitted camera parameters. This improves marker dropouts in the recorded left view; it does not guarantee all corners in every frame.

ArUco marker-corner refinement is disabled (`CORNER_REFINE_NONE`), following
[OpenCV’s ChArUco recommendation](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html):
nearby chess squares can bias marker subpixel windows during homography-based
interpolation. The final ChArUco chessboard corners still receive subpixel
refinement. This policy applies to both stereo and hand–eye capture; existing
saved corners and fitted camera models are preserved.

ChArUco native-pixel refinement retries unchanged estimates with a wider initial search, then the original fine search. OpenCV can silently restore a seed when the correction exceeds the search half-size; this left some contrast-enhanced estimates several pixels off the visible corner. Refinement then uses small image patches aligned with the local board axes to better locate skewed corners. Neighbours orient each patch, while its centre stays anchored to the measured corner; image gradients determine the final position. Both steps share the existing 5 px displacement budget. Unsupported patches retain their native estimates. Refinement does not use calibration residuals to select or hide points, and it does not rewrite saved observations or fitted models. Ambiguous or blurred corners still need inspection; improved detection alone does not certify calibration accuracy.

**Live verification** under the preview is a read-only inspection tool, available before, during and after collection. Counts show the left, right and shared ChArUco corner IDs out of the selected board's total (70 for the 11 × 8 board). Open it to see a map of the actual board corners and numbered overlays in either Original or Rectified view. Green marks shared IDs, blue left-only, amber right-only, and grey missing corners. In **Detected frame**, the overlay and counts refer to the same displayed frame. **Live video** displays newer camera images without borrowing older corner coordinates; its counts are labelled Latest detection. Enabling corner IDs switches to Detected frame. Replay always uses exact saved frames and detections and is labelled Recorded.

Detection count alone cannot establish calibration quality. With a fitted model, the inspector reports stereo alignment P95 in original-camera pixels: each shared bearing is projected onto the other camera ray's epipolar plane, then converted back to sensor pixels; the larger left/right correction is used per corner. This needs no planar pose fit and cannot improve just by zooming the preview. Rectified vertical P95 is shown separately for points inside both output images, with the in-view count and the all-point value alongside it. These are current-frame diagnostics, not a new acceptance gate or a change to the saved calibration result. Hold the board still; physical depth accuracy remains unmeasured.

The acceptance target is **below 1 px RMS** for the left/right/stereo training fit, each fresh eye's reprojection, and fresh stereo prediction in **both directions**. For a stereo check, the board pose estimated in one camera predicts the matching corners in the other using the frozen calibration. All these checks use original camera pixels; preview zoom or crop cannot improve the result. Rectified vertical errors are retained as diagnostics, not acceptance criteria: their scale depends on the chosen perspective projection and becomes unstable near the fisheye horizon. These checks establish sampled geometric consistency, not metric-depth accuracy or a parameter-uncertainty estimate. Baseline is reported in millimetres; the GUI does not assume a nominal 60 mm baseline is ground truth. A single combined USB frame also does not prove simultaneous sensor exposures—hold the board still.

Preview delivery targets 30 fps independently of the full-resolution detector; the footer reports delivered video and detection rates separately. Left and right detection run concurrently with separate detector instances, and OpenCV uses its available CPU thread pool. Capture, refinement, logging and calibration keep their original resolution and detection policy. The browser requests only the newest frame, stops video requests in background tabs, and keeps at most one request in flight.

The rectified preview uses a centred perspective viewport with the average of the fitted central focal scales. A perspective image cannot show the entire fisheye field. The exported `P1`, `P2` and `Q` use that same viewport and baseline; validation still uses all original shared corners, including those outside the preview.

## Resume, replay and offline operation

The session path appears under **Session & diagnostics**, with **Open session folder**, **Open diagnostics**, and **Open latest log** when those folders exist. The latest log is found again after restarting the GUI. Folder buttons launch the desktop file manager and do not change recording or calibration state. The header’s **Results** link jumps to calibration downloads. **Sessions** switches within the running GUI; existing captures and results are retained, and resumed collection starts paused. Finish an active fit or stop diagnostic recording before switching. Only one writer may open a saved session. CLI `--session` and `--replay` still open their target directly, bypassing the startup picker.

```bash
# Resume with saved settings, initially paused
uv run stereo-calibrate gui --session sessions/SESSION_NAME

# Review originals and stored results without opening a camera or changing files
uv run stereo-calibrate gui --replay sessions/SESSION_NAME

# Fit saved training observations and freeze a new candidate
uv run stereo-calibrate calibrate sessions/SESSION_NAME

# Re-evaluate that candidate's saved fresh observations
uv run stereo-calibrate validate sessions/SESSION_NAME
```

Each session has `session.json`, full-resolution PNGs under `frames/`, candidate exports under `results/CANDIDATE_ID/`, and diagnostic runs under `diagnostics/`. Diagnostic images remain separate from fitting and validation captures. Historical audit files migrated into `diagnostics/audit/` retain their original bytes and relative image links; a checksum manifest records the move and the old directory remains a compatibility symlink. Camera intrinsics, distortion, resolution, board geometry, units, software versions, capture roles and fit provenance are recorded. `R,T` map left-camera coordinates into right-camera coordinates; translation uses millimetres. Original side-by-side frame ordering is preserved even when logical left/right views are swapped.

## Existing scripts

The [manual stereo CLI workflow](stereo_calib_charuco/README.md) remains available, as does the separate [pinhole single-camera workflow](calib_charuco/README.md). The stereo detector and fisheye helpers now share the package implementation.

The manual stereo capture script's old “50 images” label was only a collection reference. It now displays the number saved without a default quota; `--target` remains an optional reference. That script does not judge calibration quality. Use `stereo-calibrate gui` for guided capture and automatic fresh-view evaluation, or the session `calibrate` / `validate` commands for offline fitting and evaluation.

**Migration:** stereo defaults are now 11 × 8, **20 / 15 mm**, legacy layout. Old recordings made with a 90 / 67 mm board require their original board dimensions; do not recalibrate them with the new defaults. Existing parameter files are not rewritten. Do not mix the single-camera pinhole model with the fisheye stereo model.

## Development

```bash
uv run pytest -q
uv run ruff check stereo_calibration tests
node --check stereo_calibration/static/app.js
uv build
```

Tests cover numerical geometry, actual corner correspondence, automatic capture, persistence failures, held-out separation, and read-only replay. Physical lens calibration still requires a real guided capture session. See [architecture and research decisions](docs/design.md) for module boundaries, solver choices and limitations.

## Hand–eye mode: fixed camera, board on the wrist

Open **Hand–eye calibration** in the header, or **http://127.0.0.1:8765/hand-eye**.
Open your completed, geometrically checked stereo session in live mode first and
leave stereo collection paused. This mode keeps its calibration and camera settings
frozen. It consumes the existing camera owner's original packets; it never opens a
second device. Live video stays independent of the slower native-resolution detector.

The main preview defaults to **Live video · smooth**, using the same latest-only
stream as stereo calibration. The inspector below it independently loads the
exact measured image with numbered corners, at up to two updates per second.
Turning on corner IDs does not switch the main preview away from live video.
**Inspect detected frame · slower** deliberately enlarges that measured image in
the main view. The board map and counts always match the displayed inspection
frame; old points are never overlaid on newer live video.

For external diagnostics, `GET /api/live.png?workspace_id=…&after=…` returns the
latest native-resolution frame as a lossless PNG from the existing camera owner.
It preserves original stream order, without overlays, rectification, or resizing.
Headers provide the frame ID, workspace ID, and host-receive monotonic timestamp
(not a hardware exposure timestamp). An unchanged frame returns 204; a stale
workspace returns 409. Reading this endpoint does not collect calibration samples.

The main workspace shows the current action, key errors and live video. Expand
**Inspect detected corners** for the measured image, **Calibration details** for
coverage/timing/history, or **Manual controls** for individual captures and undo.
A paused guide remains visible as a neutral board reference with **Review details**;
it does not show a motion target while prediction accuracy needs review.
The blue dot remains visible in the left camera image. With a valid position
guide it marks the board position; during review or unavailable guidance it is
labelled as a fixed image-centre reference, with the target area hidden. The main
guidance uses direct position, tilt and hold cues without “optional” labels;
capture still depends on measured quality and coverage rather than exact target matching.
When accuracy fails, the guide names the error and limit. Recording attempts
have a separate status, so a timeout cannot hide the calibration failure and a
saved pose is not mistaken for a validated calibration.

Recoverable capture rejections and timeouts **retry automatically** when fresh,
valid camera and measured-arm data return. The normal quality, novelty and
stationarity checks still apply alongside the guided hold. A persistent
**Will retry automatically** message explains the wait; **Stop automatic capture**
cancels it. Ordinary positioning delays stay live guidance, without repeated
error popups or capture requests. Accuracy failures and recording faults stay
paused with their reason. For an accuracy failure, **Capture one diagnostic pose**
records one sample after review; it does not resume automatic collection or clear
the failed model. Ordinary recording faults retain **Retry capture**.
Automatic retries never clear previous validation failures.

The hand–eye fit now uses OpenCV PARK as an initializer, then jointly optimizes
the camera-to-base and board-to-tool transforms against every original corner
in both cameras. The stereo calibration, physical board dimensions and measured
robot kinematics remain fixed. Each held-out orientation group gets its own
initializer and refinement using only the remaining training observations;
fresh validation images never enter fitting. The 3 px per-eye limits are unchanged.
This follows the fixed-intrinsic reprojection formulation of
[Tabb and Ahmad Yousef](https://arxiv.org/abs/1907.12425). Optimizer method/version
and initial/final residuals are recorded. Older qualified solutions must be
reassessed before reuse under the new method.

Motion diagnostics compare the relative rotation angles measured by the arm
and by the board images. These angles must agree for a rigid, fixed-camera setup,
independently of the unknown hand–eye transforms. Disagreement may come from
mount movement, robot kinematics/encoder calibration, or camera pose bias; this
diagnostic alone cannot identify which. Never fit unverified joint offsets or
relax acceptance limits to make a failed recording pass.

Pose suggestions start with six local ±12° board rotations to establish two-axis
orientation coverage that survives leaving each orientation group out. Once that
independent coverage exists, six ±25 mm camera-axis translations also compete.
Candidates are ranked by missing rotational coverage, weakest held-out coverage,
then expected stereo reprojection information gain. The information estimate uses
a fixed rigid model and unit pixel noise; it is not a physical uncertainty estimate
and does not model arm errors. This adapts the information-gain principle of
[Yang, Rebello and Waslander](https://arxiv.org/abs/2303.06766) to manual fixed-camera
view suggestions. A short reason explains what each view adds. The guide does
not optimize arm-joint coverage or perform robot IK/collision checking. A matched
guide is not an accuracy check; after an accuracy failure, a positioning suggestion
only helps keep the board visible.
**Manual controls → Exclude rejected poses** explicitly removes failed active
training poses from fitting, using the latest full-fit assessment. Excluded
records and their original images remain under the session history; a snapshot
records the assessment and exact IDs. Excluding poses can reduce coverage and
never establishes a passed calibration by itself. After reassessment, collection
continues automatically if the remaining poses need more coverage and no
accuracy fault requires review. Pausing or restarting the app disarms collection.

The three visible stages are:

1. **Setup:** choose the measured follower arm and confirm the board is rigidly
   mounted and printed at the checked scale. The small L20 target is 4 × 5 squares,
   25 mm squares / 20 mm markers, `DICT_4X4_50`, IDs 20–29, modern layout. This
   definition is separate from the stereo session's larger board.
2. **Guided calibration:** press **Start guided capture**. Hold a comfortable first
   pose with the complete board visible in both cameras. This establishes the
   orientation reference. The board illustration pauses at the measured pose in
   red for two seconds, demonstrates **larger tilt → other tilt → in-plane turn**,
   one printed-board axis at a time,
   then holds the green suggested pose for two seconds. Each step takes two
   seconds and pauses briefly before the next axis. These
   are intrinsic board rotations, not robot joint commands. The complete sequence
   uses one measured snapshot and refreshes on the next loop, so detector noise
   cannot change the directions halfway through. An overshoot of more than one
   degree restarts the demonstration with the corrected direction. Material
   same-direction movement also refreshes the measured starting pose; sub-degree
   components are skipped.
   **Live remaining angles** replace the old combined orientation-error line:
   all three numbers refresh with each received detection (guidance targets 10 Hz;
   browser status updates up to 5 Hz),
   independently of the animation. Edge names follow the left camera view, with
   toward/away-camera or clockwise/counterclockwise directions instead of bare
   X/Y/Z labels. The larger tilt component determines the order once per suggestion;
   the order remains fixed as readings change. This is a usability heuristic,
   not a robot motion plan. Stale readings are labelled and eventually cleared.
   The 30 fps live preview remains independent. Reduced-motion mode shows the
   same steps as discrete poses. This adapts the sequential guidance idea in
   [Calibration Venus, §3.4](https://arxiv.org/abs/2009.05983); XYZ or YXZ is used
   according to the dominant initial tilt, with the turn last.
   A blue edge marker names motion toward/away from the camera;
   in-plane turns get a clockwise/counterclockwise instruction. **Suggestions
   are optional:** any valid, stationary, sufficiently different measured pose
   can be recorded, even outside the blue area or away from the suggested tilt.
   The hold bar tracks one second of stable, fresh paired wrist poses while the
   700 ms joint-stability window settles in parallel. Full detection, fresh
   feedback and stationarity through the capture decision must also pass before
   a pose saves automatically. During the
   hold, the illustration shows the actual pose and says "recording".
   **Captured — you can move
   now** confirms completion. The next suggestion advances automatically.
   **Another suggestion** changes the suggestion without crediting a capture;
   choose it if a tilt is awkward. Position remains free within your clear
   workspace. Stop automatic capture to use **Capture this pose manually**.
   Coverage and measured image agreement drive progress, rather than a pose quota.
   The sidebar shows rotation coverage, position extent, worst fit error and
   withheld-pose prediction error. The count is an audit trail only. Missing
   corners, stale feedback, movement, bad geometry or wrong resolution prevent
   capture. Repeated poses never save twice. A pending capture times out after
   10 seconds, then waits for a qualifying pose to retry automatically rather
   than requiring another click.
   Motion and reprojection failures include their measured value and limit.
   Temporary camera delays reset the hold timer and wait for fresh frames without
   disarming automatic capture. Unexpected recording failures still stop it and
   are recorded with their error message in the session log.
   Frame freshness is checked on arrival at analysis (250 ms), separately from
   bounded processing time (the existing 700 ms stationary window). Before
   recording, measured joints must remain stationary from before the image
   through the capture decision, with fresh continuous feedback. Measurements
   and saved captures include timing diagnostics for these separate checks.
   Fitting runs in one background worker, independently of the live preview.
   When two-axis coverage, full-fit error and leave-one-out prediction qualify,
   the candidate freezes automatically. Keep moving and pausing: subsequent
   poses validate it without entering the fit. Validation needs its own varied
   coverage and every measured error within budget. No Next/Solve step or second
   Start is required. A failed fresh-pose check stops automatic recording and
   remains in the report. Sufficient coverage with poor training or withheld-pose
   prediction also stops automatic recording and displays a review explanation.
   A separate camera-only positioning guide stays available while recording is
   paused: its blue area stays near the measured board position and its small
   tilt is checked for visibility in both cameras. It never qualifies a capture,
   certifies calibration, or checks robot clearance. The blue position reference
   stays visible through tracking gaps; it is a fixed image location, not a claim
   of current tracking. A fresh partial board pose can update advisory angle
   readings only after separate native and bidirectional stereo checks. Recording
   still needs all twelve corners. Missing usable geometry clears the numbers to
   “Waiting for pose” while keeping their panel visible. “Another suggestion” refreshes this advisory goal without
   restarting recording; the accuracy error remains visible below the guide.
3. **Result:** download the report and transforms. **Open session folder** opens
   the local directory directly. Nothing is applied to robot configuration.

Live hand–eye guidance first detects at native image scale. If corners or camera
geometry fail, it retries the full multiscale search on the same image. Every
frame saved for manual or automatic capture always uses the full search and the
original pixel coordinates. The camera owner supplies only the newest packet;
processing cannot build a queue of old images. The one-second pose dwell starts
while the preceding joint window settles, but moving joints, discontinuous or
stale feedback, excessive pairing skew, and motion during image processing still
block recording. This is a stop-and-capture workflow, not continuous-motion
calibration. Timing diagnostics record whether the full search was used.

The sampling design follows [ROS camera calibration's motion and novelty checks](https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/calibrator.py).
Continuous-motion hand–eye would additionally require exposure timing and pose
alignment, as in [ETH's time-offset and hand–eye workflow](https://github.com/ethz-asl/hand_eye_calibration).
The camera timestamps here remain host-receipt times, not calibrated exposure times.
The shorter holds are operational policies; they do not certify physical accuracy.

The sampling flow follows the measured pose-pair approach in
[MoveIt Calibration](https://moveit.picknik.ai/main/doc/examples/hand_eye_calibration/hand_eye_calibration_tutorial.html)
and [easy_handeye2's sampler](https://github.com/marcoesposito1988/easy_handeye2/blob/master/easy_handeye2/easy_handeye2/handeye_sampler.py).
The previous 5° suggestion match was a UI policy, not a hand–eye requirement;
the guidance policy removes it from automatic capture. OpenCV PARK initializes
the joint stereo reprojection fit, with inverse robot poses for the fixed-camera configuration. Image
quality, freshness, stationary dwell, novelty, two-axis diversity and independent
validation still apply. No suggested pose is sent to the robot.

Pause preserves the run. A saved run can be resumed only with matching stereo
calibration, arm model, target geometry and capture policy. Physically moving the
camera/arm base or remounting the board requires a **new hand–eye run**. Moving the
camera as one rigid stereo rig does not by itself require refitting stereo
intrinsics; changing optics, stereo geometry, resolution or pixel mapping does.

The read-only bridge subscribes to `/slave_{side}/joint_states_single` in Rubin's
ROS environment. It never publishes commands, calls control-mode services, opens
CAN, or changes teleop routing. FK uses named measured joints in radians and the
literal Nero `base_link → link7` URDF chain, with its SHA-256 recorded. This is the
arm's `link7` frame, **not an assumed L20 palm/TCP frame**. The board mounting
transform is estimated from observations, not taken from the mounting drawing.
Nominal URDF joint limits describe the command envelope. Finite measured angles
outside that envelope remain unchanged in this passive FK calculation; the
discrepancy is displayed and logged, rather than blocking capture or clipping
the angle. Robot command limits and control ownership are unchanged.
A measured/model-validated tool-to-palm transform is still needed by grasp code
that expresses library poses in the palm frame.

Hand–eye reuses stereo's single board illustration and camera overlay. After a
reference capture, it considers local ±12° rotations about one printed-board axis;
once two-axis coverage survives held-out orientations, it also considers ±25 mm translations.
It rejects suggestions unless the full board outline and inner corners remain
visible in **both native fisheye images**, including intermediate poses in either
UI rotation order or along a straight translation. It checks the printed face, projected size, margin and viewing
obliquity relative to each camera-to-board-centre ray. It avoids worsening an
already edge-on view. These are camera-visibility heuristics, not arm reachability
or collision checks; suggestions never command the robot.

The selected orientation and blue position anchor stay fixed while you move.
For a tilt, the anchor is the current board centre when selected. For a position
change, it is the projected new centre; metric translation error drives the cue
and a 5 mm arrival tolerance. Neither goal follows the live board. If you move far enough that a suggestion is no
longer reasonable, its motion cue disappears and **Another suggestion** requests a
new local suggestion. The board illustration is a schematic in the local viewing
frame; the live camera remains the placement reference. Animation skips components
below 1° and refreshes after substantial measured motion, while filtering jitter.

The first solvable fit is provisional if removing a saved orientation leaves
fewer than three distinct orientation groups or numerically parallel rotation
axes. The complete training set must meet the capture coverage policy; reduced
cross-validation folds need not reproduce that complete acquisition envelope.
Each observable fold must still predict its withheld images within the unchanged
3-pixel budget. Collection continues with tilt suggestions and any
pixel-error warning remains visible; a pose count alone does not qualify the fit.
Once coverage survives those held-out groups, poor image predictions cause **Review prediction
accuracy** to replace tilt guidance and stop automatic recording. Do not seek larger
tilts to clear this state. Review board rigidity, measured-pose pairing, FK/model
and camera consistency. Manual diagnostic capture remains available. The normal
capture dwell is 1 s; a detection gap over 1.5 s or drift above 3 mm / 1.5° resets
it. These are recording heuristics, not calibration accuracy tolerances. The
stationary-joint and native-pixel checks still apply. Guidance policy, source hash,
suggested target and capture mode are logged. Pause/restart/resume leave automatic
recording off.

Run the animation/status regression checks with
`node --test tests/test_handeye_frontend.mjs` alongside the Python tests.

Timing is explicit: camera packets are stamped at host receipt, robot messages
retain their source stamps and receipt times. Capture requires a stationary
0.7 s feedback window, maximum joint range 0.006 rad, source age ≤ 250 ms,
continuous feedback gaps ≤ 80 ms and a paired sample within 80 ms. These checks
reduce motion/timing error; they do not establish synchronized sensor exposures.
Missing feedback always blocks capture. No leader or command pose is substituted.

Operational defaults live in `handeye.POLICY` and are copied into every run.
Novelty is 15 mm translation **or** 8° rotation; fitting/checking also requires
at least 25° rotation span and a second rotation-vector RMS singular value
≥ 0.15 rad (8.59°). Coverage uses deterministic complete-link orientation groups
with diameter ≤ 8°, reusing the existing novelty scale. Each group contributes
one minimax representative, irrespective of repeated/noisy image count. Pairwise
rotation-vector singular values are normalized by the square root of the number
of pairs. The versioned method is recorded in assessment metadata; source hashes
identify reassessment code without rewriting original capture provenance.

These thresholds are operational conditioning rules. OpenCV requires at least
three poses with nonparallel rotation axes to solve; leave-group-out testing
requires sufficient remaining orientation support to solve every fold. Counts
alone never grant readiness. Whole orientation groups leave together, including
translated views of essentially the same tilt, and **every recorded image** is
still evaluated. This is deliberately conservative: near-duplicate measurements
cannot replace an independently omitted tilt. Every full-fit and held-out
prediction must meet the same 3 px per-eye budget. Fresh validation then checks
the fixed candidate across separate orientation support; its three-view minimum
is a coverage policy, not a separate solve or a mathematical accuracy guarantee.
These are collection/conditioning heuristics. Capture uses ≤ 2 px per-eye native
reprojection RMS and ≤ 3 px bidirectional stereo prediction RMS. Independent
checks require ≤ 3 px RMS in **each native camera image**, using all observed
corners and the frozen camera/robot transform. The 3 px value is an explicit
engineering review budget, **not a 3 mm physical tolerance or certified grasp
accuracy**. Reports also retain position/angle discrepancies without inventing
an equivalent physical accuracy guarantee.

The solver uses OpenCV's [eye-to-hand form of `calibrateHandEye`](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html)
(Park method), with inverse measured follower poses. `T_a_b` maps coordinates
from b into a; exported translations are **metres**. The equation is
`T_base_tool @ T_tool_board = T_base_camera @ T_camera_board`.
Exports include raw-left, raw-right and rectified-left camera transforms with
explicit frame names. The current OpenCV 5 wheel lacks the Python hand–eye
binding, so solving uses a small isolated, pinned OpenCV 4 environment:

```bash
uv venv --python .venv/bin/python envs/handeye-opencv4
uv pip install --python envs/handeye-opencv4/bin/python -r requirements-handeye-solver.txt
```

An independently qualified per-arm geometry profile can be installed under
`nero-linkerhand-teleop/calibration/arm_kinematics/{right,left}/profile.json`.
Its `arm_joint_zero_calibration.v1` manifest binds the source URDF, joint-zero
corrections in radians, generated model and independent-check qualification.
The app verifies these before starting a run. The generated URDF incorporates
the corrections in joint origins: **use raw measured encoder angles with this
model, without adding the offsets a second time**. Controller commands, firmware
zeros, gains and the other arm remain unchanged. Re-zeroing or replacing the
arm requires requalification of its profile.

For a separately qualified nonlinear encoder mapping, an
`arm_kinematic_calibration.v2` profile also contains `joint_angle_maps`.
Its generated `arm_model.json` embeds the corrected URDF and explicit
raw-to-model angle formulas. Load it with `ArmKinematics`; a standard URDF
parser cannot represent a nonlinear encoder mapping. The app applies the map
only to FK and retains raw recorded angles. Values outside the independently
validated interval cannot be used for calibrated FK; they are never silently
clamped or extrapolated. This empirical correction does not identify or repair
the underlying hardware or firmware cause.

A separately validated static configuration correction uses profile schema
`arm_configuration_calibration.v3` and model schema `arm_kinematic_model.v2`.
The model embeds its reference geometry, quadratic raw-angle maps and the
configuration feature coefficients. It is self-contained and must be loaded
with `ArmKinematics`; a plain URDF reader cannot apply this correction.

The gravity-shaped feature is an empirical predictor for a stationary arm
with the same rigid mounting and payload. It is not a measured stiffness,
a firmware repair, or a change to gravity feedforward. It changes FK only;
raw encoder feedback, motor commands and physical joint limits keep their
original convention. Requalify after changing the payload, arm mounting,
board attachment, joint zeros or hardware. Finite sampled pose coverage does
not certify all combinations of joint angles or dynamic operation.

The profile records prospective frames and their unchanged pixel-error limit,
failed developmental holdouts and residual angular disagreement. Passing a
pixel budget is distinct from identifying the hardware cause or validating
physical manipulation accuracy. New calibration runs freeze the complete model
and its format; never reuse a transform with different kinematics.

Each run freezes `arm_model.urdf` or `arm_model.json`; exports identify its
SHA-256 and file path. The legacy `urdf_sha256` export field fingerprints the
complete selected model file, including angle maps when present.
Use that exact model when converting raw joint feedback to `T_base_tool` with
the exported camera transform. A run recorded with a different model cannot be
resumed into the corrected geometry. Old recordings are preserved for diagnosis.

Do not install these solver packages into the camera environment. OpenCV 4 camera
environments that expose `calibrateHandEye` can call it directly. The read-only
robot bridge uses the existing environment under
`/home/pairlab/Desktop/rubin/nero-linkerhand-teleop`; its source and URDF paths are
specified in `stereo_calibration/handeye_runtime.py`.

Each run is stored under `sessions/SESSION/handeye/RUN/`: `run.json` (original
observations, timing windows, geometry, policies and provenance), `frames/*.png`,
`events.jsonl`, `feedback.log`, snapshots of the arm URDF and stereo model, frozen
`solution_ID.json`, and `result.json`. `handeye.npz` is exported only after all
independent checks pass. Training exclusions preserve the images and event trail.
**Geometric checks passed** is intentionally distinct from physical task accuracy.

Current capture readiness and failed calibration checks are displayed separately.
A sharp, stationary board can be captured while the fitted transform still fails
independent validation; an old check failure must not hide today's capture blocker.

### Offline recovery of a slightly curved target

`scripts/diagnose_handeye_depth.py` triangulates matched corners with the frozen
stereo model, then jointly fits the camera transform and each corner's 3D position
in the tool frame. It uses native fisheye reprojection, inspired by
[general hand-eye reprojection minimization](https://github.com/koide3/st_handeye_graph).
Only training poses enter the optimizer. All independent checks remain held out.
This diagnostic assumes a rigid board shape and attachment throughout the run;
it cannot correct a moving mount, inaccurate robot poses or camera calibration.
Corner-distance repeatability is reported separately from physical accuracy.

On this host, system Python already provides NumPy, OpenCV and SciPy; the live
camera environment is unchanged:

```bash
/usr/bin/python3 scripts/diagnose_handeye_depth.py \
  sessions/SESSION/handeye/RUN/run.json \
  --output output/depth-recovery-RUN
/usr/bin/python3 -m pytest -q tests/test_handeye_depth.py
```

The report and reconstructed points are diagnostic candidates only. The command
does not alter the source run, apply transforms, contact the robot or change
acceptance thresholds. Output must be outside the original run directory.

### Restarting a failed attempt

After an explicit operator request and confirmation that the board/mount are
unchanged, `handeye_recovery.fork_checkpoint` can copy the accepted training
checkpoint into a new attempt. The failed source run and all of its images,
reports and checks retain their original bytes; a separate `archive.json` marker
hides it from ordinary resume choices and points to the replacement. The new
attempt records its parent and copied capture provenance, carries no old solution
or passed status, and must qualify a new fit before fresh validation. Failed
archived views are linked by source-run/image hashes and evaluated against every
new candidate, including nested restarts. They never enter fitting or the active
capture count. A new candidate must also predict these views within the unchanged
pixel budget. Missing or changed evidence blocks resume. Archiving failure is not
evidence that the cause is fixed. Pause the source before forking.
