# Architecture and calibration decisions

## Boundaries

| Module | Owns |
| --- | --- |
| `config.py`, `boards.py` | Immutable board/camera settings, canonical OpenCV geometry, vector PDF generation |
| `detection.py`, `observations.py` | ChArUco detection, original-resolution corners, matching IDs and live measurements |
| `opencv.py`, `intrinsics.py`, `calibration.py` | OpenCV version adapter, intrinsic fitting, fixed-intrinsic stereo fit, exports |
| `validation.py`, `rectification.py` | Frozen-model evaluation in original pixels, diagnostic vertical errors, finite perspective viewport and spatial coverage |
| `guidance.py` | Board pose, movement goals, duplicate suppression and steady-pose selection |
| `verification.py` | Re-evaluation of saved check images, passing region coverage, and targeted confirmation after corrective fits |
| `camera.py` | A single V4L2 reader, latest-frame handoff, or read-only recorded-frame replay |
| `session.py` | Session lock, atomic files, capture roles, candidate history and fit snapshots |
| `controller.py` | Single-writer workflow state machine and command queue |
| `preview.py`, `web.py`, `static/` | Display rendering, local HTTP endpoints and browser interface |

The camera thread continuously replaces the latest frame, so a slow detector does not accumulate an old-frame queue. The controller alone changes workflow/session state. Numerical fitting runs in a spawned worker process with a training-only snapshot; its result is accepted only if the training IDs still match. Measured-frame JPEGs and their corner overlays have matching frame IDs. Live video has an independent latest-frame endpoint and never overlays corners from an older measurement. The image shown may be resized, but detection, calibration, validation and saved originals use the capture resolution.

The server binds to loopback and checks Host/Origin. A plain JavaScript frontend avoids a separate Node build or production server. The local stereo dashboard in `graph-as-policy/gap/connector/nero_l20/` informed the visual layout and camera-ownership approach; no robot-runtime dependencies were imported.

## State and data

The main progression is `ready → collecting → fitting → validating → done / needs_attention`. An early provisional fit returns to collection. Pausing is separate from the phase. The UI only freezes a final candidate once the training goals are covered after applying its estimated geometry.

The interface presents collection, fitting, validation and corrective retries as one **Guided collection** stage. The operator follows short movement cues throughout; the training/check separation remains internal and appears in diagnostics. The guide's board images are rendered from the active `BoardConfig`, including exact row/column counts, marker dictionary, legacy layout and proportions, rather than a decorative checkerboard.

Original PNGs are written before manifest references. File writes use a temporary file, flush/fsync and atomic replacement. Failed manifest writes roll back in-memory capture and phase changes. Undo is a manifest flag, not physical image deletion. The manifest is the authoritative session index; unreferenced artifacts left by an interrupted export are not silently imported on resume.

Validation records retain the ID of the candidate active when they were captured. Evaluation uses check images from the current candidate and its predecessors, recomputing every residual and region under the current model. Previous candidates and reports remain unchanged. Starting another training round clears only the active candidate pointer; its report remains visible while collecting corrective training. A candidate records which corrected regions require a new post-fit confirmation. Region status and residuals are recomputed on writable resume; replay remains read-only. Replay cannot save, refit, undo or reconnect a camera. A filesystem lock prevents two calibration writers from opening the same session.

## Research translated into changes

- **Steady automatic capture and visible coverage.** OpenCV's [interactive calibration application](https://docs.opencv.org/4.x/d7/d21/tutorial_interactive_calibration.html) uses an interactive collection workflow. This implementation adds stereo-specific shared-corner checks, movement cues and a stable hold, so manual save-key timing is no longer required.
- **Correct geometry and meaningful image features.** OpenCV's [ChArUco documentation](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html) distinguishes chessboard corners from marker corners and requires matching board geometry. Object coordinates now come from the same canonical board object as detection. Rectification validation uses corresponding actual ChArUco corners; the legacy marker fallback uses all corresponding marker corners. An arithmetic mean of distorted marker corners is not treated as a projected physical marker centre.
- **Use the fisheye model consistently.** The [OpenCV fisheye API](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) supplies the intrinsic, stereo and rectification operations. Each eye is fitted first, then intrinsics are fixed for the stereo extrinsic fit. Fisheye bearings are undistorted before pose estimation; pinhole distortion parameters are not substituted into fisheye operations.
- **Do not infer accuracy from a low training RMS alone.** [Calib.io's calibration guidance](https://calib.io/blogs/knowledge-base/calibration-best-practices) emphasizes target quality, feature size, varied views and residual interpretation. The dashboard reports coverage, retains failed fresh views, and evaluates new observations against frozen calibration parameters. A pass establishes the recorded geometric consistency, not depth accuracy on arbitrary scenes.

OpenCV's ChArUco tutorial warns that marker-corner refinement can be influenced by nearby chessboard features. The existing refinement policy was not changed merely because that warning exists: toggling refinement on this rig's poorly resolved tabletop A4 frame did not recover useful detection. Moving the board closer did. A future detector-policy change should compare residuals and repeatability on representative physical images, not just maximize the number of detected corners.

## Compatibility work that was necessary

OpenCV 5 moved fisheye calibration flags into the top-level namespace and their numeric values differ from the 4.x namespace. `opencv.py` resolves the installed enum rather than copying numbers. It also supplies contiguous `(1, N, C)` point arrays, which were exercised with OpenCV 4.13 and 5.0. The OpenCV 5 point-layout failure is tracked in [opencv/opencv#29692](https://github.com/opencv/opencv/issues/29692).

The installed stereo bindings require equal point counts across pairs. The fitter therefore intersects corner IDs after per-eye training filtering and uses a deterministic spatially spread subset with the minimum available count across usable pairs. It does not throw away all but an arbitrary fixed twelve corners. This workaround is isolated in `calibration.py`; complete original observations remain in the session, and validation uses all shared corners.

Fisheye fitting explicitly enables `CALIB_USE_INTRINSIC_GUESS`; without it OpenCV ignores the supplied starting matrix. On a pose-initialization or conditioning failure, progressively larger subsets of the most central views initialize K, then **all observations** are fitted together. High-order distortion from the central subset is not extrapolated into the full fit: D starts at zero and all four coefficients remain free. Conditioning checks stay enabled, no training view is discarded by this recovery, and subsequent corner-filter iterations reuse the converged K/D. This recovered the real 12-pair wide-field session after the generic initialization failed. A geometry-only synthetic regression reproduces the difficult poses with known parameters; see also OpenCV's [initialization issue](https://github.com/opencv/opencv/issues/23962).

The guided fitter retains the existing per-eye training-corner rejection policy. It does not duplicate the legacy standalone script's separate stereo-pair MAD filter. Rejected training corners and pairs made unusable by that filtering are recorded. There is no outlier removal, viewport clipping, or favorable-view selection in fresh validation.

## Heuristics versus acceptance checks

The 1-second hold, 3-frame minimum, board-relative movement tolerance, pose-novelty spacing, position targets, 1.3× near scale and 15° tilt goals are operator guidance heuristics. They prevent accidental duplicate captures and encourage diversity; they do not prove sufficient parameter observability. Before a provisional model exists, raw projected edge ratios provide approximate tilt guidance. Afterwards, fisheye-aware pose estimates replace them.

The 1 px RMS budget applies to the training fit and fresh mono / bidirectional inter-camera reprojection in original sensor pixels. In each direction, pose is estimated in the source camera, transformed with frozen R/T, and projected into the target camera; the target image is not used to refine that prediction. All shared corners participate, with no held-out outlier filtering. Rectified vertical median/P95 remain diagnostic measurements (historical 0.5/2 px references), not acceptance gates. Sharpness and pixel square size are advisory; there is no universal hard cutoff for them.

There is no 12- or 50-pair completion rule. Three poses are the numerical seed floor, not evidence of sufficiency. The guided controller additionally waits for a centre view and both tilt axes before trying provisional geometry; it retries as new movement goals are covered. Complete pose coverage triggers the candidate fit. Completion requires all saved check measurements to pass under the current model, full check-region coverage, and post-fit confirmation of corrected regions. Each failed check automatically requests a new training pose in its failed region, then refits and re-evaluates every saved check. Training collected before that retry cannot satisfy its request. Passing regions retain coverage only after their residuals pass again; any newly failing region becomes a retry target. Failed measurements remain in the denominator even when a newer capture of the same area passes. No check image is promoted into training. Retry targets, reasons, original candidate provenance, and confirmation requirements survive restart.

These images remain separate from numerical fitting, but their residuals guide adaptive collection. Reused checks therefore establish sampled geometric consistency, not an unbiased final-test accuracy estimate. This distinction is recorded in the exported report. A persistent failed observation needs diagnosis of its residuals, detection, board flatness, or model limitations; repeatedly capturing favourable replacements cannot erase it.

Fresh per-eye reprojection was previously recorded but did not affect acceptance, allowing an intrinsically poor model with good epipolar alignment to pass. The [Kalibr validator](https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_camera_validator) evaluates mono and bidirectional inter-camera reprojection in original camera geometry. That approach also resolves a verified failure on this rig: OpenCV's fisheye boundary-based rectified viewport returned a focal length of **0.00192 px** despite fitted focal lengths near **587 px**, shrinking vertical errors to almost zero. Conversely, a finite perspective viewport amplifies peripheral vertical errors near its horizon. Changing viewport scale is therefore unsuitable for determining calibration quality.

`rectification.py` keeps OpenCV's baseline-aligned rotations and explicitly constructs a centred perspective projection at the mean fitted focal scale, with consistent P1/P2/Q. No arbitrary focal-ratio threshold or crop-dependent acceptance gate is introduced. Regression checks cover recovery of known 3D points from disparity and prove that shrinking the preview projection cannot hide bad stereo geometry. This remains a sampled geometric consistency decision, not a parameter-uncertainty or physical-depth guarantee. The projected board centre is calculated from its physical object-space centre, not the mean of distorted outline vertices.

A measured square scale controls translation units. A wrong uniform scale can leave pixel residuals small while making baseline and depth scale wrong. Physical measurement, target flatness, fixed camera mode/focus and rig stability remain necessary. Neither the application nor a combined USB frame establishes hardware exposure synchronization. Accuracy in millimetres at a specified distance needs a separate physical depth experiment with known geometry.

## Verification scope

Numerical fixtures exercise recovery of known fisheye intrinsics and relative pose, correspondence-sensitive rectification, and the complete nine-position validation transition. Retry regressions verify that seven passing areas survive re-evaluation, failed or missing areas are targeted, changed geometry invalidates old pass flags, corrected areas require a new confirmation, and a later favourable frame cannot conceal a saved failure. Rendered-board frames exercise actual detection, hold timing, PNG preservation, pause and duplicate suppression. Storage tests exercise failed writes and lock ownership; HTTP replay checks ensure reading does not alter the source session.

The browser was also checked against the physical stereo camera. The supplied PDFs were rendered and detected (70 and 35 corners respectively), and print layout was visually inspected. A closer physical 11 × 8 sheet matched the selected dictionary and legacy geometry. A complete physical guided calibration and independent depth accuracy measurement have not been performed by these automated checks.


## Session navigation and marker recovery

`Workspace` owns one active `Controller`. Plain GUI startup lists saved sessions
without opening the camera. Switching constructs the destination before releasing
the old session, pauses collection on the controller thread, closes its camera and
writer, and starts a fresh controller. This resets hold history, coverage, preview
maps and fit ownership together. Active fitting and diagnostic recording must
finish before a switch. A workspace revision binds browser commands and frames to
the selected session, preventing delayed requests from crossing session boundaries.
Discovery reads session metadata without acquiring writer locks. Existing CLI
resume/replay flags bypass the picker. Session controls live in `sessions.js`.

The picker uses a native HTML dialog for keyboard focus and Escape behavior,
following [MDN dialog guidance](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/dialog).

OpenCV documents `tryRefineMarkers` as board-based recovery of rejected ArUco
candidates before ChArUco interpolation in its
[ChArUco tutorial](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html).
It was previously disabled. On the retained 72-frame lower-right sequence,
enabling it improved native left corner counts from 51–68 (median 61) to 61–70
(median 66); median adjacent-frame ID changes fell from 10 to 4. The right retained
70 throughout. Recovered observations are measured from the image, not smoothed or
filled from prior frames. Candidate selection remains the strongest scale/contrast
observation; this can replace a candidate and change which IDs are present.

All 108 retained diagnostic pairs and 26 original captures pass the existing RMS
checks with recovery, in the evaluated modes. Additional edge points raise retake
live in-view vertical P95 from 0.590 to 0.656 px and maximum from 1.408 to 1.968 px;
paired common-point P95 remains approximately 0.59 px. More detections do not imply
lower all-point error. Full-mode P95 changes from 0.628 to 0.655 px. The model,
original captures and saved observations are unchanged. These are dataset-specific
checks, not certification of unsampled rays or physical depth accuracy. Disabling
marker corner refinement was also tested; it did not improve these observed dropouts,
so that unrelated setting was retained.


## Diagnostic storage and folder navigation

New `--inspection-log` runs default to `SESSION/diagnostics/runs/TIMESTAMP`.
An explicit directory remains supported. Folder navigation resolves app-owned
session directories and the current or latest matching diagnostic log. It reads
only metadata lines, outside the per-frame poll. A retained log under a session's
`diagnostics/` travels with that session, even if its original absolute metadata
path is now historical. The API uses the same local-origin and workspace revision
checks as session actions, and passes an absolute file URI to the desktop opener
without a shell. It does not issue capture or session-switch commands.

The retained lower-right sequence still has marker decoding dropouts after recovery.
All 276 missing left corner observations across 72 frames lack at least one of
the two adjacent markers, mostly at the board's right edge. Detected marker cells
(short quadrilateral side divided by seven modules) have a left 10th percentile
of 2.19 px versus 3.09 px right; left minimum is 1.49 px. This supports insufficient
local image detail at the oblique pose as a contributor, not a diagnosis of faulty
focus. The historical frames do not isolate focus, illumination, motion and
perspective. No extra rejection threshold, synthetic corners or temporal smoothing
was introduced. The numerical report is kept alongside the diagnostic run.


## Preview latency and CPU concurrency

The old browser loop waited 250 ms after fetching state, fetching an image and
decoding it, so video stayed below 4 fps even with a 30 fps camera. Measured images
also waited for serial left/right detection. Camera acquisition already ran on its
own thread and kept only the latest frame; there was no camera-frame backlog to
solve by adding another acquisition thread.

`live_preview.py` serves the most recent camera frame independently of the
controller's detection loop. HTTP rendering uses its own lock and a one-frame cache;
no work queue grows behind detection. The 1600-pixel-wide preview is presentation
only. Rectified maps directly produce the display size with the same field of view,
including resize's half-pixel centre convention. Original source pixels, saved
images, observations, fitted projections and validation thresholds are unchanged.
`video.js` allows one outstanding fetch/decode and drops responses after session or
projection changes. The default is smooth Live video; Detected frame and replay
retain exact image/measurement correspondence. Guidance still uses full-resolution
measurements; only the static target area is drawn on newer live video.

The two eye detectors now run in a persistent two-worker executor. Each has its
own OpenCV detector, and both finish before the next stereo pair is submitted,
even if one fails. Configuration changes and shutdown keep the same lifecycle.
The previous forced `cv2.setNumThreads(2)` was removed, allowing OpenCV's normal
available-CPU setting or an explicit environment override. OpenCV's
[Python wrappers release the GIL around native operations](https://github.com/opencv/opencv/blob/5.x/modules/python/src2/cv2_util.hpp),
so these native detector calls can overlap. Its
[thread controls](https://docs.opencv.org/4.13.0/db/de0/group__core__utils.html) and
[oversubscription guidance](https://docs.opencv.org/doc/doxygen/html/de/d55/group__core__parallel__backend.html)
motivate benchmarking the combination instead of assuming every extra thread helps.

On the installed OpenCV 5.0.0 pthreads build, a Core Ultra 7 270K Plus has 24
available CPUs. A shuffled 12-configuration benchmark used 1, 2, 4, 8, 12 and 24
OpenCV threads, with serial or concurrent eyes, over six recorded stereo pairs.
Median native detection improved from 147.8 ms (serial eyes, 2 threads) to 63.8 ms
(concurrent eyes, 24 threads), with identical IDs and coordinates in every mode.
Eight OpenCV threads plus concurrent eyes took 65.8 ms. Rechecking all 72 retained
lower-right images in live mode and six full-mode samples produced bit-identical
IDs, point coordinates and quality records. This build has no CUDA support for the
installed detector; no GPU implementation or alternate detection algorithm was
introduced. Benchmark scripts and results are under the session's
`diagnostics/performance/`. Throughput depends on scene, display and machine load;
receipt-to-encode ages are software timings, not sensor-exposure latency.
