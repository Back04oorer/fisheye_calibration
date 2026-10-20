"""Single-writer workflow controller; camera and numerical fit have separate owners."""

import copy
import logging
import multiprocessing
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from .calibration import Calibration
from .camera import CameraSource, ReplaySource
from .config import BoardConfig, CameraConfig
from .guidance import (
    TRAINING_GOALS,
    VALIDATION_GOALS,
    Hold,
    achieved,
    capture_ready,
    describe,
    live_adjustment,
    ready_to_fit,
)
from .inspection import inspect_observation
from .inspection_log import InspectionLog
from .live_preview import LivePreview
from .observations import Detector, Observation, usable
from .preview import Preview
from .session import Session, fit_snapshot
from .validation import coverage


class Controller:
    def __init__(
        self,
        board=None,
        camera=None,
        *,
        session=None,
        replay=False,
        output="sessions",
        inspection_log=None,
        inspection_images=False,
        stream_only=False,
    ):
        if inspection_images and not inspection_log:
            raise ValueError("Diagnostic images require an inspection log directory.")
        if stream_only and inspection_log:
            raise ValueError("Stream-only mode cannot record board inspection.")
        self.session = session
        self.replay = replay
        self.handeye_active = False
        # Separate from capture pause: manual inspection may need detection
        # while paused. This opt-in mode suspends both board detector loops.
        self.stream_only = bool(stream_only)
        self.detection_lock = threading.RLock()
        self.board = session.board if session else board or BoardConfig()
        self.camera = session.camera if session else camera or CameraConfig()
        self.output = Path(output)
        self.phase = "replay" if replay else "ready"
        if session and not replay:
            self.phase = session.data["phase"]
            if self.phase in ("fitting", "ready", "paused"):
                self.phase = "collecting"
        self.paused = bool(stream_only or (session and not replay))
        previous = session.candidate if session else None
        if not previous and session and session.data.get("training_request"):
            previous = next(reversed(session.data["candidates"]), None)
        self.model = Calibration.from_dict(previous["model"]) if previous else None
        self.last_fit_goals = set()
        self.fit_future = None
        self.final_fit = False
        self.pool = None
        self.source = ReplaySource(session) if replay else CameraSource(self.camera)
        self.detection_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='charuco-eye')
        self.detector = Detector(self.board, executor=self.detection_pool)
        self.hold = Hold()
        self.renderer = Preview()
        self.video = LivePreview()
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.public = {}
        self.previews = {}
        self.error = None
        self.message = ("Stream only. Board detection is suspended; camera video stays live."
                        if stream_only else
                        "Show your selected board to both cameras, then start collection.")
        self.hold_progress = 0.0
        self.last_capture_at = None
        self.signatures = []
        self.done_goals = set()
        self.reference_scale = None
        self.last_capture = None
        self.observation = None
        self.inspection = None
        self.signature = None
        self.capture_eligible = False
        self.details = []
        self.sequence = -1
        self.detection_fps = 0.0
        if session and session.candidate and not replay:
            self.recheck_candidate()
            self.paused = True  # Resuming a session still starts with capture paused.
        else:
            self.refresh_progress()
        self.inspection_log = None
        if inspection_log:
            if not session or not session.candidate or (not replay and self.phase != "done"):
                raise ValueError("Inspection recording requires a completed calibration session.")
            self.inspection_log = InspectionLog(
                session.path / "diagnostics" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if inspection_log == "auto" else inspection_log,
                {
                    "session": str(session.path),
                    "candidate": session.candidate["id"],
                    "board": self.board.to_dict(),
                    "camera": self.camera.to_dict(),
                    "model": self.model.to_dict(),
                    "source": "recorded" if replay else "live",
                    "scope": "Every processed detection frame, including zero detections. Camera acquisition may run faster than detection. No fit captures are written.",
                },
                save_images=inspection_images,
            )
        self.publish()
        self.thread = threading.Thread(target=self._run, name="calibration-controller", daemon=True)

    def start(self):
        self.source.start()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.source.close()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError("The previous session is still stopping. Please retry shortly.")
        self.detection_pool.shutdown(wait=True, cancel_futures=True)
        if self.inspection_log:
            self.inspection_log.close()
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
        if self.session:
            self.session.close()

    def state(self):
        with self.lock:
            state = copy.deepcopy(self.public)
            # CameraSource opens asynchronously after initial publication.
            # Stream-only deliberately skips detector publication, so read its
            # tiny hardware metadata snapshot directly on each state request.
            state["camera_actual"] = dict(self.source.actual)
            return state

    def image(self, frame_id, mode):
        with self.lock:
            return self.previews.get(frame_id, {}).get(mode)

    def live_image(self, mode, after=-1):
        with self.lock:
            source, camera, model = self.source, self.camera, self.model
            target = None if self.handeye_active else (self.public.get('goal') or {}).get('target')
        return self.video.image(source, camera, model, tuple(target) if target else None, mode, after)

    def command(self, action, payload=None):
        future = Future()
        self.commands.put((action, payload or {}, future))
        return future.result(timeout=30)

    def resume_detection(self):
        """Restore observation only; capture stays disarmed until requested."""
        self.stream_only = False
        if self.message.startswith("Stream only."):
            self.message = "Board detection resumed. Capture is paused."
        self.publish()

    def _suspend_detection(self):
        # The same lock brackets hand-eye processing. Acknowledgment therefore
        # means an in-flight observation can no longer republish stale results.
        self.stream_only = True  # Prevent another detector pass while draining the current one.
        with self.detection_lock:
            self.paused = True
            self.hold.reset()
            self.hold_progress = 0.0
            self.observation = self.inspection = self.signature = None
            self.capture_eligible = False
            self.details = []
            self.detection_fps = 0.0
            self.sequence = -1
            self.last_capture_at = None
            with self.lock:
                self.previews.clear()
            self.message = "Stream only. Board detection is suspended; camera video stays live."

    def _command(self, action, payload):
        if action == "stream_only":
            if self.inspection_log and self.inspection_log.active:
                raise ValueError("Stop diagnostic recording before suspending board detection.")
            self._suspend_detection()
            return
        if action == "resume_detection":
            self.resume_detection()
            return
        if action == "stop_inspection":
            if self.inspection_log:
                self.inspection_log.close()
            return
        if self.inspection_log and self.inspection_log.active and action not in ("pause", "resume"):
            raise ValueError(
                "Stop diagnostic recording before changing calibration or camera state."
            )
        if action == "prepare_switch":
            if self.fit_future:
                raise ValueError("A fit is still running. Switch sessions when it finishes.")
            self.paused = True
            if self.replay:
                self.source.playing.clear()
            self.hold.reset()
            self.hold_progress = 0.0
            return
        if self.replay:
            if action == "pause":
                self.source.playing.clear()
                self.paused = True
            elif action in ("start", "resume"):
                self.resume_detection()
                self.source.playing.set()
                self.paused = False
            else:
                raise ValueError("Replay is read-only. Only pause and resume are available.")
            return
        if action == "configure":
            if self.session:
                raise ValueError(
                    "Board and camera settings are fixed for this session. Open a new session to change them."
                )
            board = BoardConfig(**payload["board"])
            camera = CameraConfig(**payload["camera"])
            self.source.close()
            self.board, self.camera = board, camera
            self.source = CameraSource(camera)
            self.detector = Detector(board, executor=self.detection_pool)
            self.source.start()
            self.sequence = -1
            self.observation = None
            self.inspection = None
            self.signature = None
            self.capture_eligible = False
            self.details = []
            with self.lock:
                self.previews.clear()
            self.error = None
        elif action in ("start", "resume"):
            if self.source.error:
                raise ValueError(self.source.error)
            if self.session is None:
                if self.observation is None:
                    raise ValueError("Wait for a camera frame before starting.")
                valid, reason = usable(self.observation, self.board)
                if not valid:
                    raise ValueError(reason)
                name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                self.session = Session.create(self.output / name, self.board, self.camera)
                self.session.data["camera_actual"] = self.source.actual
                self.phase = "collecting"
                self.session.phase(self.phase)
            self.paused = False
            self.error = None
        elif action == "pause":
            self.paused = True
        elif action == "reconnect":
            self.source.close()
            self.source = CameraSource(self.camera)
            self.source.start()
            self.sequence = -1
            self.observation = None
            self.inspection = None
            self.signature = None
            self.capture_eligible = False
            self.details = []
            with self.lock:
                self.previews.clear()
            self.paused = True
            self.error = None
        elif action == "undo":
            if self.phase != "collecting" or self.fit_future:
                raise ValueError("Undo is available during training collection only.")
            self.session.undo()
            self.refresh_progress()
        elif action == "fit":
            if self.session is None or self.phase != "collecting":
                raise ValueError("Collect training images before fitting.")
            self.start_fit(final=True)
        elif action == "more":
            if self.session is None or self.fit_future:
                raise ValueError("No completed fitting round to extend.")
            self.begin_refinement()
        else:
            raise ValueError(f"Unknown action: {action}")
        if action in ("start", "resume", "configure", "more"):
            self.resume_detection()
        self.hold.reset()
        self.hold_progress = 0.0

    def begin_refinement(self):
        """Keep failed checks, request new training where they failed, then refit."""
        candidate = self.session.candidate
        report = candidate.get("report") if candidate else None
        failures = [v for v in (report or {}).get("validation_views", []) if not v["passed"]]
        revisit = {key.removeprefix("check-") for view in failures for key in view.get("goals", [])}
        requested = [g.key for g in TRAINING_GOALS if g.key in revisit]
        reasons = dict.fromkeys(
            reason
            for view in failures
            for reason in view.get("failed_checks", [view.get("error", "Fresh check failed")])
        )
        reason = "; ".join(reasons) or (
            "Training residuals exceed the 1 px target"
            if report and not report["fit_passed"]
            else "Collect another set of varied training poses"
        )
        self.session.new_round(requested or [g.key for g in TRAINING_GOALS], reason=reason)
        self.phase = "collecting"
        self.paused = False
        self.error = None
        self.last_fit_goals = set()
        self.message = "Automatic retry: follow the guide for a new training pose."
        self.hold.reset()
        self.hold_progress = 0.0
        self.refresh_progress()

    @property
    def validating(self):
        return (
            (self.phase in ("validating", "done", "needs_attention") or self.replay)
            and self.session
            and self.session.candidate
        )

    @property
    def goals(self):
        return VALIDATION_GOALS if self.validating else TRAINING_GOALS

    @property
    def next_goal(self):
        if self.session and not self.validating:
            pending = set((self.session.data.get("training_request") or {}).get("goals", []))
            pending.difference_update(self.done_goals)
            if pending:
                return next(g for g in self.goals if g.key in pending)
        if self.validating:
            pending = {
                region["key"]
                for region in (self.session.candidate.get("report") or {}).get("regions", [])
                if region["status"] in ("failed", "confirm")
            }
            if pending:
                return next(g for g in self.goals if g.key in pending)
        return next((g for g in self.goals if g.key not in self.done_goals), None)

    def refresh_progress(self):
        self.signatures = []
        self.done_goals = set()
        if self.session is None:
            return
        training = self.session.observations("training")
        if training:
            try:
                first = describe(training[0], self.board, self.model)
                self.reference_scale = first["scale"] if first else None
            except (ValueError, cv2.error):
                self.reference_scale = None
        role = "validation" if self.validating else "training"
        for observation in self.session.observations(role):
            try:
                signature = describe(observation, self.board, self.model)
            except (ValueError, cv2.error):
                signature = None
            if signature:
                self.signatures.append(signature)
                if role == "training":
                    self.done_goals.update(achieved(signature, self.reference_scale))
        if role == "validation":
            report = self.session.candidate.get("report") or {}
            self.done_goals = {
                region["key"]
                for region in report.get("regions", [])
                if region["status"] == "passed"
            }
            # Read-only replay of an older report retains its recorded results.
            # Writable sessions are re-evaluated by recheck_candidate on startup.
            if "regions" not in report:
                passed = {v["name"] for v in report.get("validation_views", []) if v["passed"]}
                for observation in self.session.observations(role):
                    if observation.name in passed:
                        signature = describe(observation, self.board, self.model)
                        self.done_goals.update(achieved(signature, None, validation=True))
        # A refinement round must add new observations in the failed regions.
        # Old coverage must not immediately satisfy this request again.
        request = self.session.data.get("training_request")
        if role == "training" and request:
            required = set(request["goals"])
            self.done_goals.difference_update(required)
            previous = set(request["existing_captures"])
            for capture in self.session.captures("training"):
                if capture["id"] not in previous:
                    signature = describe(
                        Observation.from_dict(capture["observation"]), self.board, self.model
                    )
                    self.done_goals.update(
                        required.intersection(achieved(signature, self.reference_scale))
                    )

    def start_fit(self, final):
        if self.fit_future:
            raise ValueError("A fit is already running.")
        if not ready_to_fit(self.signatures, self.done_goals):
            raise ValueError("First cover the centre and tilt the board about both axes.")
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        self.final_fit = final
        self.last_fit_goals = self.done_goals.copy()
        self.fit_future = self.pool.submit(fit_snapshot, self.session.training_snapshot())
        self.phase = "fitting"
        self.session.phase(self.phase)
        self.message = "Fitting camera geometry. The preview remains live."
        self.hold.reset()

    def check_fit(self):
        if not self.fit_future or not self.fit_future.done():
            return
        future, self.fit_future = self.fit_future, None
        try:
            result = future.result()
            self.model = Calibration.from_dict(result["model"])
            self.error = None
            self.phase = "collecting"
            self.refresh_progress()
            if self.final_fit and len(self.done_goals) == len(TRAINING_GOALS):
                self.session.accept_fit(result)
                self.recheck_candidate()
            else:
                self.message = (
                    "Provisional geometry ready. Follow the remaining coverage and tilt goals."
                )
            self.session.phase(self.phase)
        except Exception as error:
            logging.getLogger(__name__).exception("Calibration worker operation failed")
            self.phase = "collecting"
            if self.final_fit:
                self.paused = True
                self.error = (
                    f"Fit could not finish: {error}. Your captures are saved. "
                    "Review the solver diagnostics before changing the capture plan."
                )
            else:
                # A small, diverse seed can still be numerically insufficient.
                # Continue the planned movements and retry as coverage increases.
                self.message = "Initial geometry is not ready. Continue the movement guide."
            self.session.phase(self.phase)

    def live_pose(self, observation):
        """A partial/unsolved live detection is a missed frame, not a session failure.

        Check the existing capture requirements before asking PnP for a pose.
        Keep solver failures recoverable here; stored-view validation remains strict.
        """
        valid, reason = usable(observation, self.board)
        if not valid:
            return None, reason
        try:
            signature = describe(observation, self.board, self.model)
        except (cv2.error, ValueError):
            logging.getLogger(__name__).debug("Live board pose unavailable", exc_info=True)
            signature = None
        if signature is None:
            return None, "Board pose unclear. Adjust the angle and keep corners visible."
        return signature, reason

    def process(self, packet):
        started = time.monotonic()
        if packet.record:
            observation = Observation.from_dict(packet.record["observation"])
            details = packet.record["quality"]
        else:
            observation, details = self.detector.observe(packet.image, self.camera.swap, full=False)
            observation.name = str(packet.sequence)
        self.observation, self.details = observation, details
        signature, self.message = self.live_pose(observation)
        collecting = (
            not self.replay and not self.paused and self.phase in ("collecting", "validating")
        )
        if collecting and signature:
            novel = self.pose_ready(signature)
            if novel:
                self.hold_progress, best = self.hold.update(
                    observation, packet.timestamp, packet.image, details
                )
                self.message = (
                    "Hold still…" if best is None else "Checking full-resolution corners…"
                )
                if best:
                    _, image, best_observation, _ = best
                    checked, quality = self.detector.observe(image, self.camera.swap, full=True)
                    selected_signature, reason = self.live_pose(checked)
                    if selected_signature:
                        goals = achieved(
                            selected_signature,
                            self.reference_scale,
                            validation=bool(self.validating),
                        )
                        # The full detector may recover a different set of corners and pose.
                        # Only this final observation can decide whether a capture is useful.
                        novel = self.pose_ready(selected_signature)
                        if novel:
                            self.session.append(
                                image,
                                checked,
                                role="validation" if self.validating else "training",
                                frame_id=int(best_observation.name),
                                goals=goals,
                                signature=selected_signature,
                                quality=quality,
                            )
                            self.last_capture = selected_signature
                            self.last_capture_at = time.monotonic()
                            self.message = "Captured. Move to the next pose."
                            self.refresh_progress()
                            self.after_capture()
                        else:
                            self.message = (
                                "This pose is already covered. Follow the next movement cue."
                            )
                    else:
                        self.message = reason
                        signature = None
                    self.hold.reset()
                    self.hold_progress = 0.0
            else:
                self.hold.reset()
                self.hold_progress = 0.0
                self.message = "This pose is already covered. Follow the next movement cue."
        else:
            self.hold.reset()
            self.hold_progress = 0.0
        # Recompute after a save: the guide must advance immediately instead of
        # asking the operator to hold a pose that has just been recorded.
        self.signature = signature
        self.inspection = inspect_observation(observation, self.board, self.model)
        self.inspection["source"] = "recorded" if packet.record else "live"
        self.inspection["frame_id"] = packet.sequence
        if self.inspection_log:
            self.inspection_log.frame(packet, self.inspection, details)
        self.capture_eligible = bool(signature and self.pose_ready(signature))
        previews = self.renderer.render(
            packet.image, observation, signature, self.next_goal, self.camera, self.model
        )
        self.sequence = packet.sequence
        self.detection_fps = 1.0 / max(time.monotonic() - started, 0.001)
        with self.lock:
            self.previews[packet.sequence] = previews
            while len(self.previews) > 4:
                del self.previews[next(iter(self.previews))]

    def pose_ready(self, signature):
        return capture_ready(
            signature,
            self.signatures,
            self.last_capture,
            self.done_goals,
            self.reference_scale,
            validation=bool(self.validating),
        )

    def recheck_candidate(self):
        """Keep only coverage that passes again under this frozen calibration."""
        report = self.session.validate()
        self.phase = "validating"
        self.refresh_progress()
        if not report["fit_passed"] or any(
            not view["passed"] for view in report["validation_views"]
        ):
            self.begin_refinement()
        else:
            self.phase = "done" if report["passed"] else "validating"
            self.message = "Saved checks re-evaluated. Follow the remaining movement cues."
            self.session.phase(self.phase)

    def after_capture(self):
        if self.validating:
            self.recheck_candidate()
        elif ready_to_fit(self.signatures, self.done_goals):
            complete = len(self.done_goals) == len(TRAINING_GOALS)
            if complete or (self.model is None and self.done_goals - self.last_fit_goals):
                self.start_fit(final=complete)

    def publish(self):
        candidate = self.session.candidate if self.session else None
        observations = self.session.observations("training") if self.session else []
        goal = self.next_goal
        request = (
            self.session.data.get("training_request")
            if self.session and self.phase in ("collecting", "fitting")
            else None
        )
        review = candidate
        if review is None and request:
            review = next(reversed(self.session.data["candidates"]), None)
        report = review.get("report") if review else None
        state = {
            "phase": self.phase,
            "paused": self.paused,
            "stream_only": self.stream_only,
            "replay": self.replay,
            "message": self.message,
            "error": self.error or self.source.error,
            "board": self.board.to_dict(),
            "camera": self.camera.to_dict(),
            "camera_actual": self.source.actual,
            "configured": self.session is not None,
            "session": str(self.session.path) if self.session else None,
            "frame_id": self.sequence,
            "detection_fps": self.detection_fps,
            "hold_progress": self.hold_progress,
            "just_captured": self.last_capture_at is not None
            and time.monotonic() - self.last_capture_at < 2,
            "adjustment": live_adjustment(
                goal,
                self.signature,
                self.reference_scale,
                self.done_goals,
                ready=self.capture_eligible,
                validation=bool(self.validating),
                previous=[*self.signatures, self.last_capture],
            ),
            "quality": self.details,
            "inspection": self.inspection,
            "inspection_recording": self.inspection_log.status() if self.inspection_log else None,
            "common_corners": len(self.observation.common()[0]) if self.observation else 0,
            "goal": {
                "key": goal.key,
                "title": goal.title,
                "instruction": goal.instruction,
                "kind": goal.kind,
                "target": goal.target,
            }
            if goal
            else None,
            "goals": [
                {"key": g.key, "title": g.title, "done": g.key in self.done_goals}
                for g in self.goals
            ],
            "training_count": len(observations),
            "refinement_reason": request.get("reason") if request else None,
            "can_fit": ready_to_fit(self.signatures, self.done_goals),
            "validation_count": len((report or {}).get("validation_views", [])),
            "saved_views": self.session.saved_views() if self.session else [],
            "coverage": {eye: coverage(observations, eye) for eye in ("left", "right")},
            "has_model": self.model is not None,
            "candidate": candidate["id"] if candidate else None,
            "report": report,
            "fit_rms": list(self.model.rms) if self.model else None,
        }
        with self.lock:
            self.public = state

    def _run(self):
        try:
            while not self.stop.is_set():
                while not self.commands.empty():
                    action, payload, future = self.commands.get_nowait()
                    try:
                        self._command(action, payload)
                        self.publish()
                        future.set_result(self.state())
                    except Exception as error:
                        logging.getLogger(__name__).exception("Calibration worker operation failed")
                        future.set_exception(error)
                self.check_fit()
                if self.handeye_active or self.stream_only:
                    # Hand-eye consumes this same source at native resolution. Keep
                    # acquisition/video live without running the old-board detector.
                    self.stop.wait(0.05)
                    continue
                if self.source.error:
                    self.paused = True
                    self.hold.reset()
                    self.hold_progress = 0.0
                    self.stop.wait(0.1)
                else:
                    packet = self.source.next(self.sequence)
                    if packet:
                        try:
                            with self.detection_lock:
                                if not self.stream_only:
                                    self.process(packet)
                        except Exception as error:
                            logging.getLogger(__name__).exception(
                                "Calibration worker operation failed"
                            )
                            self.error = f"Processing paused: {error}"
                            if self.inspection_log:
                                self.inspection_log.write(
                                    {
                                        "type": "processing_error",
                                        "frame_id": packet.sequence,
                                        "error": str(error),
                                    }
                                )
                            self.paused = True
                            self.hold.reset()
                            self.sequence = packet.sequence
                self.publish()
        except Exception as error:
            logging.getLogger(__name__).exception("Calibration worker operation failed")
            self.error = str(error)
            self.paused = True
            self.publish()
