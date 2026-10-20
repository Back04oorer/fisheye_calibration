"""Independent hand-eye runs sharing the controller's native camera packets."""

import copy
import hashlib
import json
import logging
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .folders import open_directory
from .guidance import describe
from .handeye import (
    COVERAGE_METHOD,
    CROSS_VALIDATION_METHOD,
    FIT_METHOD,
    L20_BOARD,
    POLICY,
    ArmKinematics,
    CaptureBlocked,
    camera_board_pose,
    inverse,
    novel,
    training_assessment,
    transform,
    validation_report,
)
from .handeye_guidance import (
    GUIDE_POLICY,
    GuidedHold,
    LocalGuidance,
    PositionThenTilt,
    pose_preview,
    requires_review,
)
from .handeye_kinematics import select_arm_model
from .handeye_recovery import load_prior_validation
from .inspection import inspect_observation
from .observations import Detector
from .preview import Preview
from .validation import board_pose

ROBOT_ROOT = Path("/home/pairlab/Desktop/rubin/nero-linkerhand-teleop")
ARM_URDF = ROBOT_ROOT / "agx_arm_urdf/nero/urdf/nero_description.urdf"
ARM_MODEL_PROFILES = ROBOT_ROOT / "calibration/arm_kinematics"
ROOT = Path(__file__).resolve().parents[1]
RECOVERABLE_CAPTURE_ISSUES = {
    "capture_rejected",
    "capture_timeout",
    "capture_interrupted",
    "capture_retry",
    "capture_wait",
    "training_excluded",
}


def utc():
    return datetime.now(UTC).isoformat()


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class PoseFeed:
    def __init__(self, side, path):
        self.side, self.path = side, path
        self.lock = threading.Lock()
        self.history = deque(maxlen=3000)
        self.error = None
        self.process = None
        self.thread = None
        self.stderr = None

    def start(self):
        env = ROBOT_ROOT / "scripts/env/nero_linkerhand_env.sh"
        python = ROBOT_ROOT / "env/nero_linkerhand_ros2_py312/bin/python3"
        if not env.is_file() or not python.is_file():
            self.error = "ROS feedback environment not found. See setup instructions."
            return
        # Arguments are shell-quoted paths, never values supplied by the browser.
        command = (
            f"source {shlex.quote(str(env))}; exec {shlex.quote(str(python))} -u "
            f"{shlex.quote(str(ROOT / 'scripts/handeye_feedback.py'))} --side {self.side}"
        )
        self.stderr = (self.path / "feedback.log").open("ab")
        self.process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True, name="handeye-feedback")
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.add(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue
            self.error = "Robot feedback reader stopped. Open session folder → feedback.log."
        except OSError as error:
            self.error = str(error)

    def add(self, row):
        now = time.monotonic()
        if (
            row["topic"] != f"/slave_{self.side}/joint_states_single"
            or not 0 <= now - row["received_monotonic"] <= POLICY["feedback_max_age_s"]
            or not 0 <= row["received_wall"] - row["stamp_s"] <= POLICY["feedback_max_age_s"]
        ):
            return
        row = dict(row)
        row["sample_monotonic"] = row["received_monotonic"] + row["stamp_s"] - row["received_wall"]
        with self.lock:
            if not self.history or row["stamp_s"] > self.history[-1]["stamp_s"]:
                self.history.append(row)

    def status(self):
        with self.lock:
            age = time.monotonic() - self.history[-1]["sample_monotonic"] if self.history else None
        return {
            "connected": age is not None and age <= POLICY["feedback_max_age_s"],
            "age_s": age,
            "topic": f"/slave_{self.side}/joint_states_single",
            "error": self.error,
        }

    def paired(self, timestamp, kinematics, *, through=None, require_stationary=True):
        """Pair measured joints; moving pairs are advisory unless stationarity is required.

        Live dwell can track the current pose while the preceding joint window
        settles. Recording always uses the default strict check through decision.
        Freshness, continuity, joint validity and skew apply in both modes.
        """
        with self.lock:
            rows = list(self.history)
        window = POLICY["stationary_window_s"]
        end = max(timestamp + 0.15, through) if through is not None else timestamp + 0.15
        relevant = [r for r in rows if timestamp - window - 0.1 <= r["sample_monotonic"] <= end]
        if len(relevant) < 5:
            raise ValueError("Waiting for fresh measured follower feedback.")
        times = np.array([r["sample_monotonic"] for r in relevant])
        if (
            times[0] > timestamp - window
            or times[-1] < timestamp
            or np.diff(times).max() > POLICY["pair_max_skew_s"]
        ):
            raise ValueError("Hold the wrist still; waiting for a continuous feedback window.")
        if through is not None and not 0 <= through - times[-1] <= POLICY["pair_max_skew_s"]:
            raise ValueError(
                "Waiting for current measured arm feedback through the capture decision."
            )
        nearest = min(relevant, key=lambda r: abs(r["sample_monotonic"] - timestamp))
        skew = abs(nearest["sample_monotonic"] - timestamp)
        if skew > POLICY["pair_max_skew_s"]:
            raise ValueError("Camera and measured arm pose are too far apart in time.")
        values = []
        for r in relevant:
            if len(r["names"]) != len(r["positions"]) or len(set(r["names"])) != len(r["names"]):
                raise ValueError("Joint feedback names/positions are invalid.")
            q = dict(zip(r["names"], r["positions"]))
            if any(n not in q or not np.isfinite(q[n]) for n in kinematics.names):
                raise ValueError("Measured follower feedback must contain every arm joint.")
            values.append([q[n] for n in kinematics.names])
        ranges = np.ptp(values, axis=0)
        joint_index = int(np.argmax(ranges))
        motion = float(ranges[joint_index])
        if require_stationary and motion > POLICY["stationary_joint_range_rad"]:
            limit = POLICY["stationary_joint_range_rad"]
            raise CaptureBlocked(
                "motion",
                "Wrist is moving. "
                f"{kinematics.names[joint_index]} varied {np.degrees(motion):.2f}° "
                f"over {times[-1] - times[0]:.1f} s; limit {np.degrees(limit):.2f}°. Pause briefly to capture.",
                motion_joint=kinematics.names[joint_index],
                joint_range_rad=motion,
                joint_range_limit_rad=limit,
            )
        # Only the paired sample needs FK; the complete window checks joint motion.
        pose = kinematics.pose(nearest["names"], nearest["positions"])
        return pose, {
            "nearest": nearest,
            "window": relevant,
            "pair_skew_s": skew,
            "joint_range_rad": motion,
            "stationary": motion <= POLICY["stationary_joint_range_rad"],
            "stationary_through_monotonic": float(times[-1]),
            "decision_feedback_age_s": float(through - times[-1]) if through is not None else None,
            "joint_limit_diagnostics": kinematics.limit_diagnostics(
                nearest["names"], nearest["positions"]
            ),
            "fk_policy": (
                "raw_feedback_unmodified; qualified_joint_angle_map_for_fk_only; nominal_limits_are_diagnostics_only"
                if getattr(kinematics, "joint_angle_maps", {})
                else "measured_angles_unmodified; nominal_limits_are_diagnostics_only"
            ),
            "timing_scope": "Host receipt for camera; stamped measured joints. Stationary capture; exposure synchronization not established.",
        }

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.thread:
            self.thread.join(timeout=2)
        if self.stderr:
            self.stderr.close()


class HandEye:
    def __init__(self, workspace, *, feed_factory=PoseFeed):
        self.workspace = workspace
        self.feed_factory = feed_factory
        self.lock = threading.RLock()
        self.active = False
        self.stop_event = threading.Event()
        self.thread = None
        self.feed = None
        self.path = None
        self.data = None
        self.measurement = None
        self.preview = None
        self.previews = {}
        self.error = None
        self.pending = None
        self.busy = False
        self.assessing = False
        self.readiness = None
        self.assessment_generation = 0
        self.assessment_future = None
        self.assessment_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="handeye-assessment"
        )
        self.auto_capture = False
        self.retry_once = False
        self.guide_skip = 0
        self.hold = GuidedHold()
        self.position_guide = PositionThenTilt()
        self.local_guide = LocalGuidance()
        self.placement_local_guide = LocalGuidance()
        self.placement_position_guide = PositionThenTilt()
        self.placement = self.placement_context = None
        self.position_anchor = None
        self.last_capture = None
        self.controller = None
        self.revision = None
        self.message = "Open a completed stereo calibration, then start a hand–eye session."

    def runs(self):
        with self.workspace.access() as c:
            parent = c.session.path / "handeye" if c and c.session else None
        result = []
        if parent:
            for path in sorted(parent.glob("*/run.json"), reverse=True):
                if (path.parent / "archive.json").is_file():
                    continue
                try:
                    d = json.loads(path.read_text())
                    result.append(
                        {
                            "id": path.parent.name,
                            "side": d["side"],
                            "samples": len(d["samples"]),
                            "created_at": d["created_at"],
                        }
                    )
                except (OSError, ValueError, KeyError):
                    continue
        return result

    def begin(self, side, revision, run_id=None):
        if side not in ("right", "left"):
            raise ValueError("Choose the left or right follower arm.")
        with self.workspace.access(revision) as c, self.lock:
            if self.active or self.busy:
                raise ValueError("Finish or pause this hand–eye session first.")
            if not c or c.replay or not c.session or not c.model or not c.session.candidate:
                raise ValueError("Open a saved stereo calibration in live mode first.")
            if c.fit_future or (c.inspection_log and c.inspection_log.active):
                raise ValueError("Finish stereo fitting or diagnostic recording first.")
            if not (c.session.candidate.get("report") or {}).get("passed"):
                raise ValueError(
                    "The selected stereo calibration must pass its geometric checks first."
                )
            if not c.paused:
                raise ValueError("Pause stereo collection before starting hand–eye mode.")
            if tuple(c.model.image_size) != tuple(c.camera.eye_size):
                raise ValueError("Camera resolution does not match the saved calibration.")
            arm_model, arm_profile = select_arm_model(ARM_URDF, side, ARM_MODEL_PROFILES)
            self.kinematics = ArmKinematics(arm_model)
            calibration = (
                c.session.path / "results" / c.session.candidate["id"] / "stereo_params.npz"
            )
            fingerprint = hashlib.sha256(calibration.read_bytes()).hexdigest()
            parent = c.session.path / "handeye"
            if run_id:
                if run_id not in {r["id"] for r in self.runs()}:
                    raise ValueError("Select a hand–eye run belonging to this stereo session.")
                path = parent / run_id
                data = json.loads((path / "run.json").read_text())
                if (
                    data["calibration_sha256"] != fingerprint
                    or data["side"] != side
                    or data["urdf_sha256"] != self.kinematics.sha256
                    or data["board"] != L20_BOARD.to_dict()
                    or data["policy"] != POLICY
                ):
                    raise ValueError("Run provenance or capture policy changed; start a new run.")
            else:
                path = parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path.mkdir(parents=True, exist_ok=False)
                (path / "frames").mkdir()
                data = {
                    "schema": "fixed_camera_handeye.v1",
                    "created_at": utc(),
                    "side": side,
                    "board": L20_BOARD.to_dict(),
                    "camera": c.camera.to_dict(),
                    "camera_actual": dict(c.source.actual),
                    "stereo_session": str(c.session.path),
                    "stereo_candidate": c.session.candidate["id"],
                    "calibration_sha256": fingerprint,
                    "calibration": c.model.to_dict(),
                    "urdf": str(arm_model),
                    "urdf_sha256": self.kinematics.sha256,
                    "kinematic_calibration": arm_profile,
                    "base_frame": f"{side}/base_link",
                    "tool_frame": f"{side}/link7",
                    "feedback_topic": f"/slave_{side}/joint_states_single",
                    "robot_commands": "NONE; subscriber only",
                    "policy": dict(POLICY),
                    "software": {"opencv": cv2.__version__, "numpy": np.__version__},
                    "source_sha256": {
                        name: hashlib.sha256(
                            (Path(__file__).parent / name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "handeye.py",
                            "handeye_runtime.py",
                            "handeye_solver.py",
                            "handeye_guidance.py",
                            "handeye_information.py",
                            "handeye_kinematics.py",
                            "handeye_configuration.py",
                            "handeye_refinement.py",
                            "detection.py",
                            "observations.py",
                            "validation.py",
                        )
                    },
                    "stage": "training",
                    "samples": [],
                    "solution": None,
                    "report": None,
                }
                data["arm_model_file"] = "arm_model.json" if self.kinematics.joint_angle_maps else "arm_model.urdf"
                data["arm_model_format"] = self.kinematics.model_format
                (path / data["arm_model_file"]).write_bytes(arm_model.read_bytes())
                (path / "stereo_params.npz").write_bytes(calibration.read_bytes())
                save_json(path / "run.json", data)
            data["prior_validation_evidence"] = load_prior_validation(data, path)
            self.path, self.data = path, data
            self.controller, self.revision = c, revision
            self.model = c.model
            if (data.get("capture_issue") or {}).get("status") in (
                "retry_pending",
                "retry_waiting",
            ):
                self._capture_issue(
                    "capture_interrupted",
                    "Recording is paused after reopening this session. Start automatic capture to resume.",
                    retryable=self._issue_retryable(data.get("capture_issue")),
                )
            if (
                data["solution"]
                and data["stage"] in ("checking", "complete")
                and (
                    (data.get("readiness") or {}).get("coverage_method") != COVERAGE_METHOD
                    or (data.get("readiness") or {}).get("cross_validation_method")
                    != CROSS_VALIDATION_METHOD
                    or data["solution"].get("coverage_method") != COVERAGE_METHOD
                    or data["solution"].get("fit_method") != FIT_METHOD
                )
            ):
                previous = {
                    key: copy.deepcopy(data.get(key))
                    for key in ("stage", "solution", "report", "readiness")
                }
                save_json(
                    path / f"qualification_before_reassessment_{uuid.uuid4().hex[:8]}.json",
                    previous,
                )
                self._invalidate_export(
                    "Training qualification used an older coverage or fit method."
                )
                if (path / "result.json").exists():
                    (path / "result.json").replace(
                        path / f"result_before_reassessment_{uuid.uuid4().hex[:8]}.json"
                    )
                data.update(stage="training", solution=None, report=None, readiness=None)
                self.persist()
                self.event(
                    "training_qualification_invalidated",
                    current_method=COVERAGE_METHOD,
                    current_fit_method=FIT_METHOD,
                    previous_method=(previous["readiness"] or {}).get("coverage_method"),
                )
            if data["solution"] and data["stage"] in ("checking", "complete"):
                previous_report = data["report"]
                report = validation_report(
                    data["samples"],
                    data["solution"],
                    self.model,
                    prior_validation=data.get("prior_validation_evidence"),
                )
                if report != previous_report:
                    save_json(
                        path / f"validation_before_reassessment_{uuid.uuid4().hex[:8]}.json",
                        {
                            "report": previous_report,
                            "stage": data["stage"],
                            "reassessed_at": utc(),
                        },
                    )
                data["report"] = report
                data["stage"] = "complete" if report["passed"] else "checking"
                self.export_result()
                self.persist()
                self.event("validation_reassessed", passed=report["passed"], report=report)
            self.measurement, self.preview, self.error, self.pending = None, None, None, None
            self.previews.clear()
            self.auto_capture = False
            self.retry_once = False
            self.readiness = copy.deepcopy(data.get("readiness"))
            self.assessing = False
            self.guide_skip = 0
            self.hold.reset()
            self.position_guide = PositionThenTilt()
            self.local_guide = LocalGuidance()
            self.placement_local_guide = LocalGuidance()
            self.placement_position_guide = PositionThenTilt()
            self.placement = self.placement_context = None
            self.position_anchor = None
            self.last_capture = None
            self.feed = self.feed_factory(side, path)
            self.feed.start()
            self.active = True
            self.stop_event.clear()
            c.handeye_active = True
            c.resume_detection()
            self.message = "Checking board and measured arm feedback. You retain teleop control."
            self.event(
                "started",
                resumed=bool(run_id),
                guidance_policy=dict(GUIDE_POLICY),
                guidance_sha256=hashlib.sha256(
                    Path(__file__).with_name("handeye_guidance.py").read_bytes()
                ).hexdigest(),
                fk_policy="measured_angles_unmodified; nominal_limits_are_diagnostics_only",
            )
            self.thread = threading.Thread(
                target=self._run, daemon=True, name="handeye-measurement"
            )
            self.thread.start()
            if self.data["stage"] == "training":
                self._schedule_assessment()
        return self.state()

    def event(self, kind, **fields):
        if self.path:
            with (self.path / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"utc": utc(), "event": kind, **fields}, allow_nan=False) + "\n"
                )

    def persist(self):
        save_json(self.path / "run.json", self.data)

    @staticmethod
    def _issue_retryable(issue):
        return bool(
            issue and issue.get("retryable", issue.get("kind") in RECOVERABLE_CAPTURE_ISSUES)
        )

    def _capture_issue(self, kind, message, *, status="retry_required", retryable=None, **details):
        """Persist an attempted-capture outcome, never ordinary positioning advice."""
        if self.data is None:
            return
        self.data["capture_issue"] = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "status": status,
            "retryable": kind in RECOVERABLE_CAPTURE_ISSUES if retryable is None else retryable,
            "message": message,
            "created_at": utc(),
            "updated_at": utc(),
            "details": details,
        }
        self.message = message
        if status == "retry_required":
            self.auto_capture = False
            self.retry_once = False
            self.hold.reset()
        try:
            self.persist()
            self.event("capture_issue", issue=self.data["capture_issue"])
        except OSError as error:
            # A full/unwritable disk cannot persist an error about itself. Keep
            # the issue visible in memory and report this additional failure.
            logging.getLogger(__name__).exception("Could not persist capture retry status")
            self.data["capture_issue"]["details"]["status_write_error"] = str(error)
            self.auto_capture = False
            self.retry_once = False
            self.data["capture_issue"].update(status="retry_required", retryable=False)

    def _wait_for_automatic_retry(self, reason, *, kind="capture_wait", once=False):
        """Use the existing guided dwell; waiting frames do not write new issues."""
        if not self.active or self.data["stage"] not in ("training", "checking"):
            return False
        if requires_review(self._recording_review()):
            return False
        issue = self.data.get("capture_issue")
        if issue and issue["status"] != "resolved" and not self._issue_retryable(issue):
            return False
        self.pending = None
        self.auto_capture = True
        self.retry_once = once
        if issue and issue["status"] == "retry_waiting":
            return True
        self.hold.reset()
        self._capture_issue(
            kind,
            "Will retry automatically when the board and measured arm are ready. " + reason,
            status="retry_waiting",
            retryable=True,
        )
        return True

    def _cancel_capture_retry(self, message):
        was_waiting = bool(self.pending or self.auto_capture)
        self.pending = None
        self.auto_capture = self.retry_once = False
        self.hold.reset()
        issue = self.data.get("capture_issue") if self.data else None
        if issue and issue["status"] in ("retry_waiting", "retry_pending"):
            issue.update(status="retry_required", updated_at=utc(), message=message)
            self.persist()
            self.event("capture_retry_cancelled", issue_id=issue["id"], reason=message)
        elif was_waiting and not (issue and issue["status"] == "retry_required"):
            self._capture_issue("capture_interrupted", message, retryable=True)

    def _resolve_capture_issue(self, sample_id):
        issue = self.data.get("capture_issue")
        if issue and issue["status"] != "resolved":
            issue.update(status="resolved", updated_at=utc(), resolved_sample_id=sample_id)
            self.event("capture_issue_resolved", issue_id=issue["id"], sample_id=sample_id)

    def _expire_pending(self, reason):
        if self.pending is not None and time.monotonic() - self.pending >= 10:
            self.pending = None
            if self._wait_for_automatic_retry(reason, kind="capture_timeout", once=self.retry_once):
                return True
            self._capture_issue(
                "capture_timeout",
                "No pose was saved within 10 seconds. "
                + reason
                + " Show the full board, hold the wrist still, then retry.",
                retryable=False,
                last_reason=reason,
            )
            return True
        return False

    def _failed_training_ids(self):
        if not self.data or self.data["stage"] != "training" or not self.readiness:
            return []
        active = {
            s["id"]
            for s in self.data["samples"]
            if s["role"] == "training" and not s.get("excluded")
        }
        fit = self.readiness.get("fit") or []
        # Refuse a partial/stale assessment rather than guessing which unseen
        # records should be removed. LOO and independent checks never enter.
        if not active.issubset({row["id"] for row in fit}):
            return []
        failed = {row["id"] for row in fit if row.get("passed") is False} & active
        return [
            s["id"] for s in self.data["samples"] if s["role"] == "training" and s["id"] in failed
        ]

    def _exclude_failed_training(self):
        if self.data["stage"] != "training" or self.pending or self.assessing:
            raise ValueError(
                "Wait for a completed training assessment and no pending capture before excluding poses."
            )
        failed = self._failed_training_ids()
        if not failed:
            raise ValueError(
                "The latest full-fit assessment has no failed active training poses to exclude."
            )
        original = copy.deepcopy(self.data)
        timestamp = utc()
        snapshot = f"assessment_before_exclusion_{uuid.uuid4().hex[:12]}.json"
        save_json(
            self.path / snapshot,
            {
                "utc": timestamp,
                "action": "exclude_failed_training",
                "criterion": "current_full_fit",
                "excluded_ids": failed,
                "assessment": copy.deepcopy(self.readiness),
                "active_training_ids": [
                    s["id"]
                    for s in self.data["samples"]
                    if s["role"] == "training" and not s.get("excluded")
                ],
                "policy": copy.deepcopy(self.data["policy"]),
                "calibration_sha256": self.data["calibration_sha256"],
                "urdf_sha256": self.data["urdf_sha256"],
                "prior_validation_evidence": copy.deepcopy(
                    self.data.get("prior_validation_evidence")
                ),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
        for sample in self.data["samples"]:
            if sample["role"] == "training" and sample["id"] in failed:
                sample.update(
                    excluded=True,
                    excluded_at=timestamp,
                    exclusion_reason="Operator excluded a failed current full-fit prediction.",
                    exclusion_assessment=snapshot,
                )
        self.auto_capture = False
        self.hold.reset()
        self.guide_skip = 0
        self.readiness = self.data["readiness"] = None
        try:
            self.persist()
        except OSError:
            self.data, self.readiness = original, original.get("readiness")
            raise
        remaining = sum(
            s["role"] == "training" and not s.get("excluded") for s in self.data["samples"]
        )
        self.event(
            "failed_training_excluded",
            sample_ids=failed,
            assessment=snapshot,
            remaining_training_count=remaining,
        )
        self.auto_capture = True
        self.retry_once = False
        self._capture_issue(
            "training_excluded",
            f"{len(failed)} poses excluded; {remaining} remain. Automatic collection will continue after assessment; hold a comfortable visible pose.",
            status="retry_waiting",
            retryable=True,
            excluded_ids=failed,
            remaining_training_count=remaining,
            assessment=snapshot,
        )
        self._schedule_assessment()

    def _schedule_assessment(self):
        """Called under the state lock; expensive fitting runs on one worker.

        Generation tokens make pause/resume, exclusions and newer captures
        invalidate an old result. A stale worker never writes to a new run.
        Only a small, immutable geometry snapshot enters the solver: feedback
        windows and camera images remain on their original capture records.
        """
        self.assessment_generation += 1
        generation, path = self.assessment_generation, self.path
        if self.assessment_future:
            self.assessment_future.cancel()
        samples = [
            copy.deepcopy(
                {
                    key: s[key]
                    for key in ("id", "role", "T_base_tool", "T_camera_board", "observation")
                }
            )
            for s in self.data["samples"]
            if s["role"] == "training" and not s.get("excluded")
        ]
        self.assessing = True
        self.hold.reset()
        self.assessment_future = self.assessment_executor.submit(
            training_assessment,
            samples,
            self.model,
            prior_validation=copy.deepcopy(self.data.get("prior_validation_evidence")),
        )
        self.assessment_future.add_done_callback(
            lambda future: self._assessment_finished(future, generation, path)
        )

    def _assessment_finished(self, future, generation, path):
        if future.cancelled():
            return
        try:
            readiness = future.result()
            error = None
        except Exception as failure:
            logging.getLogger(__name__).exception("Hand–eye accuracy assessment failed")
            readiness, error = None, str(failure)
        with self.lock:
            if (
                not self.active
                or self.stop_event.is_set()
                or generation != self.assessment_generation
                or path != self.path
                or self.data["stage"] != "training"
            ):
                return
            if error:
                self.readiness = {
                    "ready": False,
                    "reason": "Accuracy assessment failed: " + error,
                    "solution": None,
                }
                self.auto_capture = False
                self.event("assessment_failed", error=error)
            else:
                self.readiness = readiness
                previous_method = (self.data.get("readiness") or {}).get("coverage_method")
                if readiness.get("coverage_method") != previous_method:
                    self.event(
                        "assessment_method_changed",
                        previous=previous_method,
                        current=readiness.get("coverage_method"),
                        source_sha256=hashlib.sha256(
                            Path(__file__).with_name("handeye.py").read_bytes()
                        ).hexdigest(),
                    )
                if requires_review(readiness) and self.auto_capture:
                    self.auto_capture = False
                    self.hold.reset()
                    self.event("auto_stopped", reason="accuracy_review", detail=readiness["reason"])
            self.data["readiness"] = copy.deepcopy(self.readiness)
            self.persist()
            self.message = self.readiness["reason"]
            if error or requires_review(self.readiness):
                self._capture_issue(
                    "training_accuracy",
                    self.readiness["reason"]
                    + " Capture one diagnostic pose only after reviewing the cause. Automatic recording remains stopped; existing failed evidence remains recorded.",
                    failed_training_ids=self._failed_training_ids(),
                )
            else:
                issue = self.data.get("capture_issue")
                if issue and issue["kind"] == "training_accuracy" and issue["status"] != "resolved":
                    # Reassessment may now ask for missing coverage. Preserve
                    # the prior issue in the event log without leaving a stale
                    # recording lock or claiming the calibration has passed.
                    issue.update(
                        status="resolved", updated_at=utc(),
                        resolution="training_reassessed", resolution_reason=self.readiness["reason"],
                    )
                    self.persist()
                    self.event(
                        "training_accuracy_review_cleared", issue_id=issue["id"],
                        ready=self.readiness["ready"], reason=self.readiness["reason"],
                    )
            self.event("training_assessed", readiness=self.readiness)
            if self.readiness["ready"] and not self.pending:
                self._freeze_candidate(self.readiness["solution"])
            # Completion is visible only after the result and stage are saved.
            self.assessing = False

    def _freeze_candidate(self, candidate):
        """Freeze only an independently assessed training fit, never its checks."""
        result = copy.deepcopy(candidate)
        result["id"] = uuid.uuid4().hex[:12]
        result["base_frame"] = self.data["base_frame"]
        result["tool_frame"] = self.data["tool_frame"]
        result["camera_frame"] = "camera_left_raw_optical"
        result["calibration_sha256"] = self.data["calibration_sha256"]
        result["urdf_sha256"] = self.data["urdf_sha256"]
        result["T_base_camera_left_rectified"] = (
            np.asarray(result["T_base_camera_left_raw"]) @ transform(self.model.R1.T, np.zeros(3))
        ).tolist()
        result["T_base_camera_right_raw"] = (
            np.asarray(result["T_base_camera_left_raw"])
            @ inverse(transform(self.model.R, self.model.T / 1000))
        ).tolist()
        self.data["solution"] = result
        self.data["stage"] = "checking"
        self.data["report"] = None
        self.hold.reset()
        self.guide_skip = 0
        save_json(self.path / f"solution_{result['id']}.json", result)
        self.persist()
        self.event("solved", automatic=True, solution=result)
        self.message = (
            "Coverage and prediction accuracy passed. Keep moving to fresh poses; "
            "the next recordings validate the frozen calibration."
        )

    def _recording_review(self):
        """Validation disagreement is separate from the frozen training fit."""
        report = self.data.get("report") if self.data else None
        if report and (
            any(not row["passed"] for row in report.get("checks", []))
            or (report.get("prior_validation") or {}).get("passed") is False
        ):
            return {
                "ready": False,
                "status": "independent_validation",
                "reason": report.get("reason") or "An independent prediction check failed.",
            }
        return self.readiness

    def _guidance(self, samples, measured, stage):
        measured = measured or {}
        guide = self.local_guide.guide(
            samples,
            measured.get("T_camera_board"),
            stage,
            self.guide_skip,
            model=getattr(self, "model", None),
            readiness=self._recording_review(),
            signature=measured.get("signature"),
        )
        target = guide.get("position_target") or (0.5, 0.5)
        guide = self.position_guide.apply(guide, measured.get("signature"), target)
        guide["pose_preview"] = pose_preview(
            samples, guide, measured.get("signature"), measured.get("T_camera_board")
        )
        return guide

    def _placement_token(self):
        return (
            self.guide_skip,
            self.data["stage"],
            tuple(s["id"] for s in self.data["samples"] if not s.get("excluded")),
        )

    def _next_sample_index(self):
        """Keep numbering monotonic when rejected records move to an archive."""
        prefixes = [
            int(s["id"].split("_", 1)[0])
            for s in self.data["samples"]
            if s["id"].split("_", 1)[0].isdigit()
        ]
        return max(
            int(self.data.get("next_sample_index", 0)),
            max(prefixes, default=-1) + 1,
            len(self.data["samples"]),
        )

    def _partial_placement_pose(self, observation, signature):
        """Validate a partial *display* estimate; never a capture qualification.

        Reuse describe's left PnP and independently fit the right eye. Require
        both native fits and both cross-camera predictions to meet the existing
        image error budgets. Missing corners are never synthesized as evidence.
        """
        ids, left, right = observation.common()
        total = (L20_BOARD.columns - 1) * (L20_BOARD.rows - 1)
        if not 6 <= len(ids) < total or tuple(observation.image_size) != tuple(
            self.model.image_size
        ):
            raise ValueError(
                "Partial positioning needs at least six shared corners in the calibrated image size."
            )
        objects = L20_BOARD.create().getChessboardCorners()[ids].astype(float)
        if np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0)) < 2:
            raise ValueError("Show another row of corners for partial positioning.")
        if (
            not signature
            or signature.get("rotation_camera") is None
            or signature.get("translation_camera_m") is None
        ):
            raise ValueError("A current board pose estimate is unavailable.")
        left_R = np.asarray(signature["rotation_camera"], dtype=float)
        left_t = np.asarray(signature["translation_camera_m"], dtype=float).reshape(3, 1) * 1000
        right_r, right_t = board_pose(objects, right, self.model.K2, self.model.D2)
        right_R = cv2.Rodrigues(right_r)[0]
        metrics = {}
        for key, R, t, K, D, pixels, limit in (
            (
                "left_native_rms_px",
                left_R,
                left_t,
                self.model.K1,
                self.model.D1,
                left,
                POLICY["native_reprojection_rms_px"],
            ),
            (
                "right_native_rms_px",
                right_R,
                right_t,
                self.model.K2,
                self.model.D2,
                right,
                POLICY["native_reprojection_rms_px"],
            ),
            (
                "left_to_right_rms_px",
                self.model.R @ left_R,
                self.model.R @ left_t + self.model.T.reshape(3, 1),
                self.model.K2,
                self.model.D2,
                right,
                POLICY["capture_stereo_rms_px"],
            ),
            (
                "right_to_left_rms_px",
                self.model.R.T @ right_R,
                self.model.R.T @ (right_t - self.model.T.reshape(3, 1)),
                self.model.K1,
                self.model.D1,
                left,
                POLICY["capture_stereo_rms_px"],
            ),
        ):
            projected = cv2.fisheye.projectPoints(
                objects.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, D
            )[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
            if not np.isfinite(rms) or rms > limit:
                raise ValueError(
                    "Partial board pose is inconsistent between the camera views; show more corners."
                )
            metrics[key] = rms
        return transform(left_R, left_t.ravel() / 1000).tolist(), {
            "source": "partial_stereo_checked_left_pnp",
            "partial": True,
            "shared_corners": len(ids),
            "expected_corners": total,
            "native_limit_px": POLICY["native_reprojection_rms_px"],
            "stereo_limit_px": POLICY["capture_stereo_rms_px"],
            **metrics,
        }

    @staticmethod
    def _placement_unavailable(measured=None):
        counts = (measured or {}).get("corners") or [None, None]
        total = (L20_BOARD.columns - 1) * (L20_BOARD.rows - 1)
        result = {
            "key": "unavailable",
            "available": False,
            "title": "Waiting for a fresh stereo board view",
            "instruction": "Show the complete board in both cameras for a small local positioning suggestion.",
            "adjustment": {"kind": "unavailable"},
            "pose_preview": None,
            "position_target": None,
            "target_rotation": None,
            "at_target": False,
            "positioning_reached": False,
            "advisory_only": True,
            "source": "stereo_board_pose",
            "scope_label": "Camera-only placement",
            "scope": "Camera-only board positioning; does not qualify a capture or assess robot clearance.",
            "visibility_counts": {
                "left": counts[0],
                "right": counts[1],
                "shared": (measured or {}).get("shared"),
                "expected": total,
            },
        }
        if measured and not measured.get("T_camera_board"):
            shared = measured.get("shared")
            if all(n is not None for n in counts) and counts[0] != counts[1]:
                missing_eye = "left" if counts[0] < counts[1] else "right"
                count = min(counts)
                result.update(
                    title=f"Show the board to the {missing_eye} camera",
                    instruction=f"The {missing_eye} camera sees {count}/{total} corners. Turn the printed face toward the {missing_eye} camera until all {total} are visible. Keep the board nearby and clear of the arm and table.",
                )
            elif shared is not None and shared < total:
                result.update(
                    title="Show the full printed face",
                    instruction=f"{shared}/{total} shared corners visible. Show more of the printed face to both cameras and clear anything covering it; the partial pose is not reliable enough for a tilt suggestion.",
                )
            elif measured.get("reason"):
                result["instruction"] = measured["reason"]
        return result

    def _update_placement_guidance(self, measured, observation=None):
        """Camera-only coaching during review; never used by capture or fitting.

        Compute once per processed image, not on browser polls. The independent
        local planner keeps its absolute target until a capture or guide_next.
        """
        self.placement = None
        measured.pop("T_camera_board_advisory", None)
        measured.pop("placement_pose_quality", None)
        measured.pop("placement_pose_reason", None)
        if not requires_review(self._recording_review()):
            return
        pose = measured.get("T_camera_board")
        signature = measured.get("signature")
        timing = measured.get("timing") or {}
        if (
            not signature
            or not timing.get("fresh_at_ingress")
            or time.monotonic() - measured["camera_monotonic"]
            > POLICY["camera_max_age_s"] + POLICY["stationary_window_s"]
        ):
            return
        pose_quality = {"source": "stereo_board_pose", "partial": False}
        if pose is None and observation is not None:
            try:
                pose, pose_quality = self._partial_placement_pose(observation, signature)
                measured["T_camera_board_advisory"] = pose
                measured["placement_pose_quality"] = pose_quality
            except (ValueError, cv2.error) as error:
                measured["placement_pose_reason"] = str(error)
                return
        if pose is None:
            return
        # Use camera observations only for orientation ranking. In particular,
        # do not reuse the disputed hand-eye transform or measured-arm coverage
        # as a positioning claim while the calibration is under review.
        camera_samples = [
            {
                "id": s["id"],
                "role": "training",
                "T_camera_board": s["T_camera_board"],
                "T_base_tool": s["T_camera_board"],
            }
            for s in self.data["samples"]
            if not s.get("excluded")
        ]
        if not camera_samples:
            return
        guide = self.placement_local_guide.guide(
            camera_samples,
            pose,
            "training",
            self.guide_skip,
            model=self.model,
            signature=signature,
        )
        guide = self.placement_position_guide.apply(
            guide,
            signature,
            guide.get("position_target") or [signature["x"], signature["y"]],
        )
        guide["pose_preview"] = pose_preview(camera_samples, guide, signature, pose)
        reached = bool(guide.get("at_target"))
        if reached:
            guide.update(
                title="Suggested position reached",
                instruction="The board matches this positioning suggestion. Calibration remains under review; recording checks still apply.",
            )
        guide = {
            **self._placement_unavailable(measured),
            **guide,
            "available": guide["key"] != "unavailable",
            "positioning_reached": reached,
            "frame_id": measured["frame_id"],
            "source": pose_quality["source"],
            "pose_quality": pose_quality,
        }
        if guide["key"] != "unavailable":
            guide["key"] = "placement-" + guide["key"]
        self.placement, self.placement_context = guide, self._placement_token()

    def _placement_state(self, measured):
        if (
            not self.active
            or not measured
            or measured["age_s"] >= GUIDE_POLICY["maximum_detection_gap_s"]
            or (measured.get("timing") or {}).get("processing_ms", float("inf"))
            > POLICY["stationary_window_s"] * 1000
        ):
            return self._placement_unavailable()
        if (
            not (measured.get("T_camera_board") or measured.get("T_camera_board_advisory"))
            or self.placement is None
            or self.placement.get("frame_id") != measured.get("frame_id")
            or self.placement_context != self._placement_token()
        ):
            return self._placement_unavailable(measured)
        return copy.deepcopy(self.placement)

    def suspend_detection(self):
        """Forget live results without closing the run, camera or feedback feed."""
        with self.lock:
            self._cancel_capture_retry("Board detection suspended. Resume detection before recording.")
            self.measurement = self.preview = None
            self.previews.clear()
            self.last_capture = None
            self.message = "Stream only. Board detection is suspended; camera video and arm feedback stay live."

    def state(self):
        with self.lock:
            stream_only = bool(getattr(self.controller, "stream_only", False))
            samples = self.data["samples"] if self.data else []
            count = lambda role: sum(s["role"] == role and not s.get("excluded") for s in samples)
            measured = None if stream_only else copy.deepcopy(self.measurement)
            if measured:
                measured["age_s"] = time.monotonic() - measured["measured_monotonic"]
            feedback = self.feed.status() if self.feed else {"connected": False, "age_s": None}
            ready = bool(
                self.active
                and not self.busy
                and not self.assessing
                and measured
                and measured["ready"]
                and measured["age_s"] < 3
                and feedback["connected"]
            )
            stage = self.data["stage"] if self.data else "setup"
            capture_issue = copy.deepcopy(self.data.get("capture_issue")) if self.data else None
            review = self._recording_review()
            recording_review = copy.deepcopy(review) if requires_review(review) else None
            can_retry = (
                self.active
                and not self.busy
                and not self.assessing
                and not self.pending
                and stage in ("training", "checking")
            )
            failed_training = self._failed_training_ids()
            guide = self._guidance(
                samples,
                measured if measured and measured["age_s"] < 3 else None,
                stage,
            )
            return {
                "active": self.active,
                "stream_only": stream_only,
                "busy": self.busy,
                "assessing": self.assessing,
                "readiness": copy.deepcopy(self.readiness),
                "workflow_automatic": True,
                "recovery": copy.deepcopy(self.data.get("recovery")) if self.data else None,
                "pending": bool(self.pending),
                "stage": stage,
                "message": "Stream only. Board detection is suspended; camera video and arm feedback stay live."
                if stream_only
                else recording_review["reason"]
                if recording_review
                else capture_issue["message"]
                if capture_issue and capture_issue["status"] != "resolved"
                else self.message,
                "recording_review": recording_review,
                "placement_guidance": self._placement_state(measured) if recording_review else None,
                "capture_issue": capture_issue,
                "capture_wait_reason": (measured.get("reason") if measured else self.message)
                if self.pending or self.auto_capture
                else None,
                "can_retry_capture": can_retry,
                "can_exclude_failed_training": bool(
                    can_retry and stage == "training" and failed_training
                ),
                "failed_training_ids": failed_training,
                "error": self.error,
                "path": str(self.path) if self.path else None,
                "run_id": self.path.name if self.path else None,
                "workspace_id": self.revision,
                "side": self.data["side"] if self.data else "right",
                "board": L20_BOARD.to_dict(),
                "kinematic_calibration": copy.deepcopy(self.data.get("kinematic_calibration"))
                if self.data else None,
                "arm_model_file": self.data.get("arm_model_file", "arm_model.urdf") if self.data else None,
                "feedback": feedback,
                "measurement": measured,
                "capture_ready": ready and not self.pending and stage in ("training", "checking"),
                "can_solve": self.active
                and not self.busy
                and not self.pending
                and stage == "training"
                and not self.assessing
                and bool(self.readiness and self.readiness.get("ready")),
                "training_count": count("training"),
                "check_count": count("check"),
                "diversity": guide["diversity"],
                "guidance": guide,
                "auto_capture": self.auto_capture,
                "hold_progress": self.hold.progress
                if measured and measured["age_s"] < 1.5 and feedback["connected"]
                else 0.0,
                "just_captured": self.last_capture is not None
                and time.monotonic() - self.last_capture < 3,
                "policy": dict(POLICY),
                "solution": copy.deepcopy(self.data["solution"]) if self.data else None,
                "report": copy.deepcopy(self.data["report"]) if self.data else None,
                "captures": [
                    {"id": s["id"], "role": s["role"], "excluded": s.get("excluded", False)}
                    for s in samples
                ],
            }

    def command(self, action, revision):
        try:
            return self._command(action, revision)
        except (ValueError, OSError, TimeoutError, cv2.error) as error:
            if (
                action in ("capture", "retry_capture")
                and self.data
                and revision == self.revision
                and self.pending is None
            ):
                with self.lock:
                    if self._wait_for_automatic_retry(
                        str(error), kind="capture_rejected", once=True
                    ):
                        return self.state()
                    else:
                        self._capture_issue(
                            "capture_rejected",
                            "Capture was not started. "
                            + str(error)
                            + " Review the cause, then retry.",
                            action=action,
                            retryable=False,
                        )
            raise

    def _command(self, action, revision):
        with self.workspace.access(revision):
            if revision != self.revision:
                raise ValueError("The hand–eye run belongs to another stereo session.")
        if action == "pause":
            self.close()
            return self.state()
        if action == "folder":
            if not self.path:
                raise ValueError("Start a session before opening its folder.")
            open_directory(self.path)
            return self.state()
        with self.lock:
            if not self.active or self.busy:
                raise ValueError("Start or resume a hand–eye session first; wait for ongoing work.")
            if action in ("capture", "retry_capture", "auto_start", "guide_next"):
                if getattr(self.controller, "stream_only", False):
                    # This only clears the suspension flag; it does not take
                    # detection_lock while holding the hand-eye state lock.
                    self.controller.resume_detection()
                    self.message = "Checking fresh board and measured arm feedback."
            if action in ("auto_start", "auto_stop", "guide_next"):
                if action == "auto_stop":
                    self._cancel_capture_retry(
                        "Automatic retry stopped. Start automatic capture when you want to continue."
                    )
                    self.message = "Automatic capture stopped. You can use manual capture."
                elif action == "auto_start":
                    review = self._recording_review()
                    if requires_review(review):
                        raise ValueError(
                            review["reason"]
                            + " Review this inconsistency before restarting automatic recording."
                        )
                    issue = self.data.get("capture_issue")
                    if issue and issue["status"] != "resolved" and not self._issue_retryable(issue):
                        raise ValueError(
                            "Review the recording failure and retry one capture before restarting automatic recording."
                        )
                    if self.data["stage"] not in ("training", "checking"):
                        raise ValueError(
                            "This calibration is complete. Start a new session to record."
                        )
                    if self.pending:
                        raise ValueError("Wait for the requested capture to finish.")
                    self.auto_capture = True
                    self.retry_once = False
                    if issue and issue["status"] in ("retry_required", "retry_waiting"):
                        self._wait_for_automatic_retry(
                            "Hold a comfortable visible pose.", kind=issue["kind"]
                        )
                    self.message = (
                        "Automatic capture started. Choose a new pose and hold steady. "
                        "The suggested pose is optional."
                    )
                else:
                    self.guide_skip += 1
                self.hold.reset()
                self.event(
                    action, guidance_policy=dict(GUIDE_POLICY), suggestion_offset=self.guide_skip
                )
            elif action in ("capture", "retry_capture"):
                if self.assessing:
                    raise ValueError("Evaluating saved poses for coverage and prediction accuracy.")
                if self.pending:
                    raise ValueError("A capture is already waiting for a qualifying frame.")
                if self.data["stage"] not in ("training", "checking"):
                    raise ValueError("This calibration is complete. Start a new session to record.")
                if action == "capture" and not self.state()["capture_ready"]:
                    raise ValueError(
                        self.measurement.get("reason", "Wait for board and arm feedback.")
                        if self.measurement
                        else self.message
                    )
                self.auto_capture = False
                self.retry_once = True
                self.hold.reset()
                self.pending = time.monotonic()
                self.message = "Capturing the next steady frame…"
                previous_issue = self.data.get("capture_issue")
                retryable = not requires_review(self._recording_review()) and (
                    not previous_issue
                    or previous_issue["status"] == "resolved"
                    or self._issue_retryable(previous_issue)
                )
                self._capture_issue(
                    "capture_retry",
                    ("Retry requested. " if action == "retry_capture" else "Capture requested. ")
                    + "Show the full board and hold the wrist still; waiting up to 10 seconds for a valid camera and measured-arm pair.",
                    status="retry_pending",
                    retryable=retryable,
                    action=action,
                )
                self.event("capture_requested", stage=self.data["stage"], action=action)
            elif action == "exclude_failed_training":
                self._exclude_failed_training()
            elif action == "solve":
                # Retained for older clients; normal guided collection freezes
                # an accurate, observable candidate without this action.
                if not self.state()["can_solve"]:
                    raise ValueError(
                        (self.readiness or {}).get("reason")
                        or "Calibration is still assessing coverage and prediction accuracy."
                    )
                self._freeze_candidate(self.readiness["solution"])
            elif action == "undo":
                if self.data["stage"] != "training" or self.pending:
                    raise ValueError(
                        "Only training captures may be excluded before fitting. Check failures remain recorded."
                    )
                self.auto_capture = False
                self.hold.reset()
                self.guide_skip = 0
                active = [s for s in self.data["samples"] if not s.get("excluded")]
                if active:
                    active[-1]["excluded"] = True
                    self.persist()
                    self.event("excluded", sample=active[-1]["id"])
                    self._schedule_assessment()
            else:
                raise ValueError("Unknown hand–eye action.")
        return self.state()

    def _process(self, packet, detector):
        started = time.monotonic()
        ingress_age = started - packet.timestamp
        timing = {
            "ingress_age_ms": ingress_age * 1000,
            "ingress_limit_ms": POLICY["camera_max_age_s"] * 1000,
            # This is a separate bounded analysis deadline. The image must
            # still be fresh at ingress, and measured joints must stay still
            # from before image receipt through the final capture decision.
            "processing_limit_ms": POLICY["stationary_window_s"] * 1000,
            "fresh_at_ingress": 0 <= ingress_age <= POLICY["camera_max_age_s"],
        }
        with self.lock:
            # Full native-pixel multiscale detection is mandatory on the frame
            # that can finish a dwell or fulfil a manual request. Guidance uses
            # the cheaper native-scale search, with fallback on poor geometry.
            needs_full = self.pending is not None or (
                self.auto_capture
                and self.hold.started is not None
                and packet.timestamp - self.hold.started >= GUIDE_POLICY["hold_seconds"]
            )
        detection_ms = 0.0
        for full in ([True] if needs_full else [False, True]):
            detection_started = time.monotonic()
            observation, quality = detector.observe(
                packet.image, self.controller.camera.swap, full=full
            )
            detection_ms += (time.monotonic() - detection_started) * 1000
            geometry_error = None
            try:
                camera_pose, metrics = camera_board_pose(observation, self.model)
            except (ValueError, cv2.error) as error:
                geometry_error = error
                camera_pose = metrics = None
            if geometry_error is None:
                break
        detected = time.monotonic()
        timing["detection_ms"] = detection_ms
        timing["full_detection"] = full
        measurement = {
            "frame_id": packet.sequence,
            "camera_monotonic": packet.timestamp,
            "measured_monotonic": time.monotonic(),
            "ready": False,
            "corners": [len(observation.left_ids), len(observation.right_ids)],
            "shared": len(observation.common()[0]),
            "quality": quality,
            "timing": timing,
            "native_eye_size": list(observation.image_size),
            "inspection": {
                **inspect_observation(observation, L20_BOARD, self.model),
                "frame_id": packet.sequence,
                "source": "live",
            },
        }
        inspected = time.monotonic()
        timing["inspection_ms"] = (inspected - detected) * 1000
        image = packet.image.copy()
        w = image.shape[1] // 2
        views = [image[:, :w], image[:, w:]]
        if self.controller.camera.swap:
            views.reverse()
        # Browser SVG overlays use this exact frame's native-pixel observations.
        preview = Preview.encode(np.hstack(views))
        encoded = time.monotonic()
        timing["preview_ms"] = (encoded - inspected) * 1000
        reason = None
        arm_pose = None
        try:
            # Image guidance may remain usable with a partial board. Capture
            # still requires all corners and the unchanged geometric checks.
            measurement["signature"] = describe(observation, L20_BOARD, self.model)
        except (ValueError, cv2.error):
            measurement["signature"] = None
        described = time.monotonic()
        timing["description_ms"] = (described - encoded) * 1000
        try:
            if not timing["fresh_at_ingress"]:
                raise ValueError("Waiting for a fresh camera frame; the available frame is stale.")
            if geometry_error is not None:
                raise geometry_error
            measurement["T_camera_board"] = camera_pose.tolist()
            measurement["reprojection_px"] = [
                metrics[f"{eye}_reprojection_rms_px"] for eye in ("left", "right")
            ]
            measurement["stereo_reprojection_px"] = [
                metrics[f"{direction}_rms_px"] for direction in ("left_to_right", "right_to_left")
            ]
            # Waiting for detection naturally allows feedback to bracket image receipt.
            arm_pose, pairing = self.feed.paired(
                packet.timestamp, self.kinematics, require_stationary=False
            )
            measurement["joint_limit_diagnostics"] = pairing.get("joint_limit_diagnostics", [])
            # Read-only repeatability diagnostics must also work at a repeated
            # pose. Publish the time-paired measured angles before novelty can
            # block recording; this never arms capture or sends robot commands.
            if "nearest" in pairing:
                measurement["paired_robot"] = {
                    "T_base_tool": arm_pose.tolist(),
                    "feedback": copy.deepcopy(pairing["nearest"]),
                    "pair_skew_s": pairing["pair_skew_s"],
                    "stationary": pairing["stationary"],
                    "joint_range_rad": pairing["joint_range_rad"],
                    "fk_policy": pairing["fk_policy"],
                }
            if not self.feed.status()["connected"]:
                raise ValueError("Measured arm feedback stopped; capture is blocked.")
            with self.lock:
                previous = [s["T_base_tool"] for s in self.data["samples"] if not s.get("excluded")]
            if not novel(arm_pose, previous):
                raise ValueError(
                    "Pose already captured. Change position or wrist tilt, then pause."
                )
            measurement.update(
                ready=True,
                reason="Ready to capture this steady pose.",
                pair_skew_ms=pairing["pair_skew_s"] * 1000,
                reprojection_px=[
                    metrics["left_reprojection_rms_px"],
                    metrics["right_reprojection_rms_px"],
                ],
            )
        except (ValueError, cv2.error) as error:
            reason = str(error)
            measurement["reason"] = reason
            if isinstance(error, CaptureBlocked):
                measurement["blocker_code"] = error.code
                measurement.update(error.details)
        timing["geometry_and_pairing_ms"] = (time.monotonic() - described) * 1000
        hold_eligible = reason is None
        with self.lock:
            if getattr(self.controller, "stream_only", False):
                return
            self.measurement, self.preview = measurement, preview
            self.previews[packet.sequence] = preview
            while len(self.previews) > 3:
                del self.previews[next(iter(self.previews))]
            if not self.active:
                return
            guide_started = time.monotonic()
            guide = self._guidance(
                self.data["samples"],
                measurement,
                self.data["stage"],
            )
            measurement["guidance"] = guide
            self._update_placement_guidance(measurement, observation)
            # Detection, preview generation and guidance are not camera lag.
            # Bound their latency separately, then check the continuous joint
            # history through NOW so an old stationary image cannot capture a
            # wrist that started moving while that image was being processed.
            decision = time.monotonic()
            timing["guidance_ms"] = (decision - guide_started) * 1000
            timing["processing_ms"] = (decision - started) * 1000
            timing["decision_frame_age_ms"] = (decision - packet.timestamp) * 1000
            if reason is None:
                try:
                    if decision - started > POLICY["stationary_window_s"]:
                        raise CaptureBlocked(
                            "processing_delay",
                            "Image analysis took too long; waiting for a promptly processed frame.",
                        )
                    if not self.feed.status()["connected"]:
                        raise ValueError("Measured arm feedback stopped; capture is blocked.")
                    arm_pose, pairing = self.feed.paired(
                        packet.timestamp, self.kinematics, through=decision
                    )
                    timing["decision_feedback_age_ms"] = (
                        pairing.get("decision_feedback_age_s", 0) * 1000
                    )
                    if time.monotonic() - started > POLICY["stationary_window_s"]:
                        raise CaptureBlocked(
                            "processing_delay",
                            "Image analysis took too long; waiting for a promptly processed frame.",
                        )
                except (ValueError, cv2.error) as error:
                    reason = str(error)
                    measurement.update(ready=False, reason=reason)
                    if isinstance(error, CaptureBlocked):
                        measurement["blocker_code"] = error.code
                        measurement.update(error.details)
                    # Previous motion in the joint window may be settling while
                    # current paired poses already stay within the dwell bounds.
                    # It still blocks recording. All other failures reset dwell.
                    hold_eligible = isinstance(error, CaptureBlocked) and error.code == "motion"
            measurement["measured_monotonic"] = time.monotonic()
            automatic = False
            if self.auto_capture and not self.pending and not self.assessing:
                if guide["key"] == "done":
                    self.auto_capture = False
                    self.hold.reset()
                else:
                    progress = self.hold.update(
                        packet.timestamp,
                        guide["key"],
                        # Suggested angle/image centre are coaching, not
                        # calibration requirements. Dwell and the strict joint
                        # stability window run together; BOTH must pass to save.
                        hold_eligible,
                        arm_pose,
                    )
                    if progress >= 1 and reason is None and full:
                        self.pending = packet.timestamp
                        automatic = True
            if self.pending:
                if self.stop_event.is_set():
                    return
                if self._expire_pending(reason or "No qualifying fresh frame arrived."):
                    return
                if reason:
                    self.event("capture_wait", frame_id=packet.sequence, reason=reason)
                elif packet.timestamp >= self.pending and full:
                    role = "training" if self.data["stage"] == "training" else "check"
                    sample_index = self._next_sample_index()
                    sample_id = f"{sample_index:04d}_{uuid.uuid4().hex[:8]}"
                    sample = {
                        "id": sample_id,
                        "utc": utc(),
                        "role": role,
                        "frame_id": packet.sequence,
                        "camera_received_monotonic": packet.timestamp,
                        "T_base_tool": arm_pose.tolist(),
                        "T_camera_board": camera_pose.tolist(),
                        "pairing": pairing,
                        "observation": observation.to_dict(),
                        "signature": measurement["signature"],
                        "quality": quality,
                        "capture_mode": "guided" if automatic else "manual",
                        "guidance": guide,
                        "metrics": metrics,
                        "timing": copy.deepcopy(timing),
                        "image": f"frames/{sample_id}.png",
                    }
                    target = self.path / sample["image"]
                    try:
                        if not cv2.imwrite(str(target), packet.image):
                            raise OSError("Could not save the original stereo frame.")
                        sample["image_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    except (OSError, cv2.error) as error:
                        self.pending = None
                        self._capture_issue(
                            "capture_write_failed",
                            "Pose was not recorded. "
                            + str(error)
                            + " Check the session folder is writable, then retry.",
                            frame_id=packet.sequence,
                        )
                        return
                    self.data["samples"].append(sample)
                    self.data["next_sample_index"] = sample_index + 1
                    if role == "check":
                        self.data["report"] = validation_report(
                            self.data["samples"],
                            self.data["solution"],
                            self.model,
                            prior_validation=self.data.get("prior_validation_evidence"),
                        )
                        if self.data["report"]["passed"]:
                            self.data["stage"] = "complete"
                        self.export_result()
                    check_failed = (
                        role == "check" and not self.data["report"]["checks"][-1]["passed"]
                    )
                    if check_failed:
                        self._capture_issue(
                            "check_failed",
                            "Pose saved, but its independent prediction check failed. "
                            + self.data["report"].get(
                                "reason", "Review the board and measured pose."
                            )
                            + " Retry a steady pose after reviewing the cause; the failed check remains recorded.",
                            sample_id=sample_id,
                        )
                    else:
                        self._resolve_capture_issue(sample_id)
                    self.persist()
                    self.pending = None
                    self.hold.reset()
                    self.guide_skip = 0
                    self.last_capture = time.monotonic()
                    self.measurement["ready"] = False
                    self.measurement["reason"] = (
                        self.data["capture_issue"]["message"]
                        if check_failed
                        else "Captured. Change position or wrist tilt for the next pose."
                    )
                    self.event(
                        "captured",
                        sample_id=sample_id,
                        role=role,
                        mode=sample["capture_mode"],
                        guide=guide["key"],
                    )
                    phase_done = self.data["stage"] == "complete"
                    if phase_done or check_failed:
                        self.auto_capture = False
                        self.event(
                            "auto_stopped",
                            reason="check_failed" if check_failed else "stage_complete",
                        )
                    if self.retry_once:
                        self.auto_capture = self.retry_once = False
                    if role == "training":
                        self._schedule_assessment()
                    self.message = (
                        self.data["capture_issue"]["message"]
                        if check_failed
                        else "Captured. You can move to the next pose."
                        if self.data["stage"] != "complete"
                        else "Independent geometric checks passed. Collection is complete."
                    )
            elif self.data["stage"] in ("training", "checking"):
                if self.last_capture is None or time.monotonic() - self.last_capture >= 3:
                    self.message = measurement["reason"]

    def _invalidate_export(self, reason):
        if (self.path / "handeye.npz").exists():
            invalidated = self.path / "invalidated_exports"
            invalidated.mkdir(exist_ok=True)
            destination = invalidated / f"handeye_{uuid.uuid4().hex[:12]}.npz"
            (self.path / "handeye.npz").replace(destination)
            self.event("export_invalidated", previous_export=str(destination), reason=reason)

    def export_result(self):
        solution = copy.deepcopy(self.data["solution"])
        report = self.data["report"]
        solution["status"] = (
            "geometric_checks_passed" if report["passed"] else "needs_independent_checks"
        )
        result = {
            "solution": solution,
            "validation": report,
            "board": self.data["board"],
            "policy": self.data["policy"],
            "run_path": str(self.path),
            "arm_model": {
                "file": self.data.get("arm_model_file", "arm_model.urdf"),
                "format": self.data.get("arm_model_format", "urdf"),
                "sha256": self.data["urdf_sha256"],
                "input_convention": "Raw measured follower encoder angles in radians.",
                "calibration": self.data.get("kinematic_calibration"),
            },
            "not_included": "No measured tool-to-palm transform; no automatic robot configuration or motion.",
        }
        save_json(self.path / "result.json", result)
        if not report["passed"]:
            self._invalidate_export(
                report.get("reason", "Independent validation no longer passes.")
            )
        # A machine-readable export is available only once independent checks pass.
        if report["passed"]:
            np.savez(
                self.path / "handeye.npz",
                **{k: np.asarray(v) for k, v in solution.items() if k.startswith("T_")},
                length_unit="m",
                base_frame=solution["base_frame"],
                tool_frame=solution["tool_frame"],
                calibration_sha256=solution["calibration_sha256"],
                urdf_sha256=self.data["urdf_sha256"],
                arm_model_file=str(self.path / self.data.get("arm_model_file", "arm_model.urdf")),
                physical_accuracy_verified=False,
            )

    def _run(self):
        sequence = -1
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="handeye-eye") as pool:
            detector = Detector(L20_BOARD, executor=pool)
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    if getattr(self.controller, "stream_only", False):
                        self.stop_event.wait(0.05)
                        continue
                    with self.controller.source.condition:
                        packet = self.controller.source.latest
                    if self.controller.source.error:
                        raise ValueError(self.controller.source.error)
                    if (
                        packet
                        and packet.sequence != sequence
                        and started - packet.timestamp <= POLICY["camera_max_age_s"]
                    ):
                        sequence = packet.sequence
                        with getattr(self.controller, "detection_lock", self.lock):
                            if not getattr(self.controller, "stream_only", False):
                                self._process(packet, detector)
                        with self.lock:
                            self.error = None
                    elif not packet or started - packet.timestamp > POLICY["camera_max_age_s"]:
                        # Warm-up or a transient late frame is a wait state, not
                        # a recording failure. It must reset the dwell and make
                        # capture ineligible without silently disarming auto.
                        with self.lock:
                            self.measurement = None
                            self.hold.reset()
                            self.message = (
                                "Waiting for a fresh camera frame. "
                                "Capture resumes when the camera is ready."
                            )
                            self._expire_pending(
                                "Camera frames are missing or stale. Check the camera connection."
                            )
                except Exception as error:
                    logging.getLogger(__name__).exception("Hand–eye measurement failed")
                    with self.lock:
                        attempted = bool(self.pending or self.auto_capture)
                        self.error = str(error)
                        self.measurement = None
                        self.hold.reset()
                        if self.auto_capture:
                            self.auto_capture = False
                            self.event(
                                "auto_stopped",
                                reason="measurement_or_recording_error",
                                error=str(error),
                            )
                        if self.pending:
                            self.event("capture_failed", reason=str(error))
                            self.pending = None
                        if attempted:
                            self._capture_issue(
                                "capture_failed",
                                "Capture did not finish. "
                                + str(error)
                                + " Review the connection or recording error, then retry.",
                            )
                        self.message = str(error)
                # Target 10 Hz guidance without accumulating stale frames. Full
                # capture checks may take longer; live video remains independent.
                self.stop_event.wait(max(0, 0.1 - (time.monotonic() - started)))

    def close(self):
        self.stop_event.set()
        with self.lock:
            self._cancel_capture_retry(
                "Recording is paused. Resume the session and start automatic capture when ready."
            )
            self.assessment_generation += 1
            self.assessing = False
            if self.assessment_future:
                self.assessment_future.cancel()
            self.hold.reset()
            was_active = self.active
            self.busy = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise TimeoutError("Hand–eye detection is still stopping. Retry shortly.")
        if self.feed:
            self.feed.close()
        if self.controller:
            self.controller.handeye_active = False
        with self.lock:
            self.active = False
            self.busy = False
            if was_active:
                self.message = "Paused. Captures are saved; arm teleop is unchanged."
                self.event("paused")
