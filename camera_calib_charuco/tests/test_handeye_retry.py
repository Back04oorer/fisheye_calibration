"""Explicit retry/exclusion preserves evidence and never bypasses capture gates."""

import copy
import hashlib
import json
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from test_handeye import fixture_samples

from stereo_calibration.camera import Frame
from stereo_calibration.handeye import L20_BOARD, POLICY
from stereo_calibration.handeye_runtime import HandEye, PoseFeed
from stereo_calibration.observations import Observation


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    samples, model, _, _ = fixture_samples()
    h = HandEye(SimpleNamespace(access=lambda *args: nullcontext(None)))
    h.path, h.model, h.revision, h.active = tmp_path, model, "workspace", True
    h.kinematics = None
    h.data = {
        "samples": [],
        "stage": "training",
        "side": "right",
        "solution": None,
        "report": None,
        "board": L20_BOARD.to_dict(),
        "policy": dict(POLICY),
        "base_frame": "right/base_link",
        "tool_frame": "right/link7",
        "calibration_sha256": "camera",
        "urdf_sha256": "model",
    }
    (tmp_path / "frames").mkdir()
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    feed = SimpleNamespace(connected=True, moving=False, current=samples[0], close=lambda: None)
    feed.status = lambda: {"connected": feed.connected, "age_s": 0 if feed.connected else 2}

    def paired(timestamp, kinematics, *, through=None, require_stationary=True):
        if not feed.connected:
            raise ValueError("Measured arm feedback is missing. Check the feedback connection.")
        if feed.moving:
            raise ValueError("Wrist is moving. Hold still.")
        return np.asarray(feed.current["T_base_tool"]), {"pair_skew_s": 0.001}

    feed.paired = paired
    h.feed = feed
    h.controller = SimpleNamespace(
        camera=SimpleNamespace(swap=False),
        handeye_active=True,
        source=SimpleNamespace(condition=threading.Condition(), latest=None, error=None),
    )
    h.persist()
    yield h, samples, clock
    h.close()
    h.assessment_executor.shutdown(wait=True, cancel_futures=True)


class Detector:
    def __init__(self, h, *, hidden=False):
        self.h, self.hidden = h, hidden

    def observe(self, image, swap, full):
        obs = Observation.from_dict(self.h.feed.current["observation"])
        if self.hidden:
            obs.left_ids, obs.left_points = obs.left_ids[:6], obs.left_points[:6]
        return obs, []


def process(h, clock, detector=None, *, timestamp=None):
    frame = Frame(
        int(clock[0] * 100),
        clock[0] if timestamp is None else timestamp,
        np.zeros((720, 2560, 3), np.uint8),
    )
    h._process(frame, detector or Detector(h))


def wait_assessment(h):
    deadline = time.monotonic() + 5
    while h.assessing and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not h.assessing


def test_explicit_exclusion_uses_full_fit_only_and_preserves_every_frame_and_check(
    runtime, monkeypatch
):
    h, samples, _clock = runtime
    training = copy.deepcopy(samples[:13])
    for sample in training:
        sample["role"] = "training"
    checks = copy.deepcopy(samples[16:20])
    h.data["samples"] = training + checks
    h.data["prior_validation_evidence"] = {
        "samples": copy.deepcopy(checks),
        "source_run_id": "archived",
    }
    for sample in h.data["samples"]:
        sample["image"] = f"frames/{sample['id']}.png"
        (h.path / sample["image"]).write_bytes(b"original-frame-" + sample["id"].encode())
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (h.path / "frames").iterdir()
    }
    original_checks = copy.deepcopy(checks)
    h.readiness = h.data["readiness"] = {
        "ready": False,
        "status": "accuracy",
        "reason": "Prediction error.",
        "sample_count": 13,
        "coverage": {"sufficient": True},
        "fit": [{"id": s["id"], "passed": s["id"] in ("0", "2")} for s in training],
        "leave_one_out": [{"id": s["id"], "passed": False} for s in training],
    }
    observed = []

    def assess(rows, model, *, prior_validation=None):
        observed.append((copy.deepcopy(rows), copy.deepcopy(prior_validation)))
        return {
            "ready": False,
            "reason": "Need useful two-axis coverage.",
            "status": "coverage",
            "coverage": {"sufficient": False},
            "solution": None,
        }

    monkeypatch.setattr("stereo_calibration.handeye_runtime.training_assessment", assess)
    h.persist()
    assert len(h.state()["failed_training_ids"]) == 11
    h.command("exclude_failed_training", "workspace")
    wait_assessment(h)
    assert [
        s["id"] for s in h.data["samples"] if s["role"] == "training" and not s.get("excluded")
    ] == ["0", "2"]
    assert h.data["samples"][13:] == original_checks
    assert [row["id"] for row in observed[0][0]] == ["0", "2"]
    assert observed[0][1] == h.data["prior_validation_evidence"]
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (h.path / "frames").iterdir()
    } == hashes
    snapshot = json.loads(next(h.path.glob("assessment_before_exclusion_*.json")).read_text())
    assert len(snapshot["excluded_ids"]) == 11 and len(snapshot["assessment"]["fit"]) == 13
    issue = h.state()["capture_issue"]
    assert issue["kind"] == "training_excluded" and issue["status"] == "retry_waiting"
    assert issue["message"].startswith("11 poses excluded; 2 remain.")
    assert h.state()["can_retry_capture"] and h.auto_capture
    assert h.data["solution"] is None and not (h.path / "handeye.npz").exists()
    assert json.loads((h.path / "run.json").read_text())["capture_issue"] == issue


def test_partial_assessment_cannot_silently_exclude_unassessed_records(runtime):
    h, samples, _clock = runtime
    h.data["samples"] = copy.deepcopy(samples[:4])
    h.readiness = {"fit": [{"id": "0", "passed": False}]}
    before = copy.deepcopy(h.data)
    with pytest.raises(ValueError, match="no failed active"):
        h.command("exclude_failed_training", "workspace")
    assert h.data == before and not list(h.path.glob("assessment_before_exclusion_*.json"))


def test_recoverable_rejection_queues_one_retry_and_recovers_only_after_full_dwell(runtime):
    h, _samples, clock = runtime
    process(h, clock, Detector(h, hidden=True))
    assert h.state()["capture_issue"] is None
    queued = h.command("capture", "workspace")
    assert queued["capture_issue"]["status"] == "retry_waiting"
    assert queued["capture_issue"]["retryable"] and queued["auto_capture"]
    original_issue = copy.deepcopy(h.state()["capture_issue"])
    original_run = (h.path / "run.json").read_bytes()
    original_events = (h.path / "events.jsonl").read_bytes()
    for _ in range(4):
        clock[0] += 0.55
        process(h, clock, Detector(h, hidden=True))
    assert not h.data["samples"] and h.state()["hold_progress"] == 0
    assert (h.path / "run.json").read_bytes() == original_run
    assert (h.path / "events.jsonl").read_bytes() == original_events
    for index in range(3):
        clock[0] += 0.55
        process(h, clock)
        if index < 2:
            assert not h.data["samples"]
            assert h.state()["capture_issue"] == original_issue
    wait_assessment(h)
    assert len(h.data["samples"]) == 1
    assert h.state()["capture_issue"]["status"] == "resolved"
    assert not h.auto_capture  # A rejected one-shot request does not become endless collection.
    assert h.data["solution"] is None


def test_retry_never_saves_without_feedback_or_from_stale_frame(runtime):
    h, _samples, clock = runtime
    h.feed.connected = False
    h.command("retry_capture", "workspace")
    clock[0] += 0.1
    process(h, clock)
    assert not h.data["samples"] and h.pending is not None
    h.feed.connected = True
    clock[0] += 1
    process(h, clock, timestamp=clock[0] - 0.8)
    assert not h.data["samples"] and "stale" in h.measurement["reason"]
    h.feed.moving = True
    clock[0] += 0.1
    process(h, clock)
    assert not h.data["samples"] and "moving" in h.measurement["reason"]


def test_pending_retry_expires_even_without_any_new_camera_frame(runtime):
    h, _samples, clock = runtime
    h.command("retry_capture", "workspace")
    h.controller.source.latest = None

    class OneIteration:
        def __init__(self):
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, duration):
            self.stopped = True

        def set(self):
            self.stopped = True

    h.stop_event = OneIteration()
    clock[0] += 10.1
    h._run()
    assert h.pending is None and not h.data["samples"]
    issue = h.state()["capture_issue"]
    assert issue["kind"] == "capture_timeout" and "Camera frames" in issue["message"]
    assert issue["status"] == "retry_waiting" and issue["retryable"]
    assert h.auto_capture
    h.stop_event = threading.Event()
    h.message = "Waiting for a fresh camera frame."
    assert h.state()["message"] == issue["message"]
    assert json.loads((h.path / "run.json").read_text())["capture_issue"] == issue


def test_image_write_failure_requests_retry_and_never_appends_sample(runtime, monkeypatch):
    h, _samples, clock = runtime
    h.command("retry_capture", "workspace")
    monkeypatch.setattr("stereo_calibration.handeye_runtime.cv2.imwrite", lambda *args: False)
    clock[0] += 0.1
    process(h, clock)
    assert not h.data["samples"] and h.pending is None
    issue = h.state()["capture_issue"]
    assert issue["kind"] == "capture_write_failed" and "writable" in issue["message"]
    assert not issue["retryable"] and not h.auto_capture
    with pytest.raises(ValueError, match="Review the recording failure"):
        h.command("auto_start", "workspace")
    clock[0] += 0.1
    process(h, clock)
    assert h.state()["capture_issue"] == issue


@pytest.mark.parametrize("action", ["auto_stop", "pause"])
def test_stop_and_pause_cancel_pending_retry_and_persist_disarmed_state(runtime, action):
    h, _samples, clock = runtime
    h.command("retry_capture", "workspace")
    assert h.pending is not None
    h.command(action, "workspace")
    assert h.pending is None and not h.auto_capture
    issue = h.state()["capture_issue"]
    assert issue["status"] == "retry_required" and issue["retryable"]
    assert json.loads((h.path / "run.json").read_text())["capture_issue"] == issue
    for _ in range(3):
        clock[0] += 0.55
        process(h, clock)
    assert not h.data["samples"]
    if action == "auto_stop":
        h.command("auto_start", "workspace")
        assert h.auto_capture and h.state()["capture_issue"]["status"] == "retry_waiting"
        for _ in range(3):
            clock[0] += 0.55
            process(h, clock)
        wait_assessment(h)
        assert len(h.data["samples"]) == 1
        assert h.auto_capture  # Explicit Start means continued guided collection.


def test_ordinary_auto_waits_for_feedback_and_motion_without_new_failure_issue(runtime):
    h, _samples, clock = runtime
    h.command("auto_start", "workspace")
    h.feed.connected = False
    for _ in range(3):
        clock[0] += 0.55
        process(h, clock)
    assert h.auto_capture and not h.data["samples"] and h.state()["capture_issue"] is None
    h.feed.connected, h.feed.moving = True, True
    clock[0] += 0.55
    process(h, clock)
    assert not h.data["samples"] and h.state()["hold_progress"] == 0
    h.feed.moving = False
    for index in range(3):
        clock[0] += 0.55
        process(h, clock)
        if index < 2:
            assert not h.data["samples"]
    wait_assessment(h)
    assert len(h.data["samples"]) == 1
    clock[0] += 0.55
    process(h, clock)  # The already-saved pose is advice, not another retry alarm.
    assert h.state()["capture_issue"] is None


def test_legacy_training_excluded_issue_can_start_automatic_collection(runtime):
    h, _samples, _clock = runtime
    h.data["capture_issue"] = {
        "id": "legacy",
        "kind": "training_excluded",
        "status": "retry_required",
        "message": "11 excluded; 2 remain.",
    }
    h.command("auto_start", "workspace")
    assert h.auto_capture and h.state()["capture_issue"]["status"] == "retry_waiting"
    assert h.state()["capture_issue"]["retryable"]


@pytest.mark.parametrize("delay, expected_capture", [(0.27, True), (0.71, False)])
def test_fresh_ingress_has_separate_bounded_processing_budget(runtime, delay, expected_capture):
    h, _samples, clock = runtime

    class SlowDetector(Detector):
        def observe(self, *args, **kwargs):
            clock[0] += delay
            return super().observe(*args, **kwargs)

    h.command("retry_capture", "workspace")
    process(h, clock, SlowDetector(h))
    wait_assessment(h)
    timing = h.measurement["timing"]
    assert timing["fresh_at_ingress"] and timing["ingress_age_ms"] == 0
    assert timing["detection_ms"] == pytest.approx(delay * 1000)
    assert timing["decision_frame_age_ms"] == pytest.approx(delay * 1000)
    assert timing["ingress_limit_ms"] == 250
    assert bool(h.data["samples"]) == expected_capture
    if expected_capture:
        assert h.data["samples"][0]["timing"] == timing
    else:
        assert h.measurement["blocker_code"] == "processing_delay"
        assert not h.measurement["ready"] and h.pending is not None


def test_processing_allowance_does_not_accept_stale_at_ingress(runtime):
    h, _samples, clock = runtime
    h.command("retry_capture", "workspace")
    process(h, clock, timestamp=clock[0] - 0.251)
    assert not h.data["samples"] and not h.measurement["ready"]
    assert not h.measurement["timing"]["fresh_at_ingress"]
    assert "stale" in h.measurement["reason"]


def test_stationarity_is_rechecked_through_decision_after_image_processing(runtime):
    h, _samples, clock = runtime
    paired = h.feed.paired
    paired_times = []

    def moved_after_image(timestamp, kinematics, *, through=None, require_stationary=True):
        paired_times.append(through)
        if through is not None:
            raise ValueError("Wrist is moving. Movement after the image was received.")
        return paired(timestamp, kinematics)

    h.feed.paired = moved_after_image
    h.command("retry_capture", "workspace")
    process(h, clock)
    assert paired_times == [None, clock[0]]
    assert not h.data["samples"] and not h.measurement["ready"]
    assert "moving" in h.measurement["reason"]


def test_retry_timeout_keeps_calibration_review_primary_and_attempt_secondary(runtime):
    h, _samples, clock = runtime
    review = {
        "ready": False,
        "status": "accuracy",
        "reason": "Full-fit prediction error is 27.37 px; limit 3 px.",
        "coverage": {"sufficient": True},
        "fit_max_rms_px": 27.37,
    }
    h.readiness = h.data["readiness"] = copy.deepcopy(review)
    h._capture_issue("training_accuracy", review["reason"])
    state = h.command("retry_capture", "workspace")
    assert state["recording_review"] == review
    assert state["message"] == review["reason"]
    assert state["capture_issue"]["status"] == "retry_pending"
    clock[0] += 10.1
    h._expire_pending("Waiting for current measured arm feedback.")
    state = h.state()
    assert state["recording_review"] == review and state["message"] == review["reason"]
    assert state["capture_issue"]["kind"] == "capture_timeout"
    assert not state["capture_issue"]["retryable"] and not state["auto_capture"]
    stored = json.loads((h.path / "run.json").read_text())
    assert stored["readiness"] == review
    assert stored["capture_issue"] == state["capture_issue"]


def test_dwell_overlaps_real_joint_window_and_final_frame_is_fully_checked(runtime, monkeypatch):
    h, samples, clock = runtime
    start = clock[0]
    h.feed = PoseFeed("right", h.path)
    h.kinematics = SimpleNamespace(
        names=["yaw"],
        pose=lambda names, positions: np.asarray(samples[0]["T_base_tool"]),
        limit_diagnostics=lambda names, positions: [],
    )
    h.feed.current = samples[0]
    monkeypatch.setattr(h, "_schedule_assessment", lambda: None)
    calls = []

    class CountingDetector(Detector):
        def observe(self, image, swap, full):
            calls.append(full)
            return super().observe(image, swap, full)

    detector = CountingDetector(h)
    h.auto_capture = True
    for index in range(12):
        clock[0] = start + index * 0.1
        # The arm stops at start. Historical motion must block early saves,
        # but must not postpone the start of the current-pose dwell.
        h.feed.history.clear()
        for offset in np.arange(-1, index * 0.1 + 0.001, 0.01):
            h.feed.history.append({
                "sample_monotonic": start + round(float(offset), 6),
                "names": ["yaw"],
                "positions": [0.0 if offset < -0.001 else 0.1],
            })
        process(h, clock, detector)
        if index < 10:
            assert not h.data["samples"]
        if index == 4:
            assert h.measurement["blocker_code"] == "motion"
            assert h.hold.progress == pytest.approx(0.4)
        if h.data["samples"]:
            break
    assert len(h.data["samples"]) == 1
    assert clock[0] - start <= 1.11  # Former serial waits needed at least 1.7 s.
    assert calls[0] is False and calls[-1] is True
    saved = h.data["samples"][0]
    assert saved["timing"]["full_detection"]
    assert saved["pairing"]["stationary"]
    assert saved["pairing"]["nearest"]["positions"] == [0.1]
    assert saved["pairing"]["decision_feedback_age_s"] <= POLICY["pair_max_skew_s"]


@pytest.mark.parametrize("failure", ["visibility", "stereo"])
def test_fast_guidance_cannot_save_a_frame_rejected_by_full_detection(runtime, failure):
    h, _samples, clock = runtime
    calls = []

    class FullFailure(Detector):
        def observe(self, image, swap, full):
            calls.append(full)
            obs, quality = super().observe(image, swap, full)
            if full:
                if failure == "visibility":
                    obs.left_ids, obs.left_points = obs.left_ids[:6], obs.left_points[:6]
                else:
                    obs.right_points[:, 0] += 30
            return obs, quality

    h.auto_capture = True
    detector = FullFailure(h)
    for _ in range(3):
        process(h, clock, detector)
        clock[0] += 0.55
    assert calls == [False, False, True]
    assert not h.data["samples"] and not h.measurement["ready"]
    assert h.hold.progress == 0


def test_fast_detection_falls_back_when_native_scale_misses_corners(runtime):
    h, _samples, clock = runtime
    calls = []

    class NeedsFull(Detector):
        def observe(self, image, swap, full):
            calls.append(full)
            self.hidden = not full
            return super().observe(image, swap, full)

    process(h, clock, NeedsFull(h))
    assert calls == [False, True]
    assert h.measurement["ready"] and h.measurement["shared"] == 12
    assert not h.data["samples"]


def test_manual_request_during_fast_detection_waits_for_a_fully_checked_frame(runtime):
    h, _samples, clock = runtime

    class RequestDuringDetection(Detector):
        def observe(self, image, swap, full):
            h.pending = clock[0]
            return super().observe(image, swap, full)

    process(h, clock, RequestDuringDetection(h))
    assert not h.data["samples"] and h.pending is not None
    clock[0] += 0.1
    process(h, clock)
    wait_assessment(h)
    assert len(h.data["samples"]) == 1
    assert h.data["samples"][0]["timing"]["full_detection"]
