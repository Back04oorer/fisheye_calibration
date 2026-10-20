"""Adaptive recording's worker may not block UI or apply stale calibration."""

import copy
import json
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_handeye import fixture_samples

from stereo_calibration.camera import Frame
from stereo_calibration.guidance import describe
from stereo_calibration.handeye import L20_BOARD, POLICY, solve
from stereo_calibration.handeye_guidance import GUIDE_POLICY
from stereo_calibration.handeye_runtime import HandEye
from stereo_calibration.observations import Observation


@pytest.fixture
def runtime(tmp_path):
    samples, model, _, _ = fixture_samples()
    workspace = SimpleNamespace(access=lambda *args: nullcontext(None))
    h = HandEye(workspace)
    h.path, h.model, h.revision, h.active = tmp_path, model, "workspace", True
    h.feed = SimpleNamespace(status=lambda: {"connected": True}, close=lambda: None)
    h.data = {
        "samples": samples[:4],
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
    h.persist()
    yield h
    h.close()
    h.assessment_executor.shutdown(wait=True, cancel_futures=True)


def assessment(samples, *, ready):
    return {
        "ready": ready,
        "reason": "Validated" if ready else "Need better prediction accuracy",
        "solution": solve(samples) if ready else None,
        "sample_count": len(samples),
    }


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_slow_fit_keeps_state_responsive_and_pause_discards_its_result(runtime, monkeypatch):
    h = runtime
    entered, release = threading.Event(), threading.Event()
    ready = assessment(h.data["samples"], ready=True)

    def slow_assessment(samples, model, **kwargs):
        entered.set()
        assert release.wait(3)
        return ready

    monkeypatch.setattr("stereo_calibration.handeye_runtime.training_assessment", slow_assessment)
    with h.lock:
        h._schedule_assessment()
    assert entered.wait(1)
    try:
        started = time.monotonic()
        state = h.state()
        assert time.monotonic() - started < 0.2
        assert state["assessing"] and not state["capture_ready"]
        assert state["workflow_automatic"]
        queued = h.command("capture", "workspace")
        assert queued["capture_issue"]["status"] == "retry_waiting"
        assert queued["assessing"]
        h.close()
        release.set()
        h.assessment_future.result(timeout=2)
        assert h.data["stage"] == "training" and h.data["solution"] is None
        assert not list(h.path.glob("solution_*.json"))
    finally:
        release.set()


def test_exclusion_invalidates_in_flight_fit_and_only_latest_snapshot_is_applied(
    runtime, monkeypatch
):
    h = runtime
    entered, release = threading.Event(), threading.Event()
    calls = []
    ready = assessment(h.data["samples"], ready=True)

    def assess(samples, model, **kwargs):
        calls.append([s["id"] for s in samples])
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            return ready
        return assessment(samples, ready=False)

    monkeypatch.setattr("stereo_calibration.handeye_runtime.training_assessment", assess)
    with h.lock:
        h._schedule_assessment()
    assert entered.wait(1)
    try:
        h.command("undo", "workspace")
        release.set()
        wait_until(lambda: not h.assessing)
        assert calls == [["0", "1", "2", "3"], ["0", "1", "2"]]
        assert h.readiness["sample_count"] == 3 and not h.readiness["ready"]
        assert h.data["stage"] == "training" and h.data["solution"] is None
        assert not list(h.path.glob("solution_*.json"))
    finally:
        release.set()


def test_resume_assessment_freezes_only_ready_training_and_keeps_auto_armed(runtime, monkeypatch):
    h = runtime
    original = copy.deepcopy(h.data["samples"])
    h.auto_capture = True
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.training_assessment",
        lambda samples, model, **kwargs: assessment(samples, ready=True),
    )
    with h.lock:
        h._schedule_assessment()
    wait_until(lambda: h.data["stage"] == "checking")
    assert h.auto_capture and h.data["report"] is None
    assert h.data["samples"] == original
    assert h.state()["readiness"]["ready"]
    assert h.state()["guidance"]["key"] != "done"
    assert h.data["solution"]["training_ids"] == [s["id"] for s in original]
    assert not (h.path / "handeye.npz").exists()


def test_archived_runs_are_not_offered_for_resume(runtime, tmp_path):
    h = runtime
    parent = tmp_path / "handeye"
    for name in ("archived", "current"):
        path = parent / name
        path.mkdir(parents=True)
        (path / "run.json").write_text(
            json.dumps(
                {
                    "side": "right",
                    "samples": [],
                    "created_at": "2026-10-12",
                }
            )
        )
    (parent / "archived/archive.json").write_text("{}")
    h.workspace.access = lambda *args: nullcontext(
        SimpleNamespace(
            session=SimpleNamespace(path=tmp_path),
        )
    )
    assert [r["id"] for r in h.runs()] == ["current"]


def test_accuracy_review_stops_auto_recording_and_rejects_rearm(runtime, monkeypatch):
    h = runtime
    h.auto_capture = True
    inconsistent = {
        "ready": False,
        "reason": "Prediction exceeds the error budget.",
        "coverage": {"sufficient": True},
        "status": "accuracy",
        "solution": None,
    }
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.training_assessment",
        lambda samples, model, **kwargs: inconsistent,
    )
    with h.lock:
        h._schedule_assessment()
    wait_until(lambda: not h.assessing)
    assert not h.auto_capture and h.data["stage"] == "training"
    assert h.state()["guidance"]["key"] == "review"
    issue = h.state()["capture_issue"]
    assert issue["kind"] == "training_accuracy" and issue["status"] == "retry_required"
    h.message = "Ready to capture a new pose."
    assert h.state()["message"] == inconsistent["reason"]
    assert h.state()["recording_review"] == inconsistent
    assert h.state()["capture_issue"] == issue
    with pytest.raises(ValueError, match="Review this inconsistency"):
        h.command("auto_start", "workspace")


def test_prior_failure_evidence_is_passed_only_as_separate_validation(runtime, monkeypatch):
    h = runtime
    evidence = {"samples": [{"id": "old-failed", "role": "check"}], "source_run_id": "archive"}
    h.data["prior_validation_evidence"] = evidence
    received = []

    def assess(samples, model, *, prior_validation=None):
        received.append((samples, prior_validation))
        return assessment(samples, ready=False)

    monkeypatch.setattr("stereo_calibration.handeye_runtime.training_assessment", assess)
    with h.lock:
        h._schedule_assessment()
    wait_until(lambda: not h.assessing)
    training, prior = received[0]
    assert prior == evidence and prior is not evidence
    assert "old-failed" not in [s["id"] for s in training]


def test_reassessment_clears_stale_training_stop_without_qualifying_or_arming(runtime, monkeypatch):
    h = runtime
    before = copy.deepcopy(h.data["samples"])
    h._capture_issue("training_accuracy", "Previous fit exceeded the pixel budget.")
    issue_id = h.data["capture_issue"]["id"]
    provisional = {
        "ready": False, "status": "stability", "solution": None,
        "reason": "Add another tilt; the current fit remains provisional.",
        "coverage": {"sufficient": True},
        "cross_validation_coverage_sufficient": False,
        "fit_max_rms_px": 3.09, "threshold_px": 3,
        "leave_one_out": [{"coverage_sufficient": False, "passed": False}],
    }
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.training_assessment",
        lambda samples, model, **kwargs: provisional,
    )
    with h.lock:
        h._schedule_assessment()
    wait_until(lambda: not h.assessing)
    assert h.data["capture_issue"]["id"] == issue_id
    assert h.data["capture_issue"]["status"] == "resolved"
    assert h.state()["recording_review"] is None
    assert h.data["samples"] == before and h.data["stage"] == "training"
    assert h.data["solution"] is None and not h.auto_capture
    assert not list(h.path.glob("solution_*.json"))
    assert "training_accuracy_review_cleared" in (h.path / "events.jsonl").read_text()
    h.command("auto_start", "workspace")
    assert h.auto_capture


def test_provisional_assessment_keeps_ongoing_capture_active(runtime, monkeypatch):
    h = runtime
    h.auto_capture = True
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.training_assessment",
        lambda samples, model, **kwargs: {
            "ready": False, "status": "stability", "solution": None,
            "reason": "More independent tilts are needed.",
            "coverage": {"sufficient": True},
            "cross_validation_coverage_sufficient": False,
            "fit_max_rms_px": 3.09, "threshold_px": 3,
            "leave_one_out": [{"coverage_sufficient": False, "passed": False}],
        },
    )
    with h.lock:
        h._schedule_assessment()
    wait_until(lambda: not h.assessing)
    assert h.auto_capture and not h.state()["recording_review"]
    assert h.data["stage"] == "training" and h.data["solution"] is None


def review_measurement(h):
    h.readiness = {
        "ready": False,
        "status": "accuracy",
        "coverage": {"sufficient": True},
        "reason": "Full-fit prediction error exceeds 3 px.",
    }
    sample = h.data["samples"][-1]
    return {
        "frame_id": 1,
        "camera_monotonic": time.monotonic(),
        "measured_monotonic": time.monotonic(),
        "T_camera_board": copy.deepcopy(sample["T_camera_board"]),
        "signature": describe(Observation.from_dict(sample["observation"]), L20_BOARD, h.model),
        "timing": {"fresh_at_ingress": True, "processing_ms": 50},
        "ready": False,
        "reason": "Measured wrist is moving.",
    }


def test_review_has_independent_camera_placement_without_arming_or_mutating_samples(
    runtime, monkeypatch
):
    h = runtime
    before = copy.deepcopy(h.data["samples"])
    h.measurement = review_measurement(h)
    h._update_placement_guidance(h.measurement)
    state = h.state()
    placement = state["placement_guidance"]
    assert state["guidance"]["key"] == "review" and state["recording_review"]
    assert placement["available"] and placement["advisory_only"]
    assert placement["key"].startswith("placement-local-")
    assert placement["pose_preview"] is not None
    assert placement["visibility"]["feasible"]
    assert placement["angle_error_deg"] <= GUIDE_POLICY["local_rotation_step_deg"] + 1e-6
    assert placement["position_target"] == [
        h.measurement["signature"]["x"],
        h.measurement["signature"]["y"],
    ]
    assert "Camera-only" in placement["scope"]
    assert "capture" not in placement["instruction"].lower()
    assert not state["auto_capture"] and not state["pending"] and not state["capture_ready"]
    with pytest.raises(ValueError, match="Review this inconsistency"):
        h.command("auto_start", "workspace")
    assert h.data["samples"] == before
    # Browser polling must neither search again nor advance/change the target.
    monkeypatch.setattr(
        h.placement_local_guide, "guide", lambda *a, **k: pytest.fail("poll recomputed placement")
    )
    for _ in range(3):
        assert h.state()["placement_guidance"] == placement


def test_camera_placement_keeps_goal_until_capture_or_next_and_hides_dropouts(runtime):
    h = runtime
    h.measurement = review_measurement(h)
    h._update_placement_guidance(h.measurement)
    original = h.state()["placement_guidance"]
    h.measurement["frame_id"] += 1
    h._update_placement_guidance(h.measurement)
    current = h.state()["placement_guidance"]
    assert current["key"] == original["key"]
    assert current["target_rotation"] == original["target_rotation"]
    h.command("guide_next", "workspace")
    assert not h.state()["placement_guidance"]["available"]  # Wait for a new measured view.
    h.measurement["frame_id"] += 1
    h._update_placement_guidance(h.measurement)
    assert h.state()["placement_guidance"]["key"] != original["key"]
    for mode in ("stale", "missing_pose", "stale_ingress", "slow_processing", "missing_frame"):
        h.measurement = review_measurement(h)
        h._update_placement_guidance(h.measurement)
        if mode == "stale":
            h.measurement["measured_monotonic"] -= GUIDE_POLICY["maximum_detection_gap_s"] + 0.1
        elif mode == "missing_pose":
            h.measurement.pop("T_camera_board")
        elif mode == "stale_ingress":
            h.measurement["timing"]["fresh_at_ingress"] = False
            h._update_placement_guidance(h.measurement)
        elif mode == "slow_processing":
            h.measurement["timing"]["processing_ms"] = 701
        else:
            h.measurement = None
        guide = h.state()["placement_guidance"]
        assert guide["key"] == "unavailable" and not guide["available"]
        assert guide["target_rotation"] is None and guide["position_target"] is None
        assert guide["pose_preview"] is None
        assert h.state()["guidance"]["key"] == "review"
    h.readiness = None
    assert h.state()["placement_guidance"] is None


def test_partial_board_placement_explains_missing_corners_without_invented_target(runtime):
    h = runtime
    h.measurement = review_measurement(h)
    h.measurement.pop("T_camera_board")
    h.measurement.update(shared=9, reason="Need 12 shared corners; seeing 9.")
    h._update_placement_guidance(h.measurement)
    guide = h.state()["placement_guidance"]
    assert guide["key"] == "unavailable" and guide["pose_preview"] is None
    assert guide["target_rotation"] is None and guide["position_target"] is None
    assert "9/12 shared corners" in guide["instruction"]
    assert h.state()["guidance"]["key"] == "review" and not h.auto_capture
    for counts, eye in [([6, 12], "left"), ([12, 6], "right")]:
        h.measurement.update(corners=counts, shared=6)
        guide = h.state()["placement_guidance"]
        assert f"{eye} camera sees 6/12" in guide["instruction"]
        assert guide["visibility_counts"] == {
            "left": counts[0],
            "right": counts[1],
            "shared": 6,
            "expected": 12,
        }
        assert guide["target_rotation"] is None and guide["pose_preview"] is None


def partial_view(h, count=6, *, right_error=0):
    obs = Observation.from_dict(h.data["samples"][-1]["observation"])
    obs.left_ids, obs.left_points = obs.left_ids[:count], obs.left_points[:count]
    obs.right_points[:, 0] += right_error
    return obs


@pytest.mark.parametrize("initialize_partial", [False, True])
def test_partial_stereo_checked_pose_initializes_or_tracks_same_advisory_goal(
    runtime, initialize_partial
):
    h = runtime
    before = copy.deepcopy(h.data["samples"])
    h.measurement = review_measurement(h)
    if not initialize_partial:
        h._update_placement_guidance(h.measurement)
        original = h.state()["placement_guidance"]
    obs = partial_view(h)
    h.measurement.pop("T_camera_board")
    h.measurement.update(
        frame_id=2,
        shared=6,
        corners=[6, 12],
        signature=describe(obs, L20_BOARD, h.model),
    )
    h._update_placement_guidance(h.measurement, obs)
    placement = h.state()["placement_guidance"]
    assert placement["available"] and placement["pose_preview"] is not None
    assert placement["pose_quality"]["partial"]
    assert placement["source"] == "partial_stereo_checked_left_pnp"
    assert placement["pose_quality"]["shared_corners"] == 6
    assert placement["visibility_counts"] == {"left": 6, "right": 12, "shared": 6, "expected": 12}
    assert placement["visibility"]["feasible"]
    if not initialize_partial:
        assert placement["key"] == original["key"]
        assert placement["target_rotation"] == original["target_rotation"]
        np.testing.assert_allclose(placement["position_target"], original["position_target"])
    assert "T_camera_board" not in h.measurement
    assert h.measurement["T_camera_board_advisory"]
    assert not h.state()["capture_ready"] and not h.auto_capture
    assert h.state()["guidance"]["key"] == "review" and h.data["samples"] == before


@pytest.mark.parametrize("count,right_error", [(5, 0), (6, 20)])
def test_partial_advisory_refuses_insufficient_or_cross_camera_inconsistent_support(
    runtime, count, right_error
):
    h = runtime
    h.measurement = review_measurement(h)
    h.measurement.pop("T_camera_board")
    obs = partial_view(h, count, right_error=right_error)
    h.measurement.update(
        signature=describe(obs, L20_BOARD, h.model), shared=count, corners=[count, 12]
    )
    h._update_placement_guidance(h.measurement, obs)
    placement = h.state()["placement_guidance"]
    assert not placement["available"] and placement["pose_preview"] is None
    assert "T_camera_board_advisory" not in h.measurement
    assert h.measurement["placement_pose_reason"]


def test_partial_processing_updates_advisory_numbers_but_cannot_record(runtime):
    h = runtime
    original = copy.deepcopy(h.data["samples"])
    review_measurement(h)
    obs = partial_view(h)
    h.controller = SimpleNamespace(camera=SimpleNamespace(swap=False))
    h.pending = time.monotonic()
    detector = SimpleNamespace(observe=lambda *args, **kwargs: (obs, []))
    h._process(Frame(1, time.monotonic(), np.zeros((720, 2560, 3), np.uint8)), detector)
    placement = h.state()["placement_guidance"]
    assert placement["available"] and placement["pose_quality"]["partial"]
    assert h.pending is not None and h.data["samples"] == original
    assert not h.measurement["ready"] and "T_camera_board" not in h.measurement
    assert "seeing 6" in h.measurement["reason"]
    # A newly observed small tilt changes the numeric delta, not the fixed goal.
    target = copy.deepcopy(placement["target_rotation"])
    T = np.asarray(h.measurement["T_camera_board_advisory"])
    T[:3, :3] = T[:3, :3] @ cv2.Rodrigues(np.array([0.005, 0, 0]))[0]
    objects = L20_BOARD.create().getChessboardCorners().astype(float)
    for eye, pose, K, D in [
        ("left", T, h.model.K1, h.model.D1),
        (
            "right",
            np.block(
                [[h.model.R, h.model.T.reshape(3, 1) / 1000], [np.zeros((1, 3)), np.ones((1, 1))]]
            )
            @ T,
            h.model.K2,
            h.model.D2,
        ),
    ]:
        ids = getattr(obs, f"{eye}_ids")
        pixels = cv2.fisheye.projectPoints(
            objects[ids].reshape(-1, 1, 3), cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3] * 1000, K, D
        )[0].reshape(-1, 2)
        setattr(obs, f"{eye}_points", pixels)
    h._process(Frame(2, time.monotonic(), np.zeros((720, 2560, 3), np.uint8)), detector)
    moved = h.state()["placement_guidance"]
    assert moved["target_rotation"] == target
    assert moved["current_rotation"] != placement["current_rotation"]
    assert h.data["samples"] == original and not h.measurement["ready"]


def test_repeated_pose_exposes_paired_measured_angles_without_recording(runtime):
    h = runtime
    original = copy.deepcopy(h.data["samples"])
    sample = original[-1]
    obs = Observation.from_dict(sample["observation"])
    feedback = {"names": [f"joint{i}" for i in range(1, 8)], "positions": [0.1] * 7}
    pairing = {
        "nearest": feedback,
        "pair_skew_s": 0.004,
        "stationary": True,
        "joint_range_rad": 0.0,
        "fk_policy": "measured_angles_unmodified; nominal_limits_are_diagnostics_only",
    }
    h.feed.paired = lambda *args, **kwargs: (np.asarray(sample["T_base_tool"]), pairing)
    h.kinematics = object()
    h.controller = SimpleNamespace(camera=SimpleNamespace(swap=False))
    detector = SimpleNamespace(observe=lambda *args, **kwargs: (obs, []))
    h._process(Frame(1, time.monotonic(), np.zeros((720, 2560, 3), np.uint8)), detector)
    state = h.state()
    paired = state["measurement"]["paired_robot"]
    assert paired["feedback"] == feedback
    assert paired["T_base_tool"] == sample["T_base_tool"]
    assert paired["stationary"] and paired["pair_skew_s"] == 0.004
    assert "already captured" in state["measurement"]["reason"]
    assert not state["capture_ready"] and not state["auto_capture"]
    assert h.data["samples"] == original
    feedback["positions"][0] = 2.0
    assert h.measurement["paired_robot"]["feedback"]["positions"][0] == 0.1


def test_capture_sequence_survives_archiving_rejected_records(runtime):
    h = runtime
    h.data["samples"] = [{**h.data["samples"][0], "id": "0017_retained"}]
    assert h._next_sample_index() == 18
    h.data["next_sample_index"] = 25
    assert h._next_sample_index() == 25
    h.data["samples"] = []
    assert h._next_sample_index() == 25
