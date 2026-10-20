"""Independent ground truth, malformed inputs, and capture boundary checks."""

import time

import cv2
import numpy as np
import pytest

from stereo_calibration.calibration import Calibration
from stereo_calibration.config import BoardConfig
from stereo_calibration.handeye import (
    L20_BOARD,
    POLICY,
    ArmKinematics,
    camera_board_pose,
    checked_transform,
    difference,
    inverse,
    novel,
    solve,
    transform,
    validation_report,
)
from stereo_calibration.handeye_runtime import PoseFeed
from stereo_calibration.observations import Observation
from stereo_calibration.rectification import horizontal_rectification


def fixture_samples(count=20, noise=0.0):
    rng = np.random.default_rng(71)
    size = (1280, 720)
    K = np.array([[590.0, 0, 640], [0, 588, 360], [0, 0, 1]])
    D = np.array([-0.03, 0.002, -0.0001, 0.00001])
    R = cv2.Rodrigues(np.array([0.002, -0.007, 0.001]))[0]
    t = np.array([-62.0, 0.4, -0.2])
    model = Calibration(
        size,
        K,
        D,
        K.copy(),
        D.copy(),
        R,
        t,
        *horizontal_rectification(K, D, K, D, size, R, t),
        (0.2, 0.2, 0.2),
    )
    X = transform(cv2.Rodrigues(np.array([0.45, -0.3, 0.21]))[0], [0.37, -0.28, 0.62])
    Z = transform(cv2.Rodrigues(np.array([-0.2, 0.42, 0.51]))[0], [0.01, -0.06, 0.15])
    objects = L20_BOARD.create().getChessboardCorners().astype(float)
    ids = np.arange(12, dtype=np.int32)
    samples = []
    for i in range(count):
        C = transform(
            cv2.Rodrigues(rng.uniform(-0.65, 0.65, 3))[0],
            rng.uniform([-0.10, -0.10, 0.35], [0.10, 0.05, 0.55]),
        )
        G = X @ C @ inverse(Z)
        right = transform(R, t / 1000) @ C
        points = []
        for T in [C, right]:
            p = cv2.fisheye.projectPoints(
                objects.reshape(-1, 1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3] * 1000, K, D
            )[0].reshape(-1, 2)
            points.append(p + rng.normal(0, noise, p.shape))
        obs = Observation(str(i), size, ids.copy(), points[0], ids.copy(), points[1])
        measured, _ = camera_board_pose(obs, model)
        samples.append(
            {
                "id": str(i),
                "role": "training" if i < 12 else "check",
                "T_base_tool": G.tolist(),
                "T_camera_board": measured.tolist(),
                "observation": obs.to_dict(),
            }
        )
    return samples, model, X, Z


def test_eye_to_hand_recovers_known_direction_units_mount_and_rectified_convention():
    samples, model, X, Z = fixture_samples()
    solution = solve(samples)
    np.testing.assert_allclose(solution["T_base_camera_left_raw"], X, atol=1e-7)
    np.testing.assert_allclose(solution["T_tool_board"], Z, atol=1e-7)
    report = validation_report(samples, solution, model)
    assert report["passed"] and len(report["checks"]) == 8
    assert max(c["left_rms_px"] for c in report["checks"]) < 1e-4
    # Arbitrary point: rectified -> raw -> base equals exported rectified composition.
    rectified = X @ transform(model.R1.T, np.zeros(3))
    p = np.array([0.2, 0.1, 0.4, 1])
    np.testing.assert_allclose(rectified @ p, X @ np.r_[model.R1.T @ p[:3], 1])


def test_fresh_checks_do_not_enter_fit_and_failure_cannot_disappear():
    samples, model, _, _ = fixture_samples()
    original = solve(samples)
    samples[12]["T_base_tool"][0][3] += 0.05
    np.testing.assert_allclose(
        solve(samples)["T_base_camera_left_raw"], original["T_base_camera_left_raw"]
    )
    report = validation_report(samples, original, model)
    assert not report["passed"] and not report["checks"][0]["passed"]
    assert all(c["passed"] for c in report["checks"][1:])


def test_mild_pixel_noise_and_insufficient_checks():
    samples, model, X, _ = fixture_samples(noise=0.08)
    solution = solve(samples)
    assert difference(solution["T_base_camera_left_raw"], X)[0] < 0.005
    assert not validation_report(samples[:14], solution, model)["passed"]
    assert validation_report(samples, solution, model)["passed"]


def test_pure_translation_rejected_even_with_many_samples():
    samples, _, _, _ = fixture_samples()
    for s in samples:
        p = np.array(s["T_base_tool"])
        p[:3, :3] = np.eye(3)
        s["T_base_tool"] = p.tolist()
    with pytest.raises(ValueError, match="two different axes"):
        solve(samples)
    with pytest.raises(ValueError, match="[Aa]t least 3"):
        solve(samples[:2])


def test_capture_rejects_preview_wrong_stereo_and_missing_corners():
    samples, model, _, _ = fixture_samples()
    obs = Observation.from_dict(samples[0]["observation"])
    obs.image_size = (800, 500)
    with pytest.raises(ValueError, match="Native"):
        camera_board_pose(obs, model)
    obs = Observation.from_dict(samples[0]["observation"])
    obs.right_points += [20, 0]
    with pytest.raises(ValueError, match="disagree"):
        camera_board_pose(obs, model)
    obs = Observation.from_dict(samples[0]["observation"])
    obs.left_ids, obs.left_points = obs.left_ids[:8], obs.left_points[:8]
    with pytest.raises(ValueError, match="12"):
        camera_board_pose(obs, model)


def test_board_print_layout_ids_and_old_sessions_unchanged():
    assert "marker_ids" not in BoardConfig().to_dict()
    board = L20_BOARD.create()
    np.testing.assert_array_equal(board.getIds().ravel(), range(20, 30))
    image = board.generateImage((400, 500), marginSize=0)
    image = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    corners, _ids, markers, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(image)
    assert len(corners) == 12 and len(markers) == 10
    assert set(marker_ids.ravel()) == set(range(20, 30))
    assert BoardConfig(**L20_BOARD.to_dict()) == L20_BOARD
    with pytest.raises(ValueError):
        BoardConfig(marker_ids=(1, 1))


def test_transform_and_novelty_guards():
    with pytest.raises(ValueError):
        checked_transform(np.diag([1, 1, -1, 1]))
    with pytest.raises(ValueError):
        checked_transform(np.full((4, 4), np.nan))
    assert not novel(np.eye(4), [np.eye(4)])
    assert novel(transform(np.eye(3), [0.02, 0, 0]), [np.eye(4)])
    assert novel(transform(cv2.Rodrigues(np.array([0.0, 0.2, 0]))[0], [0, 0, 0]), [np.eye(4)])


def simple_fk(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test"><joint name="yaw" type="revolute">
    <parent link="base_link"/><child link="link1"/><origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2"/></joint>
    <joint name="tip" type="fixed"><parent link="link1"/><child link="link7"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/></joint></robot>""")
    return ArmKinematics(path)


def test_fk_name_order_units_and_missing_feedback(tmp_path):
    k = simple_fk(tmp_path)
    out = k.pose(["unused", "yaw"], [100, np.pi / 2])
    np.testing.assert_allclose(out[:3, 3], [1, 0.5, 0], atol=1e-12)
    with pytest.raises(ValueError, match="every arm joint"):
        k.pose(["other"], [0])
    # Measured-state FK is not a command validator and must never clip feedback.
    actual = k.pose(["yaw"], [4])
    np.testing.assert_allclose(actual[:3, 3], [1 + 0.5 * np.cos(4), 0.5 * np.sin(4), 0])
    assert k.limit_diagnostics(["yaw"], [4])[0]["outside_by_rad"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        k.pose(["yaw"], [float("nan")])


def test_joint4_reported_angle_above_nominal_command_limit_is_recorded_unmodified(tmp_path):
    path = simple_fk(tmp_path).path
    path.write_text(
        path.read_text()
        .replace('name="yaw"', 'name="joint4"')
        .replace('lower="-3.2" upper="3.2"', 'lower="-1.01" upper="2.14"')
    )
    k = ArmKinematics(path)
    measured = 2.194402468532471
    pose = k.pose(["joint4"], [measured])
    np.testing.assert_allclose(pose[:3, 3], [1 + 0.5 * np.cos(measured), 0.5 * np.sin(measured), 0])
    diagnostic = k.limit_diagnostics(["joint4"], [measured])[0]
    assert diagnostic["measured_rad"] == measured
    assert diagnostic["upper_rad"] == 2.14
    assert diagnostic["outside_by_rad"] == pytest.approx(0.054402468532471)
    # Pairing/capture receives the original angle and the diagnostic, not a boundary pose.
    feed = PoseFeed("right", tmp_path)
    now = time.monotonic()
    for row in feed_rows(now):
        row.update(names=["joint4"], positions=[measured])
        feed.history.append(row)
    paired_pose, pairing = feed.paired(now, k)
    np.testing.assert_allclose(paired_pose, pose)
    assert pairing["joint_limit_diagnostics"] == [diagnostic]
    assert pairing["nearest"]["positions"] == [measured]


def feed_rows(now, moving=False):
    rows = []
    for i in range(70):
        timestamp = now - 0.9 + i * 0.015
        row = {
            "topic": "/slave_right/joint_states_single",
            "names": ["yaw"],
            "positions": [i * 0.003 if moving else 0.1],
            "stamp_s": time.time() - now + timestamp,
            "received_wall": time.time() - now + timestamp,
            "received_monotonic": timestamp,
            "sample_monotonic": timestamp,
        }
        rows.append(row)
    return rows


def test_pairing_stationary_window_missing_stale_wrong_source_and_movement(tmp_path):
    k = simple_fk(tmp_path)
    feed = PoseFeed("right", tmp_path)
    now = time.monotonic()
    rows = feed_rows(now)
    feed.history.extend(rows)
    _pose, pair = feed.paired(now, k)
    assert pair["pair_skew_s"] < 0.02
    assert pair["joint_range_rad"] == 0
    feed.history.clear()
    feed.history.extend(feed_rows(now, moving=True))
    with pytest.raises(ValueError, match="moving") as blocked:
        feed.paired(now, k)
    assert blocked.value.code == "motion"
    assert blocked.value.details["joint_range_rad"] > POLICY["stationary_joint_range_rad"]
    assert blocked.value.details["motion_joint"] == "yaw"
    advisory_pose, advisory_pair = feed.paired(now, k, require_stationary=False)
    assert not advisory_pair["stationary"]
    np.testing.assert_allclose(
        advisory_pose,
        k.pose(advisory_pair["nearest"]["names"], advisory_pair["nearest"]["positions"]),
    )
    feed.history.clear()
    with pytest.raises(ValueError, match="fresh"):
        feed.paired(now, k)
    with pytest.raises(ValueError, match="fresh"):
        feed.paired(now, k, require_stationary=False)
    row = feed_rows(time.monotonic())[-10]
    feed.add({**row, "topic": "/master_right/joint_states_single"})
    feed.add({**row, "stamp_s": row["stamp_s"] - 100})
    assert not feed.history
    feed.add(row)
    assert len(feed.history) == 1
    feed.add(row)
    assert len(feed.history) == 1  # Repeated SDK stamp is not a fresh reading.


def test_pairing_checks_motion_and_fresh_feedback_after_image_analysis(tmp_path):
    k = simple_fk(tmp_path)
    feed = PoseFeed("right", tmp_path)
    timestamp = time.monotonic()
    feed.history.extend(feed_rows(timestamp))
    for offset in np.arange(0.165, 0.31, 0.015):
        row = dict(feed.history[-1])
        row.update(sample_monotonic=timestamp + offset, positions=[0.12])
        feed.history.append(row)
    # The old image-time window cannot see movement later in image analysis.
    feed.paired(timestamp, k)
    with pytest.raises(ValueError, match="moving"):
        feed.paired(timestamp, k, through=timestamp + 0.31)
    for row in feed.history:
        row["positions"] = [0.1]
    _, pairing = feed.paired(timestamp, k, through=timestamp + 0.31)
    assert pairing["decision_feedback_age_s"] < POLICY["pair_max_skew_s"]
    while feed.history[-1]["sample_monotonic"] > timestamp + 0.14:
        feed.history.pop()
    # A stopped feedback stream can look stationary around image receipt;
    # it is insufficient to confirm the wrist at the processing decision.
    with pytest.raises(ValueError, match="current measured"):
        feed.paired(timestamp, k, through=timestamp + 0.31)


def test_single_axis_rotations_rejected():
    samples, _, _, _ = fixture_samples()
    for i, s in enumerate(samples):
        pose = np.asarray(s["T_base_tool"])
        pose[:3, :3] = cv2.Rodrigues(np.array([i * 0.1, 0.0, 0.0]))[0]
        s["T_base_tool"] = pose.tolist()
    with pytest.raises(ValueError, match="two different axes"):
        solve(samples)


@pytest.mark.parametrize(
    "automatic,corrupt_check,off_suggestion,calibrated_model",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (True, False, False, True),
        (True, False, False, "angle"),
    ],
)
def test_readonly_workflow_api_capture_solve_checks_resume_and_source_guards(
    tmp_path, monkeypatch, automatic, corrupt_check, off_suggestion, calibrated_model
):
    import hashlib
    import json
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from stereo_calibration.camera import Frame
    from stereo_calibration.config import CameraConfig
    from stereo_calibration.handeye_runtime import HandEye
    from stereo_calibration.web import create_app
    from stereo_calibration.workspace import Workspace

    source = simple_fk(tmp_path).path
    monkeypatch.setattr("stereo_calibration.handeye_runtime.ARM_URDF", source)
    monkeypatch.setattr(
        "stereo_calibration.handeye_runtime.ARM_MODEL_PROFILES", tmp_path / "profiles"
    )
    expected_model = source.read_bytes()
    expected_model_name = "arm_model.urdf"
    if calibrated_model:
        from stereo_calibration.handeye_kinematics import ANGLE_PROFILE_SCHEMA, SCHEMA, calibrated_joint_model, calibrated_urdf

        maps = {}
        if calibrated_model == "angle":
            maps = {"yaw": {"model": "relative_quadratic_v1", "reference_rad": 0.,
                            "scale_rad": 1., "coefficients_rad": [.02, .01],
                            "qualified_range_rad": [-3., 3.]}}
            expected_model = calibrated_joint_model(expected_model, {"yaw": 0.12}, maps)
            expected_model_name = "arm_model.json"
        else:
            expected_model = calibrated_urdf(expected_model, {"yaw": 0.12})
        profile_path = tmp_path / "profiles/right"
        profile_path.mkdir(parents=True)
        (profile_path / expected_model_name).write_bytes(expected_model)
        (profile_path / "profile.json").write_text(
            json.dumps(
                {
                    "schema": ANGLE_PROFILE_SCHEMA if maps else SCHEMA,
                    "side": "right",
                    "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "joint_zero_offsets_rad": {"yaw": 0.12},
                    "model_file": expected_model_name,
                    "joint_angle_maps": maps,
                    "model_sha256": hashlib.sha256(expected_model).hexdigest(),
                    "qualification": {"status": "independent_checks_passed"},
                }
            )
        )

    samples, model, X, Z = fixture_samples()

    def suggested_sample(index):
        from stereo_calibration.handeye_guidance import _about_centre, guidance

        reference = transform(np.eye(3), [-0.05, -0.06, 0.5])
        current = (
            np.asarray(h.data["samples"][-1]["T_camera_board"]) if h.data["samples"] else reference
        )
        guide = guidance(h.data["samples"], current, h.data["stage"], model=model)
        C = np.asarray(
            guide.get(
                "target_camera_pose",
                _about_centre(current, reference[:3, :3] @ np.asarray(guide["target_rotation"])),
            )
        )
        right = transform(model.R, model.T / 1000) @ C
        objects = L20_BOARD.create().getChessboardCorners().astype(float)
        points = [
            cv2.fisheye.projectPoints(
                objects.reshape(-1, 1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3] * 1000, K, D
            )[0].reshape(-1, 2)
            for T, K, D in [(C, model.K1, model.D1), (right, model.K2, model.D2)]
        ]
        obs = Observation(
            str(index), model.image_size, np.arange(12), points[0], np.arange(12), points[1]
        )
        return {
            "id": str(index),
            "role": "training",
            "T_camera_board": C.tolist(),
            "T_base_tool": (X @ C @ inverse(Z)).tolist(),
            "observation": obs.to_dict(),
        }

    def wait_assessment():
        deadline = time.monotonic() + 30
        while h.assessing and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not h.assessing

    session_path = tmp_path / "stereo"
    (session_path / "results/frozen").mkdir(parents=True)
    (session_path / "results/frozen/stereo_params.npz").write_bytes(b"fixture frozen calibration")
    candidate = {"id": "frozen", "report": {"passed": True}}
    c = SimpleNamespace(
        session=SimpleNamespace(path=session_path, candidate=candidate),
        replay=False,
        model=model,
        fit_future=None,
        inspection_log=None,
        paused=True,
        camera=CameraConfig(width=2560, height=720),
        source=SimpleNamespace(actual={}, error=None),
        handeye_active=False,
    )
    c.state = lambda: {
        "phase": "done",
        "board": BoardConfig().to_dict(),
        "camera": c.camera.to_dict(),
    }
    workspace = Workspace(output=tmp_path, controller=c)
    feed_instances = []

    class FakeFeed:
        def __init__(self, side, path):
            self.side = side
            self.current = samples[0]
            self.closed = False
            feed_instances.append(self)

        def start(self):
            pass

        def close(self):
            self.closed = True

        def status(self):
            return {"connected": getattr(self, "connected", True), "age_s": 0.01}

        def paired(self, timestamp, kinematics, *, through=None, require_stationary=True):
            if getattr(self, "moving", False):
                raise ValueError("Wrist is moving. Pause briefly to capture this pose.")
            return np.asarray(self.current["T_base_tool"]), {"pair_skew_s": 0.005}

    h = workspace.handeye = HandEye(workspace, feed_factory=FakeFeed)
    monkeypatch.setattr(h, "_run", lambda: h.stop_event.wait())
    client = TestClient(create_app(workspace), base_url="http://127.0.0.1:8765")
    revision = workspace.revision
    body = {"workspace_id": revision, "side": "right"}
    assert client.get("/hand-eye").status_code == 200
    assert (
        client.post("/api/handeye/start", json={**body, "workspace_id": "old"}).status_code == 409
    )
    c.replay = True
    assert client.post("/api/handeye/start", json=body).status_code == 409
    c.replay = False
    try:
        response = client.post("/api/handeye/start", json=body)
        assert response.status_code == 200, response.text
        wait_assessment()
        path = h.path
        assert (path / expected_model_name).read_bytes() == expected_model
        assert h.kinematics.sha256 == hashlib.sha256(expected_model).hexdigest()
        assert bool(h.data["kinematic_calibration"]) == bool(calibrated_model)
        assert c.handeye_active
        assert (
            client.post(
                "/api/control", json={"workspace_id": revision, "action": "resume"}
            ).status_code
            == 409
        )
        assert client.post("/api/sessions/open", json={"workspace_id": revision}).status_code == 409
        if not automatic:
            queued = h.command("capture", revision)
            assert queued["capture_issue"]["status"] == "retry_waiting"
            h.command("pause", revision)
            rejected_issue = h.state()["capture_issue"]
            resumed = client.post("/api/handeye/start", json={**body, "run_id": path.name})
            assert resumed.status_code == 200, resumed.text
            wait_assessment()
            assert h.state()["capture_issue"] == rejected_issue
        frame = np.zeros((720, 2560, 3), np.uint8)

        class FakeDetector:
            def observe(self, image, swap, full):
                obs = Observation.from_dict(feed_instances[-1].current["observation"])
                if getattr(self, "hidden", False):
                    obs.left_ids, obs.left_points = obs.left_ids[:6], obs.left_points[:6]
                return obs, []

        detector = FakeDetector()
        clock = [time.monotonic()]
        if automatic:
            monkeypatch.setattr(
                "stereo_calibration.handeye_runtime.time",
                SimpleNamespace(monotonic=lambda: clock[0]),
            )
            h.command("auto_start", revision)
            for j in range(2):
                clock[0] += 0.55
                h._process(Frame(-10 + j, clock[0], frame), detector)
            assert h.state()["hold_progress"] > 0 and not h.data["samples"]
            for owner, field in [
                (feed_instances[-1], "connected"),
                (feed_instances[-1], "moving"),
                (detector, "hidden"),
            ]:
                setattr(owner, field, field != "connected")
                clock[0] += 0.55
                h._process(Frame(-7, clock[0], frame), detector)
                assert not h.data["samples"] and h.state()["hold_progress"] == 0
                assert not h.state()["capture_ready"]
                setattr(owner, field, field == "connected")
            stopped = client.post(
                "/api/handeye/control", json={"workspace_id": revision, "action": "auto_stop"}
            )
            assert stopped.status_code == 200 and not stopped.json()["auto_capture"]
            for j in range(3):
                clock[0] += 0.55
                h._process(Frame(-3 + j, clock[0], frame), detector)
            assert not h.data["samples"]  # Valid stationary images do not arm recording.
        frozen_training_ids = None
        for i, sample in enumerate(samples):
            if h.data["stage"] == "complete":
                break
            if automatic and not off_suggestion:
                sample = suggested_sample(i)
            is_first_check = h.data["stage"] == "checking" and not h.data["report"]
            if is_first_check:
                frozen_training_ids = list(h.data["solution"]["training_ids"])
                assert h.state()["guidance"]["count"] == 0
                assert not h.state()["diversity"]["sufficient"]
                assert not (path / "handeye.npz").exists()
                if automatic:
                    assert h.auto_capture  # No Next/Solve or second Start is needed.
            if corrupt_check and is_first_check:
                sample["T_base_tool"][0][3] += 0.03
            feed_instances[-1].current = sample
            if automatic:
                if i == 0:
                    response = client.post(
                        "/api/handeye/control",
                        json={"workspace_id": revision, "action": "auto_start"},
                    )
                    assert response.status_code == 200, response.text
                assert h.auto_capture
                for j in range(3):
                    clock[0] += 0.55
                    h._process(Frame(i * 4 + j, clock[0], frame), detector)
                    if j < 2:
                        assert len(h.data["samples"]) == i
                assert h.data["samples"][-1]["capture_mode"] == "guided"
                recent_frames = (i * 4 + 1, i * 4 + 2)
                latest_frame = i * 4 + 2
                # Remaining at the captured pose must not save another image.
                clock[0] += 0.55
                h._process(Frame(i * 4 + 3, clock[0], frame), detector)
                assert h.state()["hold_progress"] == 0
                latest_frame = i * 4 + 3
            else:
                h._process(Frame(i * 2, time.monotonic(), frame), detector)
                assert h.state()["capture_ready"]
                response = client.post(
                    "/api/handeye/control", json={"workspace_id": revision, "action": "capture"}
                )
                assert response.status_code == 200, response.text
                h._process(Frame(i * 2 + 1, time.monotonic(), frame), detector)
                recent_frames = (i * 2, i * 2 + 1)
                latest_frame = i * 2 + 1
            assert len(h.data["samples"]) == i + 1
            wait_assessment()
            if off_suggestion and h.data["stage"] == "checking":
                assert any(not s["guidance"]["at_target"] for s in h.data["samples"])
                assert all(s["capture_mode"] == "guided" for s in h.data["samples"])
            if corrupt_check and is_first_check:
                assert h.data["stage"] == "checking" and not h.auto_capture
                assert not h.data["report"]["checks"][0]["passed"]
                assert h.state()["guidance"]["key"] == "review"
                assert h.state()["capture_issue"]["kind"] == "check_failed"
                assert h.state()["message"] == h.state()["recording_review"]["reason"]
                assert (
                    "independent prediction check failed" in h.state()["capture_issue"]["message"]
                )
                assert "move to the next pose" not in h.state()["message"]
                assert (
                    client.post(
                        "/api/handeye/control",
                        json={"workspace_id": revision, "action": "auto_start"},
                    ).status_code
                    == 409
                )
                assert not (path / "handeye.npz").exists()
                assert h.data["solution"]["training_ids"] == frozen_training_ids
                assert (
                    client.post(
                        "/api/handeye/control", json={"workspace_id": revision, "action": "undo"}
                    ).status_code
                    == 409
                )
                return  # Failure remains saved; recording stopped for review.
            # Latest and immediately preceding images both retain their own IDs.
            for frame_id in recent_frames:
                response = client.get(
                    "/api/handeye/frame.jpg",
                    params={"workspace_id": revision, "frame_id": frame_id},
                )
                assert response.status_code == 200
                assert response.headers["X-Frame-Id"] == str(frame_id)
            inspection = h.state()["measurement"]["inspection"]
            assert inspection["frame_id"] == latest_frame
            assert len(inspection["points"]["left"]) == 12
            np.testing.assert_allclose(
                [p["raw"] for p in inspection["points"]["left"]],
                sample["observation"]["left_points"],
            )
        assert h.state()["stage"] == "complete"
        assert not h.auto_capture
        assert (
            client.get(
                "/api/handeye/frame.jpg", params={"workspace_id": revision, "frame_id": 0}
            ).status_code
            == 404
        )
        board_png = client.get("/api/handeye/board.png")
        assert board_png.status_code == 200
        image = cv2.imdecode(np.frombuffer(board_png.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        _, corner_ids, _, marker_ids = cv2.aruco.CharucoDetector(L20_BOARD.create()).detectBoard(
            image
        )
        assert len(corner_ids) == 12 and set(marker_ids.ravel()) == set(range(20, 30))
        assert h.data["report"]["passed"]
        assert (path / "handeye.npz").exists()
        with np.load(path / "handeye.npz") as exported:
            assert str(exported["urdf_sha256"]) == hashlib.sha256(expected_model).hexdigest()
            assert str(exported["arm_model_file"]) == str(path / expected_model_name)
        assert (
            json.loads((path / "result.json").read_text())["arm_model"]["sha256"]
            == h.kinematics.sha256
        )
        captured = json.loads((path / "run.json").read_text())
        assert captured["solution"]["training_ids"] == frozen_training_ids
        assert 4 <= len(frozen_training_ids) < 12
        assert h.state()["workflow_automatic"]
        assert h.state()["kinematic_calibration"] == h.data.get("kinematic_calibration")
        assert h.state()["arm_model_file"] == expected_model_name
        model_download = client.get(
            f"/api/handeye/artifacts/{expected_model_name}",
            params={"workspace_id": revision, "run_id": path.name},
        )
        assert model_download.status_code == 200 and model_download.content == expected_model
        assert h.state()["readiness"]["ready"]
        assert all((path / s["image"]).is_file() for s in captured["samples"])
        assert (
            client.get(
                "/api/handeye/artifacts/run.json",
                params={"workspace_id": revision, "run_id": path.name},
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/handeye/artifacts/run.json",
                params={"workspace_id": revision, "run_id": "wrong"},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/handeye/control", json={"workspace_id": revision, "action": "undo"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/handeye/control", json={"workspace_id": revision, "action": "pause"}
            ).status_code
            == 200
        )
        assert not c.handeye_active and feed_instances[-1].closed
        response = client.post("/api/handeye/start", json={**body, "run_id": path.name})
        assert response.status_code == 200, response.text
        assert len(h.data["samples"]) == len(captured["samples"])
        assert not h.auto_capture  # Resuming a session never silently arms recording.
        if calibrated_model:
            h.close()
            # A run must never be resumed under nominal geometry after its
            # calibrated profile has been removed or replaced.
            (profile_path / "profile.json").unlink()
            response = client.post("/api/handeye/start", json={**body, "run_id": path.name})
            assert response.status_code == 409
            assert "provenance" in response.text
            assert json.loads((path / "run.json").read_text())["samples"] == captured["samples"]
            return
        assert (
            client.post("/api/handeye/start", json={**body, "run_id": "../../anything"}).status_code
            == 409
        )
        if not automatic:
            # A saved pass from an older observability/LOO method cannot retain
            # its qualification; preserve it, then assess training separately.
            previous_solution_id = h.data["solution"]["id"]
            h.data["readiness"]["coverage_method"] = "obsolete_exact_rotation_dedup"
            h.persist()
            h.close()
            response = client.post("/api/handeye/start", json={**body, "run_id": path.name})
            assert response.status_code == 200, response.text
            wait_assessment()
            assert h.data["stage"] == "checking" and h.data["report"] is None
            assert h.data["solution"]["id"] != previous_solution_id
            assert h.data["samples"] == captured["samples"]
            assert not (path / "handeye.npz").exists()
            assert not (path / "result.json").exists()
            assert list(path.glob("qualification_before_reassessment_*.json"))
            # Resuming a previously passed run must re-evaluate verified prior
            # evidence; its old executable export cannot retain a pass silently.
            h.close()
            prior = json.loads(
                json.dumps(next(s for s in captured["samples"] if s["role"] == "check"))
            )
            prior["T_base_tool"][0][3] += 0.05
            evidence = {"mount_unchanged": True, "source_run_id": "archived", "samples": [prior]}
            monkeypatch.setattr(
                "stereo_calibration.handeye_runtime.load_prior_validation",
                lambda data, path: evidence,
            )
            response = client.post("/api/handeye/start", json={**body, "run_id": path.name})
            assert response.status_code == 200, response.text
            assert h.data["stage"] == "checking" and not h.data["report"]["passed"]
            assert h.state()["guidance"]["key"] == "review"
            assert h.data["solution"]["training_ids"] == frozen_training_ids
            assert not (path / "handeye.npz").exists()
            assert list((path / "invalidated_exports").glob("handeye_*.npz"))
            assert list(path.glob("validation_before_reassessment_*.json"))
    finally:
        h.close()


def test_camera_warmup_resets_hold_without_disarming_automatic_capture(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from stereo_calibration.camera import Frame
    from stereo_calibration.handeye_runtime import HandEye

    now = time.monotonic()
    source = SimpleNamespace(condition=threading.Condition(), latest=None, error=None)
    h = HandEye(SimpleNamespace())
    h.controller = SimpleNamespace(source=source)
    h.active = h.auto_capture = True
    h.hold.progress = 0.5
    h.measurement = {"ready": True}
    processed = []

    class TwoIterations:
        count = 0

        def is_set(self):
            return self.count >= 2

        def wait(self, _duration):
            if self.count == 0:
                assert h.auto_capture
                assert h.measurement is None and h.hold.progress == 0
                source.latest = Frame(1, now, np.zeros((4, 8, 3), dtype=np.uint8))
            self.count += 1

    h.stop_event = TwoIterations()
    monkeypatch.setattr(h, "_process", lambda packet, detector: processed.append(packet.sequence))
    h._run()
    assert processed == [1]
    assert h.auto_capture and h.error is None
