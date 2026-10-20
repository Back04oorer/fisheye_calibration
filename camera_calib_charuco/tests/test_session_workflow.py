import copy

import cv2
import numpy as np
import pytest
from synthetic import observations

from stereo_calibration.calibration import fit
from stereo_calibration.camera import Frame
from stereo_calibration.config import BoardConfig, CameraConfig
from stereo_calibration.controller import Controller
from stereo_calibration.opencv import points
from stereo_calibration.session import Session, fit_snapshot
from stereo_calibration.validation import evaluate


def test_heldout_failure_remains_and_never_enters_fit(tmp_path):
    board, training, *_ = observations()
    session = Session.create(tmp_path / "session", board, CameraConfig())
    image = np.zeros((720, 2560, 3), dtype=np.uint8)
    for i, observation in enumerate(training):
        session.append(
            image, observation, role="training", frame_id=i, goals=[], signature=None, quality=[]
        )
    snapshot = session.training_snapshot()
    session.accept_fit(fit_snapshot(snapshot))
    _, fresh, *_ = observations(9, seed=19)
    fresh[-1].right_points[:, 1] += 6
    for i, observation in enumerate(fresh):
        session.append(
            image,
            observation,
            role="validation",
            frame_id=i + 20,
            goals=[],
            signature=None,
            quality=[],
        )
    report = session.validate()
    assert not report["passed"]
    assert len(report["validation_views"]) == 9
    assert not report["validation_views"][-1]["passed"]
    assert session.training_snapshot() == snapshot
    candidate = session.candidate["id"]
    session.close()
    resumed = Session(tmp_path / "session")
    assert resumed.candidate["id"] == candidate
    assert len(resumed.observations("validation")) == 9
    resumed.new_round()
    assert len(resumed.observations("validation")) == 0
    assert len(resumed.data["captures"]) == 29
    assert not resumed.data["candidates"][0]["report"]["passed"]
    gallery = resumed.saved_views()
    assert len(gallery) == 29
    assert sum(view["role"] == "validation" for view in gallery) == 9
    assert any(view["status"] == "failed" for view in gallery)
    assert sum(view["status"] == "used" for view in gallery) == len(training)
    resumed.close()


def test_good_frozen_geometry_passes_fresh_views_only_when_complete():
    board, training, *_ = observations()
    model, _ = fit(training, board)
    _, fresh, *_ = observations(9, seed=19)
    assert evaluate(fresh, model, board, goals_complete=True)["passed"]
    assert not evaluate(fresh, model, board, goals_complete=False)["passed"]


def test_epipolar_alignment_cannot_hide_bad_fresh_reprojection():
    board, training, *_ = observations()
    model, _ = fit(training, board)
    _, fresh, *_ = observations(1, seed=19)
    observation = fresh[0]
    # Move corresponding corners along their epipolar lines. Vertical alignment
    # stays perfect, but the non-rigid pattern cannot fit the known board geometry.
    for eye, K, D, R, P in (
        ("left", model.K1, model.D1, model.R1, model.P1),
        ("right", model.K2, model.D2, model.R2, model.P2),
    ):
        image = getattr(observation, f"{eye}_points")
        rectified = cv2.fisheye.undistortPoints(points(image, 2), K, D, R=R, P=P).reshape(-1, 2)
        rectified[:, 0] += 10 * np.sin(np.arange(len(rectified)))
        rays = np.column_stack([rectified, np.ones(len(rectified))]) @ np.linalg.inv(P[:, :3]).T
        rays = rays @ R
        changed = cv2.fisheye.distortPoints(points(rays[:, :2] / rays[:, 2:], 2), K, D)
        setattr(observation, f"{eye}_points", changed.reshape(-1, 2))
    report = evaluate(fresh, model, board, goals_complete=True)
    view = report["validation_views"][0]
    assert view["p95_px"] < 1e-6
    assert view["left_reprojection_rms_px"] > 1
    assert view["right_reprojection_rms_px"] > 1
    assert not report["passed"]
    assert any(check.startswith("Left reprojection") for check in view["failed_checks"])
    assert any(check.startswith("Right reprojection") for check in view["failed_checks"])


def test_shrinking_rectified_projection_cannot_hide_bad_stereo_geometry():
    board, training, *_ = observations()
    model, _ = fit(training, board)
    _, fresh, *_ = observations(1, seed=19)
    fresh[0].right_points[:, 1] += 6
    before = evaluate(fresh, model, board, goals_complete=True)
    model.P1[:2] *= 1e-6
    model.P2[:2] *= 1e-6
    after = evaluate(fresh, model, board, goals_complete=True)
    assert after["validation_views"][0]["p95_px"] < 0.001
    assert not before["passed"] and not after["passed"]
    for key in ("left_to_right_rms_px", "right_to_left_rms_px"):
        assert before["validation_views"][0][key] > 1
        assert before["validation_views"][0][key] == after["validation_views"][0][key]


def test_failed_save_does_not_claim_capture(tmp_path, monkeypatch):
    from stereo_calibration import session as persistence

    board, training, *_ = observations()
    session = Session.create(tmp_path / "session", board, CameraConfig())

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(persistence, "atomic_json", fail)
    with pytest.raises(OSError, match="disk full"):
        session.append(
            np.zeros((720, 2560, 3), dtype=np.uint8),
            training[0],
            role="training",
            frame_id=1,
            goals=[],
            signature=None,
            quality=[],
        )
    assert session.data["captures"] == []
    assert list((session.path / "frames").iterdir()) == []
    before = copy.deepcopy(session.data)
    with pytest.raises(OSError, match="disk full"):
        session.new_round()
    assert session.data == before
    session.close()


def test_automatic_capture_pause_and_duplicate_suppression(tmp_path, monkeypatch):
    board = BoardConfig()
    controller = Controller(board, CameraConfig(), output=tmp_path)
    pattern = board.create().generateImage((440, 320))
    eye = np.full((720, 1280), 255, dtype=np.uint8)
    eye[200:520, 420:860] = pattern
    image = cv2.cvtColor(np.hstack([eye, eye]), cv2.COLOR_GRAY2BGR)
    controller.process(Frame(0, 0, image))
    controller._command("start", {})
    for i in range(1, 7):
        controller.process(Frame(i, i * 0.5, image))
        controller.publish()
        if i == 1:
            assert controller.state()["adjustment"]["kind"] == "ready"
    assert len(controller.session.captures("training")) == 1
    capture = controller.session.captures("training")[0]
    assert controller.state()["adjustment"]["kind"] == "move"
    assert controller.state()["adjustment"]["target"] == (0.2, 0.2)
    assert len(controller.state()["saved_views"]) == 1
    np.testing.assert_array_equal(
        cv2.imread(str(controller.session.path / capture["frame"])), image
    )
    controller._command("pause", {})
    shifted = np.roll(image.reshape(720, 2, 1280, 3), 130, axis=2).reshape(720, 2560, 3)
    for i in range(7, 10):
        controller.process(Frame(i, i * 0.5, shifted))
    assert len(controller.session.captures("training")) == 1
    controller._command("resume", {})
    for i in range(10, 14):
        controller.process(Frame(i, i * 0.5, shifted))
    assert len(controller.session.captures("training")) == 2

    # Fast detection can suggest a new pose while the full pass recovers an old one.
    # Recheck novelty on the observation that will actually be saved.
    old = controller.session.observations("training")[0]
    detect = controller.detector.observe

    def changed_pose(frame, swap=False, *, full=True):
        observation, quality = detect(frame, swap, full=full)
        return (copy.deepcopy(old) if full else observation), quality

    monkeypatch.setattr(controller.detector, "observe", changed_pose)
    shifted_again = np.roll(image.reshape(720, 2, 1280, 3), 260, axis=2).reshape(720, 2560, 3)
    for i in range(14, 18):
        controller.process(Frame(i, i * 0.5, shifted_again))
    assert len(controller.session.captures("training")) == 2
    controller.close()


def test_session_has_single_writer_and_replay_is_readonly(tmp_path):
    session = Session.create(tmp_path / "session", BoardConfig(), CameraConfig())
    with pytest.raises(ValueError, match="already open"):
        Session(session.path)
    replay = Session(session.path, readonly=True)
    before = copy.deepcopy(replay.data)
    with pytest.raises(ValueError, match="read-only"):
        replay.save()
    assert replay.data == before
    replay.close()
    session.close()


def test_partial_and_unsolved_live_poses_recover_without_pausing_or_saving(tmp_path, monkeypatch):
    from stereo_calibration import guidance

    board, views, *_ = observations(9)
    session = Session.create(tmp_path / "live-recovery", board, CameraConfig())
    session.phase("collecting")
    controller = Controller(session=session)
    controller.model, _ = fit(views, board)
    controller._command("resume", {})
    frame = np.zeros((720, 2560, 3), np.uint8)
    complete = views[0]
    partial = copy.deepcopy(complete)
    for key in ("left_ids", "left_points", "right_ids", "right_points"):
        setattr(partial, key, getattr(partial, key)[:5])

    # These five points form one board row and reproduce the reported DLT failure
    # if they reach the solver. Test both preview and final detection paths.
    class Detection:
        preview = complete
        final = partial

        def observe(self, *_args, full=True, **_kwargs):
            return copy.deepcopy(self.final if full else self.preview), [
                {"sharpness": 100.0, "corners": 70, "square_px": 20.0}
            ] * 2

    detector = controller.detector = Detection()

    def process(index):
        controller.process(Frame(index, index * 0.6, frame))
        controller.publish()
        assert not controller.paused and controller.error is None
        assert controller.sequence == index
        assert controller.image(index, "raw")

    try:
        for i in range(3):
            process(i)
        assert controller.signature is None
        assert controller.hold_progress == 0
        assert session.captures("training") == []
        detector.preview = partial
        process(3)
        assert "seeing 5" in controller.message
        assert controller.signature is None
        detector.preview = detector.final = complete
        # Even enough corners can briefly yield an unsolved/behind-camera pose.
        # Both observed solver failure types are transient at the live boundary.
        with monkeypatch.context() as patch:
            for i, error_type in enumerate((cv2.error, ValueError), start=4):

                def fail(*_args, error_type=error_type):
                    raise error_type("Pose cannot be estimated")

                patch.setattr(guidance, "board_pose", fail)
                process(i)
                assert controller.signature is None
                assert controller.hold_progress == 0
                assert not controller.capture_eligible
                assert session.captures("training") == []
        process(6)
        process(7)
        assert session.captures("training") == []  # Hold starts again after detection recovers.
        process(8)
        assert len(session.captures("training")) == 1
        assert len(session.observations("training")[0].common()[0]) == 70
    finally:
        controller.close()


@pytest.mark.parametrize("bad_view", [False, True])
def test_guided_fresh_checks_finish_or_retry_automatically(tmp_path, bad_view):
    from stereo_calibration.guidance import POSITIONS
    from stereo_calibration.observations import Observation

    board, training, K, D, R, T = observations()
    session = Session.create(tmp_path / "guided", board, CameraConfig())
    frame = np.zeros((720, 2560, 3), np.uint8)
    for i, obs in enumerate(training):
        session.append(
            frame, obs, role="training", frame_id=i, goals=[], signature=None, quality=[]
        )
    session.accept_fit(fit_snapshot(session.training_snapshot()))
    session.phase("validating")
    controller = Controller(session=session)
    controller._command("resume", {})

    class SuppliedObservation:
        current = None

        def observe(self, *_args, **_kwargs):
            return copy.deepcopy(self.current), [
                {"sharpness": 100.0, "corners": 70, "square_px": 20.0}
            ] * 2

    detector = SuppliedObservation()
    controller.detector = detector
    obj = board.create().getChessboardCorners().astype(np.float64).reshape(1, -1, 3)
    centre = np.array([board.columns * board.square_mm / 2, board.rows * board.square_mm / 2, 0.0])
    for i, (_name, x, y) in enumerate(POSITIONS):
        ray = cv2.fisheye.undistortPoints(np.array([[[x * 1280, y * 720]]]), K, D).reshape(2)
        r = np.array([0.1 * (i % 3 - 1), 0.12 * (i % 2 * 2 - 1), 0.02])
        t = np.r_[ray * 500, 500.0] - cv2.Rodrigues(r)[0] @ centre
        left = cv2.fisheye.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
        rr = cv2.Rodrigues(R @ cv2.Rodrigues(r)[0])[0]
        right = cv2.fisheye.projectPoints(obj, rr, R @ t + T, K, D)[0].reshape(-1, 2)
        if bad_view:
            right[:, 1] += 6
        ids = np.arange(70)
        detector.current = Observation(f"fresh-{i}", (1280, 720), ids, left, ids.copy(), right)
        for j in range(3):
            controller.process(Frame(i * 3 + j, (i * 3 + j) * 0.6, frame))
        if bad_view:
            assert controller.phase == "collecting"
            assert not controller.paused
            assert session.candidate is None
            assert not session.data["candidates"][-1]["report"]["passed"]
            assert session.data["training_request"]["goals"] == ["centre"]
            assert controller.next_goal.key == "centre"
            assert len(session.training_snapshot()["observations"]) == len(training)
            # Retrying is a durable request, not a transient UI message, and the
            # failed held-out image is preserved without becoming training data.
            path = session.path
            controller.close()
            resumed = Controller(session=Session(path))
            assert resumed.next_goal.key == "centre"
            assert resumed.model is not None
            assert len(resumed.session.data["captures"]) == len(training) + 1
            assert len(resumed.session.training_snapshot()["observations"]) == len(training)
            resumed.close()
            return
    assert controller.phase == "done"
    assert len(session.captures("validation")) == 9
    assert session.candidate["report"]["passed"]
    controller.close()
    replay = Controller(session=Session(session.path, readonly=True), replay=True)
    assert len(replay.goals) == len(replay.done_goals) == 9
    replay.close()
