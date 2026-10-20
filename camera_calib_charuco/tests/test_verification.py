import copy

import numpy as np
import pytest
from synthetic import observations, regional_observations

from stereo_calibration.calibration import Calibration
from stereo_calibration.config import CameraConfig
from stereo_calibration.controller import Controller
from stereo_calibration.session import Session, fit_snapshot
from stereo_calibration.verification import evaluate_saved_checks


@pytest.mark.parametrize("damaged_check", [False, True])
def test_refit_preserves_rechecked_coverage_and_targets_only_outstanding_areas(
    tmp_path, damaged_check
):
    board, training, K, D, R, T = observations(9)
    regional = regional_observations(board, K, D, R, T)
    frame = np.zeros((720, 2560, 3), np.uint8)
    session = Session.create(tmp_path / "retry", board, CameraConfig())

    def save(observation, role):
        return session.append(
            frame,
            copy.deepcopy(observation),
            role=role,
            frame_id=0,
            goals=[],
            signature=None,
            quality=[],
        )

    for view in training:
        save(view, "training")
    session.accept_fit(fit_snapshot(session.training_snapshot()))
    # Eight regions were checked; left has not been visited yet.
    for index, view in enumerate(regional[:8]):
        view = copy.deepcopy(view)
        if damaged_check and index == 7:
            view.left_points[:, 0] += 4 * np.sin(np.arange(len(view.left_ids)))
        save(view, "validation")
    old_report = copy.deepcopy(session.validate())
    originals = copy.deepcopy(session.data["captures"])
    first_candidate = session.candidate["id"]
    session.new_round(["lower-left"], reason="Improve lower left")
    save(regional[7], "training")
    snapshot = session.training_snapshot()
    session.accept_fit(fit_snapshot(snapshot))
    second_candidate = session.candidate["id"]
    assert second_candidate != first_candidate
    assert "training_request" not in session.data
    assert session.data["candidates"][0]["report"] == old_report
    assert session.data["captures"][: len(originals)] == originals
    assert all(c["id"] not in snapshot["capture_ids"] for c in session.captures("validation"))

    report = session.validate()
    regions = {r["key"]: r["status"] for r in report["regions"]}
    assert regions["check-upper-left"] == "passed"
    assert regions["check-left"] == "missing"
    assert regions["check-lower-left"] == ("failed" if damaged_check else "confirm")
    assert len(report["reused_passed_regions"]) == 7
    assert len(report["validation_views"]) == 8
    assert not report["passed"]

    if damaged_check:
        # A new favourable photo cannot hide the original failing measurement.
        save(regional[7], "validation")
        report = session.validate()
        assert not report["passed"]
        assert len(report["validation_views"]) == 9
        assert not report["validation_views"][7]["passed"]
        assert report["validation_views"][8]["passed"]
        controller = Controller(session=session)
        assert controller.phase == "collecting"
        assert controller.next_goal.key == "lower-left"
        assert session.data["training_request"]["goals"] == ["lower-left"]
        assert session.training_snapshot() == snapshot
        path = session.path
        controller.close()
        resumed = Controller(session=Session(path))
        assert resumed.next_goal.key == "lower-left"
        resumed.close()
        return

    controller = Controller(session=session)
    assert controller.phase == "validating"
    assert controller.next_goal.key == "check-lower-left"
    assert len(controller.done_goals) == 7
    save(regional[7], "validation")
    controller.after_capture()
    assert controller.next_goal.key == "check-left"
    save(regional[8], "validation")
    controller.after_capture()
    assert controller.phase == "done"
    assert session.candidate["report"]["passed"]
    assert session.data["candidates"][0]["report"] == old_report

    # Reuse must recompute geometry, never copy a previous pass flag.
    changed = Calibration.from_dict(session.candidate["model"])
    changed.T[1] += 8
    regression = evaluate_saved_checks(session.captures("validation"), changed, board, "changed")
    assert not regression["passed"]
    assert any(not view["passed"] for view in regression["validation_views"])
    controller.close()
