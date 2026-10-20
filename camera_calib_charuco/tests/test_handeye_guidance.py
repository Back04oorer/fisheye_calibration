"""Advisory goals are local, stereo-visible, stable, and not capture gates."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from stereo_calibration.handeye import inverse, solve, transform
from stereo_calibration.handeye_guidance import (
    GUIDE_POLICY,
    GuidedHold,
    LocalGuidance,
    PositionThenTilt,
    _about_centre,
    _face_signs,
    _sequential_rotations,
    angle,
    feasible_path,
    guidance,
    pose_preview,
    requires_review,
    rotation,
    visibility,
)


@pytest.fixture
def camera():
    K = np.array([[500.0, 0, 640], [0, 500, 360], [0, 0, 1]])
    return SimpleNamespace(
        image_size=(1280, 720),
        K1=K,
        K2=K.copy(),
        D1=np.zeros(4),
        D2=np.zeros(4),
        R=np.eye(3),
        T=np.array([-60.0, 0, 0]),
    )


def sample(C, role="training", index=0):
    X = transform(rotation([15, -10, 8]), [0.4, -0.2, 0.6])
    Z = transform(rotation([20, 5, -12]), [0.03, 0.04, 0.1])
    return {
        "id": str(index),
        "role": role,
        "T_camera_board": C.tolist(),
        "T_base_tool": (X @ C @ inverse(Z)).tolist(),
    }


def centred_pose(vector=(0, 0, 0)):
    return _about_centre(transform(np.eye(3), [-0.05, -0.0625, 0.5]), rotation(vector))


def test_guided_sequence_is_local_visible_and_eventually_observes_two_axes(camera):
    current = centred_pose()
    planner, saved = LocalGuidance(), []
    for index in range(9):
        g = planner.guide(saved, current, "training", model=camera)
        assert g["target_rotation"] is not None
        reference = np.asarray(saved[0]["T_camera_board"])[:3, :3] if saved else current[:3, :3]
        target = reference @ np.asarray(g["target_rotation"])
        assert angle(current[:3, :3], target) <= 12.000001
        assert visibility(_about_centre(current, target), camera)["feasible"]
        next_pose = np.asarray(g.get("target_camera_pose", _about_centre(current, target)))
        assert np.linalg.norm(next_pose[:3, 3] - _about_centre(current, target)[:3, 3]) <= 0.0250001
        current = next_pose
        assert planner.guide(saved, current, "training", model=camera)["at_target"]
        saved.append(sample(current, index=index))
    g = planner.guide(saved, current, "training", model=camera)
    assert g["diversity"]["sufficient"] and g["key"] != "done"
    assert "minimum" not in g
    np.testing.assert_allclose(
        solve(saved)["T_base_camera_left_raw"],
        transform(rotation([15, -10, 8]), [0.4, -0.2, 0.6]),
        atol=1e-8,
    )
    assert planner.guide(saved, current, "complete", model=camera)["key"] == "done"


def test_target_does_not_chase_operator_and_large_deviation_hides_it(camera):
    current = centred_pose()
    saved, planner = [sample(current)], LocalGuidance()
    first = planner.guide(saved, current, "training", model=camera)
    target = np.asarray(first["target_rotation"])
    moved = _about_centre(current, rotation([0, 5, 0]))
    next_state = planner.guide(saved, moved, "training", model=camera)
    assert next_state["key"] == first["key"]
    np.testing.assert_allclose(next_state["target_rotation"], target)
    far = _about_centre(current, rotation([0, -30, 0]))
    blocked = planner.guide(saved, far, "training", model=camera)
    assert blocked["key"] == "unavailable" and blocked["target_rotation"] is None
    assert "too far" in blocked["instruction"]
    # Explicit new suggestion selects from the new measured pose.
    updated = planner.guide(saved, far, "training", skip=1, model=camera)
    assert updated["key"] != first["key"]
    assert updated["angle_error_deg"] <= 12.000001


def test_check_phase_has_own_coverage_and_avoids_training_rotations(camera):
    current = centred_pose()
    saved = [sample(current)]
    planner = LocalGuidance()
    g = planner.guide(saved, current, "checking", model=camera)
    assert g["count"] == 0 and not g["diversity"]["sufficient"]
    assert g["angle_error_deg"] >= GUIDE_POLICY["target_separation_deg"]
    assert g["axis_span_deg"] == [0, 0, 0]
    saved[0]["excluded"] = True
    assert planner.guide(saved, current, "training", model=camera)["key"] == "reference"


def test_both_cameras_depth_backface_clip_and_edge_on_are_rejected(camera):
    assert visibility(centred_pose(), camera)["feasible"]
    assert not visibility(transform(np.eye(3), [0, 0, -0.5]), camera)["feasible"]
    assert "turn away" in visibility(centred_pose([180, 0, 0]), camera)["reason"]
    assert "edge-on" in visibility(centred_pose([0, 75, 0]), camera)["reason"]
    clipped = centred_pose()
    clipped[1, 3] += 0.6
    assert "leave" in visibility(clipped, camera)["reason"]
    # Left eye stays valid; right optical geometry is independently tested.
    camera.T = np.array([-1400.0, 0, 0])
    right_bad = visibility(centred_pose(), camera)
    assert not right_bad["feasible"] and "right" in right_bad["reason"]


def test_never_worsens_already_oblique_view_and_checks_ui_path(camera):
    current = centred_pose([0, 52, 0])
    signs = _face_signs(centred_pose(), camera)
    assert visibility(current, camera, signs)["feasible"]
    worse = feasible_path(current, current[:3, :3] @ rotation([0, 1, 0]), camera, signs)
    assert not worse["feasible"] and "oblique" in worse["reason"]
    better = feasible_path(current, current[:3, :3] @ rotation([0, -12, 0]), camera, signs)
    assert better["feasible"] and set(better["checked_ui_orders"]) == {"XYZ", "YXZ"}


@pytest.mark.parametrize("skip", [0, 1])
def test_oblique_suggestions_improve_view_before_coverage_and_survive_point_two_degree_jitter(
    camera, skip
):
    current = centred_pose([-40, 35, 0])
    saved = [
        sample(centred_pose(vector), index=i)
        for i, vector in enumerate([(0, 0, 0), (-20, 0, 0), (0, 20, 0), (-40, 35, 0)])
    ]
    planner = LocalGuidance()
    initial = planner.guide(saved, current, "training", skip, model=camera)
    target = np.asarray(initial["target_rotation"])
    assert initial["angle_error_deg"] == pytest.approx(12)
    assert all(
        before["obliquity_deg"] - after["obliquity_deg"] > 5
        for before, after in zip(
            initial["visibility"]["eyes"], initial["visibility"]["target_eyes"]
        )
    )
    for axis in range(3):
        for sign in (-1, 1):
            jitter = np.zeros(3)
            jitter[axis] = sign * 0.2
            observed = _about_centre(current, current[:3, :3] @ rotation(jitter))
            state = planner.guide(saved, observed, "training", skip, model=camera)
            assert state["key"] == initial["key"]
            np.testing.assert_allclose(state["target_rotation"], target)
            assert state["visibility"]["feasible"]
            assert set(state["visibility"]["checked_ui_orders"]) == {"XYZ", "YXZ"}


def test_tiny_secondary_correction_may_precede_improving_tilt_within_all_hard_bounds(camera):
    start = centred_pose([-40, 35, 0])
    target = start[:3, :3] @ rotation([12, 0, 0])
    observed = _about_centre(start, start[:3, :3] @ rotation([0, -0.2, 0]))
    signs = _face_signs(centred_pose(), camera)
    initial = visibility(observed, camera, signs)
    steps = [
        visibility(_about_centre(observed, R), camera, signs)
        for R in _sequential_rotations(observed[:3, :3], target)
    ]
    assert all(step["feasible"] for step in steps)
    # This reproduces the old false rejection: a 0.2° secondary correction
    # briefly changes obliquity by 0.147°, then the main tilt improves it >5°.
    assert any(
        after["obliquity_deg"] - before["obliquity_deg"] > 0.1
        for step in steps
        for before, after in zip(initial["eyes"], step["eyes"])
    )
    result = feasible_path(observed, target, camera, signs)
    assert result["feasible"]
    assert all(
        after["obliquity_deg"] < before["obliquity_deg"] - 5
        for before, after in zip(result["eyes"], result["target_eyes"])
    )
    # The endpoint may still never deliberately worsen an already-oblique view.
    worse = feasible_path(observed, observed[:3, :3] @ rotation([0, 1, 0]), camera, signs)
    assert not worse["feasible"] and "oblique" in worse["reason"]


def test_curved_fisheye_outline_is_checked_between_four_outer_corners(camera):
    pose = transform(np.eye(3), [-0.05, -0.35, 0.5])
    outline = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.125, 0.0], [0.0, 0.125, 0.0]])
    corners = cv2.fisheye.projectPoints(
        outline.reshape(-1, 1, 3), np.zeros(3), pose[:3, 3], camera.K1, camera.D1
    )[0].reshape(-1, 2)
    shift = GUIDE_POLICY["image_margin_px"] + 0.05 - corners[:, 1].min()
    camera.K1[1, 2] += shift
    assert np.all(corners[:, 1] + shift > GUIDE_POLICY["image_margin_px"])
    result = visibility(pose, camera)
    assert not result["feasible"] and "outline" in result["reason"]


def test_obliquity_uses_actual_viewing_ray_and_preview_keeps_intrinsic_delta(camera):
    # Board normal is >60° from optical Z, yet faces the off-axis viewing ray.
    current = centred_pose([0, 65, 0])
    current[0, 3] += 0.8
    assert visibility(current, camera)["feasible"]
    saved = [sample(current)]
    g = LocalGuidance().guide(saved, current, "training", model=camera)
    assert g["target_rotation"] is not None
    preview = pose_preview(saved, g, {"rotation_camera": current[:3, :3].tolist()}, current)
    assert preview["source"] == "left_camera_board_view"
    assert np.degrees(np.arccos(preview["current_rotation"][2][2])) < 20
    current_view, target_view = map(
        np.asarray, (preview["current_rotation"], preview["target_rotation"])
    )
    np.testing.assert_allclose(
        current_view.T @ target_view, np.asarray(g["target_rotation"]), atol=1e-12
    )


def test_accuracy_review_never_draws_position_or_pose_target(camera):
    current = centred_pose()
    saved = [sample(current)]
    readiness = {
        "ready": False,
        "coverage": {"sufficient": True},
        "status": "accuracy",
        "reason": "Prediction exceeds its error budget.",
    }
    g = LocalGuidance().guide(saved, current, "training", model=camera, readiness=readiness)
    assert g["key"] == "review" and g["target_rotation"] is None
    assert "position_target" not in g
    flowed = PositionThenTilt().apply(g, {"x": 0.95, "y": 0.1})
    assert flowed["adjustment"]["kind"] == "unavailable"
    assert flowed["instruction"] == g["instruction"]
    assert pose_preview(saved, g, {"rotation_camera": current[:3, :3].tolist()}, current) is None
    assert not requires_review(None)
    assert not requires_review({"ready": False, "prior_validation": None, "status": "coverage"})
    assert requires_review({"ready": False, "prior_validation": {"passed": False}})
    stability = {
        **readiness,
        "status": "stability",
        "leave_one_out": [{"coverage_sufficient": False}],
    }
    assert not requires_review(stability)
    stability["leave_one_out"][0]["coverage_sufficient"] = True
    assert requires_review(stability)


def test_missing_geometry_and_exhausted_targets_do_not_guess_directions(camera):
    current, planner = centred_pose(), LocalGuidance()
    saved = [sample(current)]
    assert guidance(saved, current, "training")["key"] == "unavailable"
    assert planner.guide(saved, None, "training", model=camera)["target_rotation"] is None
    assert planner.guide(saved, current, "training", model=camera)["target_rotation"] is not None
    missing = planner.guide(saved, None, "training", model=camera)
    assert missing["key"] == "unavailable" and not missing["at_target"]


def test_position_anchor_is_selected_current_area_not_first_capture(camera):
    current = centred_pose()
    saved = [sample(current)]
    planner = LocalGuidance()
    g = planner.guide(saved, current, "training", model=camera, signature={"x": 0.8, "y": 0.3})
    assert g["position_target"] == [0.8, 0.3]
    next_frame = planner.guide(
        saved, current, "training", model=camera, signature={"x": 0.81, "y": 0.31}
    )
    assert next_frame["position_target"] == [0.8, 0.3]
    moved = PositionThenTilt().apply(g, {"x": 0.8, "y": 0.3}, g["position_target"])
    assert moved["adjustment"]["kind"] in ("tilt", "roll")


def test_hold_resets_on_invalid_frame_gap_changed_target_motion_and_repeated_frame():
    h = GuidedHold()
    pose = np.eye(4)
    assert h.update(10, "a", True, pose) == 0
    assert h.update(10.5, "a", True, pose) == 0.5
    assert h.update(10.5, "a", True, pose) == 0
    h.update(11, "a", True, pose)
    assert h.update(13, "a", True, pose) == 0
    assert h.update(13.5, "b", True, pose) == 0
    moving = pose.copy()
    moving[0, 3] = 0.01
    assert h.update(14, "b", True, moving) == 0
    assert h.update(14.5, "b", False, None) == 0
    h.update(15, "b", True, pose)
    assert h.update(15.5, "b", True, pose) == 0.5
    assert h.update(16.1, "b", True, pose) == 1
    h.reset()
    assert h.progress == 0
