"""View scoring selects new information and communicates metric moves correctly."""

import json

import numpy as np
from test_handeye_guidance import camera as camera_fixture
from test_handeye_guidance import centred_pose, sample

from stereo_calibration.handeye_guidance import LocalGuidance, PositionThenTilt, _validation_support
from stereo_calibration.handeye_information import ViewInformation, view_information

camera = camera_fixture


def poses():
    return [centred_pose(v) for v in [(0, 0, 0), (25, 0, 0), (0, 25, 0), (-25, 0, 0), (0, -25, 0)]]


def test_translations_wait_until_rotation_coverage_survives_held_out_orientations(camera):
    states = poses()[:3]
    assert not _validation_support(states)["sufficient"]
    saved = [sample(p, index=i) for i, p in enumerate(states)]
    first = LocalGuidance().guide(saved, states[-1], "training", model=camera)
    for skip in range(first["selection"]["candidate_count"]):
        goal = LocalGuidance().guide(saved, states[-1], "training", model=camera, skip=skip)
        assert not goal.get("translation_target_camera_m")
        assert "validation" in goal["selection_reason"]


def test_new_rotation_axis_adds_more_information_than_an_identical_view(camera):
    same_axis = [centred_pose((x, 0, 0)) for x in (0, 12, 24)]
    information = ViewInformation(same_axis, camera)
    assert information.gain(centred_pose((0, 25, 0))) > information.gain(same_axis[-1])
    matrix = view_information(same_axis[0], camera)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-7)
    assert np.linalg.eigvalsh(matrix).min() > -1e-7


def test_covered_local_candidates_choose_highest_information_gain(camera):
    states = poses()
    saved = [sample(p, index=i) for i, p in enumerate(states)]
    planner = LocalGuidance()
    selected = planner.guide(saved, states[-1], "training", model=camera)
    alternatives = [
        LocalGuidance().guide(saved, states[-1], "training", model=camera, skip=i)
        for i in range(selected["selection"]["candidate_count"])
    ]
    gains = [g["selection"]["expected_information_gain_nats"] for g in alternatives]
    assert selected["selection"]["expected_information_gain_nats"] == max(gains)
    assert any(g.get("translation_target_camera_m") for g in alternatives)
    assert all(g["visibility"]["feasible"] for g in alternatives)
    assert all(g["selection"]["robot_path_verified"] is False for g in alternatives)
    json.dumps(alternatives, allow_nan=False)


def test_translation_goal_is_fixed_and_needs_metric_arrival_not_just_image_alignment(camera):
    states = poses()
    saved = [sample(p, index=i) for i, p in enumerate(states)]
    planner = LocalGuidance()
    first = planner.guide(saved, states[-1], "training", model=camera)
    for skip in range(first["selection"]["candidate_count"]):
        g = planner.guide(saved, states[-1], "training", model=camera, skip=skip)
        if g.get("translation_target_camera_m"):
            break
    assert g.get("translation_target_camera_m")
    target = np.asarray(g["target_camera_pose"])
    np.testing.assert_allclose(np.linalg.norm(target[:3, 3] - states[-1][:3, 3]), 0.025)
    signature = dict(zip(("x", "y"), g["position_target"]))
    cue = PositionThenTilt().apply(g, signature, g["position_target"])
    assert not cue["at_target"]  # Image alignment alone cannot satisfy depth.
    assert cue["adjustment"]["kind"] in ("move", "distance")
    moved = states[-1].copy()
    moved[:3, 3] += (target[:3, 3] - moved[:3, 3]) * 0.2
    updated = planner.guide(saved, moved, "training", model=camera, skip=skip)
    assert updated["translation_target_camera_m"] == g["translation_target_camera_m"]
    reached = planner.guide(saved, target, "training", model=camera, skip=skip)
    assert reached["at_target"]
    assert (
        PositionThenTilt().apply(reached, signature, reached["position_target"])["adjustment"][
            "kind"
        ]
        == "ready"
    )


def test_information_goal_cannot_override_accuracy_review(camera):
    states = poses()
    saved = [sample(p, index=i) for i, p in enumerate(states)]
    guide = LocalGuidance().guide(
        saved,
        states[-1],
        "training",
        model=camera,
        readiness={
            "ready": False,
            "coverage": {"sufficient": True},
            "status": "accuracy",
            "reason": "Robot motion and observed board motion disagree.",
        },
    )
    assert guide["key"] == "review"
    assert guide["target_rotation"] is None
    assert not guide.get("translation_target_camera_m")
