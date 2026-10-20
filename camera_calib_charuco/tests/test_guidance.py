import copy

import numpy as np
import pytest

from stereo_calibration.guidance import (
    TRAINING_GOALS,
    VALIDATION_GOALS,
    Hold,
    capture_ready,
    live_adjustment,
)
from stereo_calibration.observations import Observation


def test_stereo_short_hold_requires_fresh_stable_frames_and_keeps_sharpest_pair():
    pixels = np.array([[x * 20, y * 20] for y in range(3) for x in range(4)], dtype=float)
    obs = Observation("test", (1280, 720), np.arange(12), pixels, np.arange(12), pixels.copy())
    frame = np.zeros((4, 8, 3), np.uint8)
    hold = Hold()

    def update(t, sharpness=1, observation=obs):
        return hold.update(observation, t, frame, [{"sharpness": sharpness}] * 2)

    assert update(10)[1] is None
    assert update(10.35, 20)[1] is None
    progress, best = update(10.71)
    assert progress == 1 and best[0] == 20
    hold.reset()
    update(20)
    update(20.3)
    assert update(21.5) == (0, None)  # Camera interruption restarts at this new frame.
    assert update(21.8)[1] is None
    assert update(21.8) == (0, None)  # Polling one frame twice cannot count as stability.
    assert update(22)[1] is None
    assert update(21.9) == (0, None)  # Out-of-order camera timestamp.
    update(23)
    moved = copy.deepcopy(obs)
    moved.right_points += 10
    assert update(23.35, observation=moved) == (0, None)
    progress, best = update(23.7, observation=moved)
    assert progress == pytest.approx(0.5) and best is None


def test_arriving_switches_to_hold_or_useful_adjustment_without_new_capture_gates():
    pose = {
        "x": 0.5,
        "y": 0.5,
        "scale": 0.3,
        "skew_x": 0.0,
        "skew_y": 0.0,
        "roll": 0.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
    }
    centre, upper_left = TRAINING_GOALS[:2]
    assert capture_ready(pose, [], None, set(), 0.3, validation=False)
    assert live_adjustment(centre, pose, 0.3, set(), ready=True)["kind"] == "ready"
    # An exact repeat must never display Hold when the capture gate rejects it.
    assert not capture_ready(pose, [pose], None, set(), 0.3, validation=False)
    assert live_adjustment(centre, pose, 0.3, set()) == {"kind": "tilt", "edge": "left"}
    done = {"tilt-left", "tilt-right", "tilt-top", "tilt-bottom"}
    assert live_adjustment(centre, pose, 0.3, done)["kind"] == "distance"
    # A distance cue must not reverse as the board crosses the near-goal size.
    old = dict(pose, scale=0.39)
    for scale in (0.395, 0.40, 0.41):
        cue = live_adjustment(centre, dict(pose, scale=scale), 0.3, done, previous=[old])
        assert cue == {"kind": "distance", "direction": "closer"}
    move = live_adjustment(upper_left, pose, 0.3, {"centre"})
    assert move == {"kind": "move", "target": (0.2, 0.2), "current": [0.5, 0.5]}
    # Independent checks need a new region; novelty alone must not show Hold.
    moved = dict(pose, x=0.6)
    assert not capture_ready(moved, [pose], None, {"check-centre"}, 0.3, validation=True)
    assert (
        live_adjustment(VALIDATION_GOALS[1], moved, 0.3, {"check-centre"}, validation=True)["kind"]
        == "move"
    )
    fresh = dict(pose, x=0.2, y=0.2)
    assert capture_ready(fresh, [pose], None, {"check-centre"}, 0.3, validation=True)
    assert (
        live_adjustment(
            VALIDATION_GOALS[1], fresh, 0.3, {"check-centre"}, ready=True, validation=True
        )["kind"]
        == "ready"
    )
