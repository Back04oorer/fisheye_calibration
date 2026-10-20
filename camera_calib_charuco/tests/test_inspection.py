import copy
import json

import numpy as np
from synthetic import observations

from stereo_calibration.calibration import Calibration
from stereo_calibration.inspection import inspect_observation
from stereo_calibration.rectification import horizontal_rectification


def fixture():
    board, views, K, D, R, T = observations(1)
    size = views[0].image_size
    rectified = horizontal_rectification(K, D, K, D, size, R, T)
    return board, views[0], Calibration(size, K, D, K, D, R, T, *rectified, (0, 0, 0))


def test_detection_counts_do_not_hide_bad_stereo_alignment_or_depend_on_viewport():
    board, view, model = fixture()
    before = copy.deepcopy(view.to_dict())
    good = inspect_observation(view, board, model)
    assert good["total"] == 70
    assert len(good["points"]["left"]) == len(good["points"]["right"]) == 70
    assert len(good["shared_ids"]) == 70
    assert good["alignment"]["p95_px"] < 1e-8
    assert view.to_dict() == before

    # All corners still detect, but a wrong stereo transform must be visible.
    changed = copy.deepcopy(model)
    changed.T[1] += 10
    bad = inspect_observation(view, board, changed)
    assert bad["shared_ids"] == good["shared_ids"]
    assert bad["alignment"]["p95_px"] > 1

    # Output zoom changes rectified diagnostics, not native-sensor alignment.
    noisy = copy.deepcopy(view)
    noisy.right_points[:, 1] += 1
    original = inspect_observation(noisy, board, model)
    zoomed = copy.deepcopy(model)
    zoomed.P1[:2] *= .4
    zoomed.P2[:2] *= .4
    scaled = inspect_observation(noisy, board, zoomed)
    assert original["alignment"] == scaled["alignment"]
    np.testing.assert_allclose(
        scaled["rectified"]["all"]["p95_px"],
        original["rectified"]["all"]["p95_px"] * .4,
    )


def test_sparse_and_missing_detections_keep_counts_and_never_reuse_a_score():
    board, view, model = fixture()
    # Sparse observations must not call the board-pose solver.
    view.right_ids = view.right_ids[:4]
    view.right_points = view.right_points[:4]
    sparse = inspect_observation(view, board, model)
    assert len(sparse["points"]["left"]) == 70
    assert len(sparse["points"]["right"]) == len(sparse["shared_ids"]) == 4
    assert sparse["error"] is None
    assert sparse["alignment"]["p95_px"] < 1e-8
    assert inspect_observation(view, board)["alignment"] is None
    view.right_ids = view.right_ids[:0]
    view.right_points = view.right_points[:0]
    missing = inspect_observation(view, board, model)
    assert missing["shared_ids"] == []
    assert missing["alignment"] is missing["rectified"] is None
    json.dumps(missing, allow_nan=False)


def test_mismatched_frame_size_reports_unavailable_measurements_but_retains_counts():
    board, view, model = fixture()
    model.image_size = (640, 360)
    report = inspect_observation(view, board, model)
    assert len(report["shared_ids"]) == 70
    assert report["alignment"] is None
    assert "size" in report["error"]
