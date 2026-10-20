import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from synthetic import observations

from stereo_calib_charuco.stereo_calib import load_intrinsics
from stereo_calib_charuco.view_stereo_undistorted import (
    BoardDetection,
    StereoParameters,
    calculate_validation_metrics,
)
from stereo_calibration.calibration import export, fit
from stereo_calibration.config import PRESETS, BoardConfig
from stereo_calibration.detection import detect_charuco_robust
from stereo_calibration.guidance import describe
from stereo_calibration.observations import Observation
from stereo_calibration.opencv import fit_intrinsics, points
from stereo_calibration.rectification import horizontal_rectification


def test_wide_field_initialization_recovers_known_geometry_without_dropping_views():
    fixture = json.loads((Path(__file__).parent / "data/wide_fisheye_poses.json").read_text())
    K, D = np.array(fixture["K"]), np.array(fixture["D"])
    corners = BoardConfig().create().getChessboardCorners().astype(float)
    objects, images = [], []
    for pose in fixture["poses"]:
        obj = points(corners[pose["ids"]], 3)
        objects.append(obj)
        images.append(
            cv2.fisheye.projectPoints(obj, np.array(pose["rvec"]), np.array(pose["tvec"]), K, D)[0]
        )
    rms, matrix, distortion, rotations, translations = fit_intrinsics(
        objects, images, tuple(fixture["image_size"])
    )
    assert rms < 1e-6
    np.testing.assert_allclose(matrix, K, atol=1e-5)
    np.testing.assert_allclose(distortion.ravel(), D, atol=1e-5)
    assert len(rotations) == len(translations) == len(fixture["poses"])
    # The same wide geometry made OpenCV's boundary-based viewport collapse.
    R = cv2.Rodrigues(np.array([0.003, -0.008, 0.001]))[0]
    T = np.array([-60.0, 0.5, 0.3])
    R1, R2, P1, P2, Q = horizontal_rectification(
        matrix, distortion, matrix, distortion, tuple(fixture["image_size"]), R, T
    )
    assert P1[0, 0] > 500
    assert 0 < P1[0, 2] < fixture["image_size"][0]
    assert 0 < P1[1, 2] < fixture["image_size"][1]
    # Projection, stereo baseline and disparity-to-depth must remain consistent.
    xyz = np.array([[-80.0, -30, 450], [140, 90, 600]])
    left_rectified = xyz @ R1.T
    right_rectified = (xyz @ R.T + T) @ R2.T
    a, b = left_rectified @ P1[:, :3].T, right_rectified @ P2[:, :3].T
    a, b = a[:, :2] / a[:, 2:], b[:, :2] / b[:, 2:]
    np.testing.assert_allclose(a[:, 1], b[:, 1], atol=1e-8)
    homogeneous = np.column_stack([a, a[:, 0] - b[:, 0], np.ones(len(a))]) @ Q.T
    np.testing.assert_allclose(homogeneous[:, :3] / homogeneous[:, 3:], left_rectified, atol=1e-8)
    # A physical board centre at the edge must not become a centre-screen target
    # because a projected outline crosses the pinhole projection's horizon.
    for index, pose in enumerate(fixture["poses"]):
        ids = np.array(pose["ids"])
        pixels = images[index].reshape(-1, 2)
        observation = Observation(
            str(index), tuple(fixture["image_size"]), ids, pixels, ids, pixels
        )
        signature = describe(observation, BoardConfig(), SimpleNamespace(K1=matrix, D1=distortion))
        expected = (
            cv2.fisheye.projectPoints(
                points([[110, 80, 0]], 3), np.array(pose["rvec"]), np.array(pose["tvec"]), K, D
            )[0].reshape(2)
            / fixture["image_size"]
        )
        np.testing.assert_allclose([signature["x"], signature["y"]], expected, atol=1e-6)


def test_known_fisheye_geometry_and_exports(tmp_path):
    # Fewer than twelve poses can determine this known geometry. Count alone
    # must neither prevent fitting nor establish validation success.
    board, views, K, D, R, T = observations(9)
    model, diagnostics = fit(views, board)
    np.testing.assert_allclose(model.K1, K, atol=1e-5)
    np.testing.assert_allclose(model.D1.reshape(-1), D, atol=1e-5)
    np.testing.assert_allclose(model.R, R, atol=1e-5)
    np.testing.assert_allclose(model.T.reshape(-1), T, atol=1e-4)
    assert max(model.rms) < 1e-6
    assert diagnostics["stereo_points_per_pair"] == 70
    export(model, tmp_path)
    loaded, _ = load_intrinsics(tmp_path / "left_fisheye.npz", (1280, 720))
    np.testing.assert_allclose(loaded, K, atol=1e-5)
    with pytest.raises(ValueError, match="image size"):
        load_intrinsics(tmp_path / "left_fisheye.npz", (640, 360))


@pytest.mark.parametrize("preset", PRESETS.values())
def test_preset_geometry_and_detection(preset):
    board, corners, markers = preset.geometry()
    for marker_id, obj in zip(board.getIds().reshape(-1), board.getObjPoints()):
        np.testing.assert_array_equal(markers[int(marker_id)], obj)
    image = board.generateImage((preset.columns * 100 + 40, preset.rows * 100 + 40), marginSize=20)
    detection, *_ = detect_charuco_robust(image, board)
    assert len(detection.ids) == len(corners)
    assert np.isclose(np.linalg.norm(corners[1] - corners[0]), preset.square_mm)


def test_rectification_uses_corresponding_corners_not_distorted_centres():
    K = np.array([[400.0, 0, 640.0], [0, 400.0, 360.0], [0, 0, 1.0]])
    D = np.array([-0.04, 0.005, 0, 0.0])
    obj = np.array([[0, 0, 0], [67, 0, 0], [67, 67, 0], [0, 67, 0]], dtype=float).reshape(1, 4, 3)
    r = np.array([0.4, 0.5, 0.1])
    t = np.array([200.0, 100.0, 240.0])
    left = cv2.fisheye.projectPoints(obj, r, t, K, D)[0].reshape(4, 2)
    right = cv2.fisheye.projectPoints(obj, r, t + [-60.0, 0, 0], K, D)[0].reshape(4, 2)
    projection = np.column_stack([K, np.zeros(3)])
    params = StereoParameters(
        (1280, 720), K, D, K, D, np.eye(3), np.eye(3), projection, projection, "fisheye"
    )
    result = calculate_validation_metrics(
        BoardDetection({0: left}), BoardDetection({0: right}), params, 1
    )
    assert result.matched_point_count == 4
    assert result.maximum < 0.001
