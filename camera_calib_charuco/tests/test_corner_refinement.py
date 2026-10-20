"""Image-based regressions for locally oriented ChArUco refinement."""

import json
from pathlib import Path

import cv2
import numpy as np

from stereo_calibration.config import BoardConfig
from stereo_calibration.corner_refinement import refine_in_board_axes
from stereo_calibration.detection import detect_charuco_robust, setup_charuco_detector


def test_opencv_recommended_marker_policy_preserves_subpixel_charuco_accuracy():
    board = BoardConfig(columns=4, rows=5, square_mm=25, marker_mm=20).create()
    detector = setup_charuco_detector(board)
    assert detector.getDetectorParameters().cornerRefinementMethod == cv2.aruco.CORNER_REFINE_NONE
    image = board.generateImage((400, 500))
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [399, 0], [399, 499], [0, 499]]),
        np.float32([[80, 60], [530, 100], [500, 520], [120, 550]]),
    )
    warped = cv2.warpPerspective(image, H, (640, 600), borderValue=180)
    warped = cv2.GaussianBlur(warped, (3, 3), 0.5)
    result, *_ = detect_charuco_robust(warped, board, detector)
    assert result is not None and len(result.ids) == 12
    # The pixel boundary lies between the last black and first white pixel.
    original = board.getChessboardCorners()[result.ids, :2] * 4 - 0.5
    expected = cv2.perspectiveTransform(original.reshape(-1, 1, 2), H)
    error = np.linalg.norm(result.corners - expected, axis=2)
    assert np.sqrt(np.mean(error**2)) < 0.2
    assert error.max() < 0.35


def test_skewed_recorded_corner_refines_on_image_without_losing_points():
    fixtures = Path(__file__).parent / "data"
    gray = cv2.imread(str(fixtures / "charuco_skewed_corner.png"), cv2.IMREAD_GRAYSCALE)
    record = json.loads((fixtures / "charuco_skewed_corner.json").read_text())
    points = np.asarray(record["points"], dtype=np.float32)
    result = refine_in_board_axes(
        gray, points, record["ids"], BoardConfig().create(), original_points=points
    )

    # Independently rerunning native refinement with a larger window and nearby
    # seeds places the printed junction around (79.5, 103), not the stalled seed
    # (78.65, 99.64). Allow for the different gradient support after warping.
    target = record["ids"].index(record["target_id"])
    np.testing.assert_allclose(result[target, 0], [79.5, 103.0], atol=0.75)
    assert result.shape == (len(points), 1, 2)
    assert np.all(np.isfinite(result))


def test_neighbours_orient_search_without_pulling_corner_to_ideal_grid():
    board = BoardConfig(columns=4, rows=4).create()
    points = np.array([[x, y] for y in range(3) for x in range(3)], dtype=np.float32)
    points = points * 40 + 40
    # Neighbours predict (80, 80), but the actual junction is displaced from it.
    points[4] = [84.5, 79.5]
    y, x = np.mgrid[:160, :160]
    gray = np.where((x >= 85) ^ (y >= 80), 255, 0).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

    result = refine_in_board_axes(gray, points, np.arange(9), board, original_points=points)

    np.testing.assert_allclose(result[4, 0], points[4], atol=0.05)


def test_collinear_partial_detection_keeps_native_estimates():
    points = np.array([[20 + 20 * x, 40] for x in range(6)], dtype=np.float32)
    gray = np.full((100, 160), 127, dtype=np.uint8)

    result = refine_in_board_axes(
        gray, points, np.arange(6), BoardConfig().create(), original_points=points
    )

    np.testing.assert_array_equal(result.reshape(-1, 2), points)
