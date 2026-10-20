"""Regression checks for OpenCV marker-ID array compatibility."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from stereo_calib_charuco import calib_charuco as calibration
from stereo_calibration.detection import refine_charuco_corners, safe_corner_subpix


def test_stalled_contrast_seed_is_recovered_without_moving_other_refinements():
    gray = cv2.imread(
        str(Path(__file__).parent / "data/charuco_stalled_corner.png"), cv2.IMREAD_GRAYSCALE
    )
    seeds = np.array([[0, 0], [33.151855, 33.455688], [30.4, 30.3]], dtype=np.float32)
    original, kept = safe_corner_subpix(gray, seeds, window=(3, 3), max_shift_px=5.0)
    np.testing.assert_array_equal(original[0, 0], seeds[1])
    refined, indices = refine_charuco_corners(gray, seeds)
    np.testing.assert_array_equal(indices, kept)
    np.testing.assert_array_equal(indices, [1, 2])
    np.testing.assert_allclose(refined[0, 0], [30.5546, 30.3564], atol=0.02)
    np.testing.assert_array_equal(refined[1], original[1])


class DetectorWithIdShape:
    """Exercise the real detector with either supported ID array layout."""

    def __init__(self, detector, shape):
        self.detector = detector
        self.shape = shape

    def detectMarkers(self, image):
        corners, ids, rejected = self.detector.detectMarkers(image)
        return corners, ids.reshape(self.shape), rejected

    def refineDetectedMarkers(self, *args):
        result = list(self.detector.refineDetectedMarkers(*args))
        result[1] = result[1].reshape(self.shape)
        return tuple(result)


class MarkerDetectionTests(unittest.TestCase):
    def setUp(self):
        self.board, _, self.marker_3d = calibration.create_charuco_board()
        self.detector, self.clahe, self.strong = calibration.setup_detector()
        self.gray = self.board.generateImage((1140, 840), marginSize=20)

    def test_flat_and_column_ids_preserve_marker_correspondences(self):
        for shape in ((-1,), (-1, 1)):
            with self.subTest(shape=shape):
                detector = DetectorWithIdShape(self.detector, shape)
                objects, points, corners, ids = calibration.detect_markers_robust(
                    self.gray,
                    detector,
                    self.clahe,
                    self.strong,
                    self.marker_3d,
                    self.board,
                )
                self.assertIsNotNone(ids)
                expected_ids = self.board.getIds().reshape(-1)
                np.testing.assert_array_equal(np.sort(ids.reshape(-1)), np.sort(expected_ids))
                expected_objects = np.asarray(
                    [self.marker_3d[int(marker_id)] for marker_id in ids.reshape(-1)]
                ).reshape(-1, 1, 3)
                np.testing.assert_array_equal(objects, expected_objects)
                self.assertEqual(points.shape, (len(expected_ids) * 4, 1, 2))
                preview = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
                cv2.aruco.drawDetectedMarkers(preview, corners, ids)

    def test_blank_image_returns_no_detection(self):
        result = calibration.detect_markers_robust(
            np.full_like(self.gray, 255),
            self.detector,
            self.clahe,
            self.strong,
            self.marker_3d,
            self.board,
        )
        self.assertEqual(result, (None, None, None, None))


if __name__ == "__main__":
    unittest.main()


def test_board_recovers_rejected_markers_in_recorded_left_view():
    from stereo_calibration.config import BoardConfig
    from stereo_calibration.detection import detect_charuco_robust, setup_charuco_detector

    gray = cv2.imread(
        str(Path(__file__).parent / 'data/charuco_marker_recovery.png'), cv2.IMREAD_GRAYSCALE
    )
    board = BoardConfig().create()
    unassisted = setup_charuco_detector(board)
    parameters = unassisted.getCharucoParameters()
    parameters.tryRefineMarkers = False
    unassisted.setCharucoParameters(parameters)
    before = detect_charuco_robust(gray, board, unassisted, scales=(1.0,))[0]
    after = detect_charuco_robust(gray, board, scales=(1.0,))[0]
    assert len(before.ids) == 52
    assert len(after.ids) == 68
    assert set(before.ids).issubset(after.ids)
    assert {37, 38, 39, 47, 48, 49, 55, 56, 57, 58, 59, 65, 66, 67, 68, 69} <= set(after.ids)
    _, old, new = np.intersect1d(before.ids, after.ids, return_indices=True)
    np.testing.assert_allclose(after.corners[new], before.corners[old], atol=0.5)
    # Recovery uses current image evidence, not cached board positions.
    assert detect_charuco_robust(np.full_like(gray, 255), board, scales=(1.0,))[0] is None
