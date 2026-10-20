"""Known-geometry camera observations, independent of the fitting implementation."""

import cv2
import numpy as np

from stereo_calibration.config import BoardConfig
from stereo_calibration.observations import Observation


def observations(count=20, seed=42):
    board = BoardConfig()
    objects = board.create().getChessboardCorners().astype(np.float64).reshape(1, -1, 3)
    K = np.array([[410.0, 0.0, 640.0], [0.0, 405.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.03, 0.002, -0.0001, 0.00001])
    R = cv2.Rodrigues(np.array([0.003, -0.008, 0.001]))[0]
    T = np.array([-60.0, 0.5, 0.3])
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        r = rng.uniform([-0.6, -0.6, -0.3], [0.6, 0.6, 0.3])
        t = rng.uniform([-190, -100, 350], [0, 20, 700])
        rr = cv2.Rodrigues(R @ cv2.Rodrigues(r)[0])[0]
        right_t = R @ t + T
        left = cv2.fisheye.projectPoints(objects, r, t, K, D)[0].reshape(-1, 2)
        right = cv2.fisheye.projectPoints(objects, rr, right_t, K, D)[0].reshape(-1, 2)
        ids = np.arange(objects.shape[1], dtype=np.int32)
        result.append(Observation(f"synthetic-{i}", (1280, 720), ids, left, ids.copy(), right))
    return board, result, K, D, R, T


def regional_observations(board, K, D, R, T):
    """Known poses covering each guide region, independent of the fitted model."""
    from stereo_calibration.guidance import POSITIONS

    obj = board.create().getChessboardCorners().astype(np.float64).reshape(1, -1, 3)
    centre = np.array([board.columns * board.square_mm / 2, board.rows * board.square_mm / 2, 0.0])
    result = []
    for i, (name, x, y) in enumerate(POSITIONS):
        ray = cv2.fisheye.undistortPoints(np.array([[[x * 1280, y * 720]]]), K, D).reshape(2)
        r = np.array([0.1 * (i % 3 - 1), 0.12 * (i % 2 * 2 - 1), 0.02])
        t = np.r_[ray * 500, 500.0] - cv2.Rodrigues(r)[0] @ centre
        left = cv2.fisheye.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
        rr = cv2.Rodrigues(R @ cv2.Rodrigues(r)[0])[0]
        right = cv2.fisheye.projectPoints(obj, rr, R @ t + T, K, D)[0].reshape(-1, 2)
        ids = np.arange(len(left), dtype=np.int32)
        result.append(Observation(name, (1280, 720), ids, left, ids.copy(), right))
    return result
