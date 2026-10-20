"""A finite horizontal perspective viewport for a wide-angle stereo model."""

import cv2
import numpy as np

from .opencv import fisheye_flag


def horizontal_rectification(K1, D1, K2, D2, size, R, T):
    # OpenCV's rotations align the baseline correctly, but its boundary-based
    # viewport estimator can collapse when fisheye image edges cross 90 degrees.
    R1, R2, *_ = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=fisheye_flag("CALIB_ZERO_DISPARITY")
    )
    # Preserve the mean central angular pixel scale. This is a viewport choice,
    # not an accuracy setting: validation works in the original camera pixels.
    focal = float(np.mean([K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]]))
    cx, cy = (np.asarray(size, dtype=float) - 1) / 2
    tx = float((R2 @ np.asarray(T).reshape(3))[0])
    if not np.isfinite(focal) or focal <= 0 or not np.isfinite(tx) or tx == 0:
        raise ValueError("Cannot form a finite horizontal stereo projection.")
    P1 = np.array([[focal, 0, cx, 0], [0, focal, cy, 0], [0, 0, 1, 0]])
    P2 = P1.copy()
    P2[0, 3] = focal * tx
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, focal], [0, 0, -1 / tx, 0]])
    return R1, R2, P1, P2, Q
