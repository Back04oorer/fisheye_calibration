"""Read-only, frame-matched detection and alignment diagnostics.

No board pose, capture gate or pass/fail decision belongs here. Counts describe
detection; bearing-plane alignment describes the fitted stereo geometry.
"""

import cv2
import numpy as np


def _rays(pixels, K, D):
    xy = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    if not np.all(np.isfinite(xy)) or np.any(np.all(xy == -1e6, axis=1)):
        raise ValueError("Some corners could not be undistorted.")
    rays = np.column_stack([xy, np.ones(len(xy))])
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def _sensor_pixels(rays, K, D):
    # The nearest point on an epipolar plane is expressed as a unit ray.
    radial = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(radial, rays[:, 2])
    k1, k2, k3, k4 = D.reshape(4)
    distorted = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    scale = np.divide(distorted, radial, out=np.ones_like(radial), where=radial != 0)
    return rays[:, :2] * scale[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def _correction(rays, other_rays, R, T, measured, K, D):
    normals = np.cross(np.asarray(T).reshape(3), other_rays @ R.T)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("Stereo alignment is undefined along the baseline.")
    normals /= lengths[:, None]
    corrected = rays - np.sum(rays * normals, axis=1)[:, None] * normals
    projected = _sensor_pixels(corrected, K, D)
    return np.linalg.norm(projected - measured, axis=1)


def _stats(values):
    return {
        "count": len(values),
        "p95_px": float(np.percentile(values, 95)) if len(values) else None,
    }


def inspect_observation(observation, board, model=None):
    """Return JSON-safe point locations and diagnostics for exactly one frame."""
    shared, left, right = observation.common()
    result = {
        "total": (board.columns - 1) * (board.rows - 1),
        "shared_ids": shared.tolist(),
        "points": {},
        "alignment": None,
        "rectified": None,
        "error": None,
    }
    for eye in ("left", "right"):
        result["points"][eye] = [
            {"id": int(i), "raw": p.tolist(), "rectified": None}
            for i, p in zip(
                getattr(observation, f"{eye}_ids"), getattr(observation, f"{eye}_points")
            )
        ]
    if model is None:
        return result
    try:
        if observation.image_size != model.image_size:
            raise ValueError("Frame size does not match the calibration.")
        rectified = {}
        for eye, K, D, R, P in (
            ("left", model.K1, model.D1, model.R1, model.P1),
            ("right", model.K2, model.D2, model.R2, model.P2),
        ):
            pixels = getattr(observation, f"{eye}_points")
            projected = (
                cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), K, D, R=R, P=P)
                .reshape(-1, 2)
                if len(pixels)
                else np.empty((0, 2))
            )
            if not np.all(np.isfinite(projected)):
                raise ValueError("Some rectified corners are not finite.")
            rectified[eye] = {}
            for point, xy in zip(result["points"][eye], projected):
                point["rectified"] = xy.tolist()
                rectified[eye][point["id"]] = xy
        if not len(shared):
            return result
        a = np.array([rectified["left"][int(i)] for i in shared])
        b = np.array([rectified["right"][int(i)] for i in shared])
        inside = np.all((a >= 0) & (a < model.image_size), axis=1) & np.all(
            (b >= 0) & (b < model.image_size), axis=1
        )
        vertical = np.abs(a[:, 1] - b[:, 1])
        result["rectified"] = {"all": _stats(vertical), "in_view": _stats(vertical[inside])}
        u, v = _rays(left, model.K1, model.D1), _rays(right, model.K2, model.D2)
        left_error = _correction(
            u, v, model.R.T, -model.R.T @ model.T, left, model.K1, model.D1
        )
        right_error = _correction(v, u, model.R, model.T, right, model.K2, model.D2)
        errors = np.maximum(left_error, right_error)
        if not np.all(np.isfinite(errors)):
            raise ValueError("Stereo alignment could not be measured for every shared corner.")
        result["alignment"] = _stats(errors)
    except (ValueError, cv2.error) as error:
        # A live measurement failure must not interrupt collection or reuse old scores.
        result["error"] = str(error)
    return result
