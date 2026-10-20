"""Evaluate frozen calibrations on complete, fresh observations."""

import cv2
import numpy as np

from .calibration import Calibration
from .observations import Observation
from .opencv import points

# Pixel-error budgets, shared by training diagnostics and independent checks.
RMS_TARGET_PX = 1.0
VERTICAL_MEDIAN_TARGET_PX = 0.5
VERTICAL_P95_TARGET_PX = 2.0


def coverage(observations, eye):
    counts = np.zeros((3, 3), dtype=int)
    for observation in observations:
        pixels = getattr(observation, f"{eye}_points")
        for x, y in np.clip((pixels / observation.image_size * 3).astype(int), 0, 2):
            counts[y, x] += 1
    return counts.tolist()


def rectified_errors(observation: Observation, model: Calibration):
    if observation.image_size != model.image_size:
        raise ValueError("Observation resolution does not match the frozen calibration.")
    ids, left, right = observation.common()
    a = cv2.fisheye.undistortPoints(
        points(left, 2), model.K1, model.D1, R=model.R1, P=model.P1
    ).reshape(-1, 2)
    b = cv2.fisheye.undistortPoints(
        points(right, 2), model.K2, model.D2, R=model.R2, P=model.P2
    ).reshape(-1, 2)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Rectification produced non-finite points; this view cannot pass.")
    return ids, a[:, 1] - b[:, 1]


def board_pose(objects, images, K, D):
    bearings = cv2.fisheye.undistortPoints(points(images, 2), K, D)
    if not np.all(np.isfinite(bearings)):
        raise ValueError("Cannot estimate board pose from these fisheye bearings.")
    ok, rvec, tvec = cv2.solvePnP(points(objects, 3), bearings, np.eye(3), np.zeros(4))
    if not ok or tvec[2, 0] <= 0:
        raise ValueError("Board pose could not be estimated in front of the camera.")
    return rvec, tvec


def view_metrics(observation, model, board):
    ids, errors = rectified_errors(observation, model)
    if len(ids) < 12:
        raise ValueError("Fewer than 12 common validation corners.")
    absolute = np.abs(errors)
    result = {
        "name": observation.name,
        "corners": len(ids),
        "median_px": float(np.median(absolute)),
        "p95_px": float(np.percentile(absolute, 95)),
        "maximum_px": float(absolute.max()),
        "mean_px": float(absolute.mean()),
    }
    # Rectified vertical errors remain useful diagnostics, but their scale is
    # chosen by P1/P2 and can explode near the perspective horizon. Acceptance
    # instead uses mono and inter-camera reprojection in original sensor pixels.
    result["vertical_diagnostic_passed"] = (
        result["median_px"] < VERTICAL_MEDIAN_TARGET_PX
        and result["p95_px"] < VERTICAL_P95_TARGET_PX
    )
    failures, poses = [], {}
    objects = board.create().getChessboardCorners().astype(np.float64)
    for eye, K, D in (("left", model.K1, model.D1), ("right", model.K2, model.D2)):
        eye_ids = getattr(observation, f"{eye}_ids")
        images = getattr(observation, f"{eye}_points")
        obj = objects[eye_ids]
        r, t = board_pose(obj, images, K, D)
        poses[eye] = (cv2.Rodrigues(r)[0], t)
        projected = cv2.fisheye.projectPoints(points(obj, 3), r, t, K, D)[0].reshape(-1, 2)
        residual = projected - images
        if not np.all(np.isfinite(residual)):
            raise ValueError(f"{eye.title()} reprojection produced non-finite residuals.")
        result[f"{eye}_reprojection_rms_px"] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        if result[f"{eye}_reprojection_rms_px"] >= RMS_TARGET_PX:
            failures.append(f"{eye.title()} reprojection RMS exceeds the 1 px target")
        result[f"{eye}_residuals"] = np.column_stack([images, residual]).tolist()
    _, left, right = observation.common()
    for direction, source, transform, translation, K, D, measured in (
        ("left_to_right", "left", model.R, model.T, model.K2, model.D2, right),
        ("right_to_left", "right", model.R.T, -model.R.T @ model.T, model.K1, model.D1, left),
    ):
        rotation, position = poses[source]
        r = cv2.Rodrigues(transform @ rotation)[0]
        t = transform @ position + translation.reshape(3, 1)
        predicted = cv2.fisheye.projectPoints(points(objects[ids], 3), r, t, K, D)[0].reshape(-1, 2)
        residual = predicted - measured
        if not np.all(np.isfinite(residual)):
            raise ValueError(f"{direction} projection produced non-finite residuals.")
        result[f"{direction}_rms_px"] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        result[f"{direction}_residuals"] = np.column_stack([measured, residual]).tolist()
        if result[f"{direction}_rms_px"] >= RMS_TARGET_PX:
            failures.append(f"Stereo {direction.replace('_', ' ')} RMS exceeds the 1 px target")
    _, left, _ = observation.common()
    result["vertical_errors"] = np.column_stack([left, errors]).tolist()
    result["failed_checks"] = failures
    result["passed"] = not failures
    return result


def evaluate(observations, model, board, *, goals_complete=False):
    views = []
    for observation in observations:
        try:
            views.append(view_metrics(observation, model, board))
        except (ValueError, cv2.error) as error:
            # A failed held-out view remains in the denominator and in the report.
            views.append({"name": observation.name, "passed": False, "error": str(error)})
    fit_ok = max(model.rms) < RMS_TARGET_PX
    passed = bool(goals_complete and views and fit_ok and all(v["passed"] for v in views))
    return {
        "passed": passed,
        "scope": "Geometric consistency; physical depth accuracy not measured.",
        "fit_rms_px": list(model.rms),
        "fit_passed": fit_ok,
        "baseline_mm": float(np.linalg.norm(model.T)),
        "validation_views": views,
        "goals_complete": goals_complete,
        "coverage": {eye: coverage(observations, eye) for eye in ("left", "right")},
        "thresholds": {
            "fit_rms_px": RMS_TARGET_PX,
            "fresh_reprojection_rms_px": RMS_TARGET_PX,
            "fresh_inter_camera_rms_px": RMS_TARGET_PX,
        },
        "vertical_diagnostic_thresholds": {
            "median_px": VERTICAL_MEDIAN_TARGET_PX,
            "p95_px": VERTICAL_P95_TARGET_PX,
        },
        "projection": (
            "Acceptance: original full-resolution camera pixels; mono and bidirectional "
            "inter-camera reprojection. Rectified vertical errors: diagnostics only. "
            "No viewport cropping or held-out corner filtering."
        ),
    }
