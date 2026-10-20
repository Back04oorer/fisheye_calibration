"""Intrinsic fitting and the existing, training-only corner filtering policy."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .opencv import fit_intrinsics


def write_log(path, message):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now().isoformat()}] {message}\n")
    print(message)


def calibrate_fisheye(observations, image_size, *, initial=None):
    return fit_intrinsics(
        [v.object_points for v in observations],
        [v.image_points for v in observations],
        image_size,
        initial=initial,
    )


@dataclass
class CalibrationObservation:
    """Object and image points retained from one source image."""

    name: str
    path: Path
    object_points: np.ndarray
    image_points: np.ndarray
    corner_ids: np.ndarray
    initial_corner_count: int
    marker_count: int
    scale: float
    variant: str
    blur_score: float


@dataclass
class RemovedCorner:
    """A ChArUco point removed after a large reprojection residual."""

    image: str
    corner_id: int
    residual_px: float
    x: float
    y: float
    iteration: int


def fisheye_residuals(observations, camera_matrix, distortion, rvecs, tvecs):
    """Return per-view RMS values and each point's Euclidean residual."""
    per_view = []
    per_point = []
    for item, rvec, tvec in zip(observations, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(
            item.object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - item.image_points.reshape(-1, 2), axis=1)
        per_point.append(errors)
        per_view.append(float(np.sqrt(np.mean(errors**2))))
    return np.asarray(per_view), per_point


def filter_corner_outliers(
    observations,
    image_size,
    threshold_px,
    min_corners,
    max_iterations,
    log_file=None,
):
    """Remove isolated high-residual corners without discarding useful views."""
    removed = []
    result = None
    for iteration in range(max_iterations + 1):
        # Retain a converged seed when refitting after training-corner removal.
        result = calibrate_fisheye(
            observations, image_size, initial=result[1:3] if result is not None else None
        )
        rms, matrix, distortion, rvecs, tvecs = result
        _, point_errors = fisheye_residuals(observations, matrix, distortion, rvecs, tvecs)
        if iteration == max_iterations:
            break

        removed_this_iteration = 0
        for item, errors in zip(observations, point_errors):
            keep = errors <= threshold_px
            if int(np.sum(keep)) < min_corners:
                keep = np.zeros(len(errors), dtype=bool)
                keep[np.argsort(errors)[:min_corners]] = True
            if np.all(keep):
                continue

            points = item.image_points.reshape(-1, 2)
            for corner_id, error, point in zip(
                item.corner_ids[~keep], errors[~keep], points[~keep]
            ):
                removed.append(
                    RemovedCorner(
                        item.name,
                        int(corner_id),
                        float(error),
                        float(point[0]),
                        float(point[1]),
                        iteration + 1,
                    )
                )
            item.object_points = item.object_points[keep]
            item.image_points = item.image_points[keep]
            item.corner_ids = item.corner_ids[keep]
            removed_this_iteration += int(np.sum(~keep))

        if log_file:
            write_log(
                log_file,
                f"角点离群迭代 {iteration + 1}: RMS={rms:.6f} px, "
                f"删除 {removed_this_iteration} 个角点",
            )
        if removed_this_iteration == 0:
            break

    return result, removed
