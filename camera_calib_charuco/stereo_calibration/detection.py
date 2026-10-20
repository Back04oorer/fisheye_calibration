"""Shared ChArUco detector; preserves the existing multiscale refinement policy."""

from dataclasses import dataclass

import cv2
import numpy as np
from cv2 import aruco

from .corner_refinement import refine_in_board_axes

DEFAULT_MIN_CHARUCO_CORNERS = 12


@dataclass
class CharucoDetection:
    """Reliable ChArUco corners detected in one grayscale image."""

    corners: np.ndarray
    ids: np.ndarray
    marker_count: int
    scale: float
    variant: str


def safe_corner_subpix(
    gray,
    points,
    window=(3, 3),
    max_shift_px: float | None = None,
):
    """Refine only points whose full search window lies inside the image."""
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points_array) == 0:
        return points_array, np.empty(0, dtype=int)

    flat = points_array.reshape(-1, 2)
    height, width = gray.shape[:2]
    margin_x = float(window[0] + 1)
    margin_y = float(window[1] + 1)
    valid = (
        np.all(np.isfinite(flat), axis=1)
        & (flat[:, 0] >= margin_x)
        & (flat[:, 0] < width - margin_x)
        & (flat[:, 1] >= margin_y)
        & (flat[:, 1] < height - margin_y)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return np.empty((0, 1, 2), dtype=np.float32), valid_indices

    initial = points_array[valid_indices].copy()
    refined = initial.copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    cv2.cornerSubPix(gray, refined, window, (-1, -1), criteria)

    refined_flat = refined.reshape(-1, 2)
    final_valid = (
        np.all(np.isfinite(refined_flat), axis=1)
        & (refined_flat[:, 0] >= margin_x)
        & (refined_flat[:, 0] < width - margin_x)
        & (refined_flat[:, 1] >= margin_y)
        & (refined_flat[:, 1] < height - margin_y)
    )
    if max_shift_px is not None:
        shifts = np.linalg.norm(refined_flat - initial.reshape(-1, 2), axis=1)
        final_valid &= shifts <= max_shift_px

    return refined[final_valid], valid_indices[final_valid]


def setup_charuco_detector(board):
    """Create the detector used for multiscale intrinsic observations."""
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    # OpenCV recommends leaving marker corners unrefined when interpolating
    # ChArUco with local homographies: nearby chess squares can pull their
    # subpixel windows away from the marker boundary. ChArUco chess corners
    # still receive subpixel refinement below, in the original image.
    # https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    charuco = aruco.CharucoParameters()
    # Reconsider rejected marker candidates using the known board arrangement.
    # This recovers weakly decoded markers before interpolating their corners;
    # no temporal history or fitted camera model is used to invent detections.
    charuco.tryRefineMarkers = True
    return aruco.CharucoDetector(board, charuco, parameters)


def refine_charuco_corners(gray, points):
    """Recover stalled native-pixel refinement without widening every corner's window."""
    initial = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    refined, kept = safe_corner_subpix(gray, initial, window=(3, 3), max_shift_px=5.0)
    # cornerSubPix silently restores the input when its correction exceeds the
    # window half-size. A contrast-enhanced ChArUco estimate can be 3–5 px off,
    # leaving that seed unchanged even though the native image has a clear corner.
    stalled = np.flatnonzero(np.all(refined == initial[kept], axis=(1, 2)))
    if len(stalled):
        coarse, valid = safe_corner_subpix(
            gray, initial[kept[stalled]], window=(5, 5), max_shift_px=5.0
        )
        fine, fine_valid = safe_corner_subpix(gray, coarse, window=(3, 3))
        destinations = stalled[valid[fine_valid]]
        # Keep the existing displacement budget relative to the original seed,
        # not to the intermediate estimate. Other corners retain their result.
        within_budget = np.linalg.norm(
            (fine - initial[kept[destinations]]).reshape(-1, 2), axis=1
        ) <= 5.0
        refined[destinations[within_budget]] = fine[within_budget]
    return refined, kept


def detect_charuco_robust(
    gray,
    board,
    detector=None,
    min_corners=DEFAULT_MIN_CHARUCO_CORNERS,
    scales=(1.0, 1.5, 2.0),
):
    """Return the strongest valid ChArUco detection across scale and contrast."""
    detector = detector or setup_charuco_detector(board)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    candidates = []

    for scale in scales:
        if scale == 1.0:
            scaled = gray
        else:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants = (
            ("raw", scaled),
            ("clahe", clahe.apply(scaled)),
            ("clahe_strong", clahe_strong.apply(scaled)),
        )
        for variant, image in variants:
            corners, ids, _marker_corners, marker_ids = detector.detectBoard(image)
            corner_count = 0 if ids is None else len(ids)
            marker_count = 0 if marker_ids is None else len(marker_ids)
            candidates.append(
                (
                    corner_count,
                    marker_count,
                    -scale,
                    variant,
                    scale,
                    corners,
                    ids,
                )
            )

    best = max(candidates, key=lambda item: item[:3])
    corner_count, marker_count, _, variant, scale, corners, ids = best
    if ids is None or corners is None or corner_count < min_corners:
        return None, corner_count, marker_count, scale, variant

    original_points = (np.asarray(corners, dtype=np.float32) / scale).reshape(-1, 1, 2)
    refined, kept_indices = refine_charuco_corners(gray, original_points)
    kept_ids = np.asarray(ids, dtype=np.int32).reshape(-1)[kept_indices]
    if len(kept_ids) < min_corners:
        return None, len(kept_ids), marker_count, scale, variant

    refined = refine_in_board_axes(
        gray, refined, kept_ids, board, original_points=original_points[kept_indices]
    )

    return (
        CharucoDetection(
            corners=refined.astype(np.float32),
            ids=kept_ids.astype(np.int32),
            marker_count=marker_count,
            scale=scale,
            variant=variant,
        ),
        len(kept_ids),
        marker_count,
        scale,
        variant,
    )
