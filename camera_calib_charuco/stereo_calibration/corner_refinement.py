"""Refine distorted checker corners in locally aligned image patches."""

import cv2
import numpy as np


def refine_in_board_axes(gray, points, ids, board, *, original_points, max_shift_px=5.0):
    """Use board neighbours for patch orientation, and pixels for corner location.

    Local homographies compensate for the skew of an image-axis search window.
    Each patch is centred on the measured corner, not the position predicted by
    its neighbours. No camera calibration or stereo error enters this operation.
    If the patch cannot support refinement, preserve the native-pixel estimate;
    never discard a corner to improve the calibration's reported residuals.
    """
    measured = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    original = np.asarray(original_points, dtype=np.float32).reshape(-1, 2)
    refined = measured.copy()
    if len(measured) < 5:
        return refined.reshape(-1, 1, 2)

    square = board.getSquareLength()
    objects = board.getChessboardCorners()[np.asarray(ids).reshape(-1), :2] / square
    # Normalize sampling density; restrict the gradient window to the white
    # margin before neighbouring markers, rather than widening it in sensor axes.
    samples_per_square = 64.0
    radius = int(np.floor(samples_per_square * (1 - board.getMarkerLength() / square) / 2)) - 1
    if radius < 2:
        return refined.reshape(-1, 1, 2)
    centre = 3 * radius + 3
    size = 2 * centre + 1
    vertices = np.array([[0, 0, 1], [size - 1, 0, 1],
                         [size - 1, size - 1, 1], [0, size - 1, 1]], dtype=np.float64)
    height, width = gray.shape[:2]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    for index, obj in enumerate(objects):
        distances = np.sum((objects - obj) ** 2, axis=1)
        neighbours = np.argsort(distances)[1:9]  # A local 3 × 3 corner neighbourhood.
        try:
            transform, _ = cv2.findHomography(objects[neighbours], measured[neighbours], 0)
            if transform is None or not np.all(np.isfinite(transform)):
                continue
            # Preserve the measured location even when a local planar prediction
            # is imperfect (curvature, print variation, or uneven corner coverage).
            anchor = cv2.perspectiveTransform(
                measured[index].astype(np.float64).reshape(1, 1, 2), np.linalg.inv(transform)
            ).reshape(2)
        except (cv2.error, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(anchor)):
            continue
        canonical = np.array(
            [[samples_per_square, 0, centre - samples_per_square * anchor[0]],
             [0, samples_per_square, centre - samples_per_square * anchor[1]],
             [0, 0, 1]], dtype=np.float64,
        )
        local_to_image = transform @ np.linalg.inv(canonical)
        boundary = vertices @ local_to_image.T
        denominator = boundary[:, 2]
        if not (np.all(denominator > 0) or np.all(denominator < 0)):
            continue  # Do not sample across a singularity of the local homography.
        boundary = boundary[:, :2] / denominator[:, None]
        if not np.all(np.isfinite(boundary)) or not np.all(
            (boundary > [1, 1]) & (boundary < [width - 2, height - 2])
        ):
            continue

        patch = cv2.warpPerspective(
            gray, local_to_image, (size, size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        corner = np.array([[[centre, centre]]], dtype=np.float32)
        cv2.cornerSubPix(patch, corner, (radius, radius), (-1, -1), criteria)
        if not np.all((corner >= radius + 1) & (corner < size - radius - 1)):
            continue
        pixel = cv2.perspectiveTransform(corner, local_to_image).reshape(2)
        if np.all(np.isfinite(pixel)) and np.linalg.norm(pixel - original[index]) <= max_shift_px:
            refined[index] = pixel

    return refined.reshape(-1, 1, 2)
