"""Small, explicit adapter for the OpenCV 4.13 and 5.0 fisheye bindings."""

import logging

import cv2
import numpy as np


def fisheye_flag(name: str) -> int:
    # OpenCV 5 moved these flags and changed their values. Never copy numeric flags.
    owner = cv2.fisheye if hasattr(cv2.fisheye, name) else cv2
    return getattr(owner, name)


def points(value, dimensions):
    # (1,N,C) works in both versions; (N,1,C) breaks intrinsic fitting in 5.0.
    return np.ascontiguousarray(value, dtype=np.float64).reshape(1, -1, dimensions)


def fit_intrinsics(objects, images, image_size, *, initial=None):
    """Fit every view, bootstrapping K from central views if pose initialization fails.

    OpenCV's homography-based pose initializer can fail for wide-angle edge views
    with a generic camera matrix. A central subset supplies an initial K only;
    the returned optimization always uses all observations and retains CHECK_COND.
    """
    width, height = image_size
    default_matrix = np.array(
        [[width * 0.375, 0, (width - 1) / 2], [0, width * 0.375, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    flags = (
        fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")
        | fisheye_flag("CALIB_CHECK_COND")
        | fisheye_flag("CALIB_FIX_SKEW")
        | fisheye_flag("CALIB_USE_INTRINSIC_GUESS")
    )
    object_points = [points(x, 3) for x in objects]
    image_points = [points(x, 2) for x in images]

    def solve(obj, img, matrix, distortion=None):
        return cv2.fisheye.calibrate(
            obj,
            img,
            image_size,
            matrix.copy(),
            np.zeros((4, 1)) if distortion is None else distortion.copy(),
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-11),
        )

    def initialization_failure(error):
        return error.func in ("InitExtrinsics", "CalibrateExtrinsics", "ComputeJacobians")

    matrix, distortion = initial if initial is not None else (default_matrix, None)
    try:
        return solve(object_points, image_points, matrix, distortion)
    except cv2.error as error:
        if not initialization_failure(error):
            raise
        original_error = error

    # Increasing prefixes let the data determine how many central views suffice.
    centre = np.asarray(image_size, dtype=float) / 2
    order = sorted(
        range(len(image_points)),
        key=lambda i: float(
            np.max(np.linalg.norm(image_points[i].reshape(-1, 2) - centre, axis=1))
        ),
    )
    for count in range(3, len(object_points)):
        selected = order[:count]
        try:
            seed = solve(
                [object_points[i] for i in selected],
                [image_points[i] for i in selected],
                default_matrix,
            )
            # Central views poorly constrain high-order peripheral distortion.
            # Initialize K from them and estimate all four D coefficients on all views.
            result = solve(object_points, image_points, seed[1])
        except cv2.error as error:
            if not initialization_failure(error):
                raise
            continue
        logging.getLogger(__name__).info(
            "Recovered fisheye initialization from %d central views; final fit used all %d views",
            count,
            len(object_points),
        )
        return result
    raise original_error
