"""Fisheye fitting. Validation observations are never accepted by this module's session API."""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import BoardConfig
from .intrinsics import CalibrationObservation, filter_corner_outliers
from .observations import Observation, usable
from .opencv import fisheye_flag, points
from .rectification import horizontal_rectification

# A minimal multi-view seed, never an acceptance criterion. The guided workflow
# also requires pose diversity; fresh measurements decide whether a fit passes.
MIN_FIT_VIEWS = 3


@dataclass
class Calibration:
    image_size: tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    rms: tuple[float, float, float]

    def to_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data):
        model = cls(
            **{
                k: tuple(v) if k in ("image_size", "rms") else np.asarray(v, dtype=np.float64)
                for k, v in data.items()
            }
        )
        model.check()
        return model

    def check(self):
        for key, shape in {
            "K1": (3, 3),
            "K2": (3, 3),
            "R": (3, 3),
            "R1": (3, 3),
            "R2": (3, 3),
            "P1": (3, 4),
            "P2": (3, 4),
            "Q": (4, 4),
        }.items():
            value = getattr(self, key)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid calibration matrix: {key}")
        if any(d.size != 4 or not np.all(np.isfinite(d)) for d in (self.D1, self.D2)):
            raise ValueError("Expected four finite fisheye distortion coefficients.")
        if any(k[0, 0] <= 0 or k[1, 1] <= 0 for k in (self.K1, self.K2)):
            raise ValueError("Fitted focal lengths must be positive.")
        if self.T.size != 3 or not np.all(np.isfinite(self.T)) or np.linalg.norm(self.T) <= 0:
            raise ValueError("Stereo baseline is not valid.")
        if (
            len(self.image_size) != 2
            or min(self.image_size) <= 0
            or not np.all(np.isfinite(self.rms))
        ):
            raise ValueError("Invalid calibration size or RMS.")


def distributed_indices(left, right, count, size):
    features = np.hstack((left / size, right / size))
    first = int(np.argmax(np.sum((features - features.mean(axis=0)) ** 2, axis=1)))
    selected = [first]
    distance = np.sum((features - features[first]) ** 2, axis=1)
    while len(selected) < count:
        distance[selected] = -1
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((features - features[index]) ** 2, axis=1))
    return np.asarray(selected)


def fit(observations: list[Observation], board: BoardConfig):
    if len(observations) < MIN_FIT_VIEWS:
        raise ValueError("A numerical seed needs at least three distinct board poses.")
    size = observations[0].image_size
    objects = board.create().getChessboardCorners().astype(np.float64)
    for observation in observations:
        if observation.image_size != size:
            raise ValueError("Training images have different resolutions.")
        valid, reason = usable(observation, board)
        if not valid:
            raise ValueError(f"{observation.name}: {reason}")
    results, retained, removed = [], [], {}
    for eye in ("left", "right"):
        views = []
        for observation in observations:
            ids = getattr(observation, f"{eye}_ids").copy()
            images = getattr(observation, f"{eye}_points").copy()
            views.append(
                CalibrationObservation(
                    observation.name,
                    Path(observation.name),
                    objects[ids].reshape(-1, 1, 3),
                    images.reshape(-1, 1, 2),
                    ids,
                    len(ids),
                    0,
                    1.0,
                    "recorded",
                    0.0,
                )
            )
        result, exclusions = filter_corner_outliers(views, size, 1.0, 12, 5)
        results.append(result)
        retained.append(views)
        removed[eye] = [asdict(x) for x in exclusions]
    # Intersect the independently retained IDs, not the pre-filter detections.
    stereo, excluded_pairs = [], []
    for left, right in zip(*retained):
        ids, li, ri = np.intersect1d(left.corner_ids, right.corner_ids, return_indices=True)
        if (
            len(ids) < 12
            or np.linalg.matrix_rank(objects[ids, :2] - objects[ids, :2].mean(axis=0)) < 2
        ):
            excluded_pairs.append(left.name)
            continue
        stereo.append(
            (
                objects[ids],
                left.image_points.reshape(-1, 2)[li],
                right.image_points.reshape(-1, 2)[ri],
            )
        )
    if len(stereo) < MIN_FIT_VIEWS:
        raise ValueError("Too few usable poses remain after training corner filtering.")
    count = min(len(x[0]) for x in stereo)
    selected = []
    for obj, left, right in stereo:
        idx = distributed_indices(left, right, count, np.asarray(size))
        selected.append((points(obj[idx], 3), points(left[idx], 2), points(right[idx], 2)))
    K1, D1 = results[0][1:3]
    K2, D2 = results[1][1:3]
    values = cv2.fisheye.stereoCalibrate(
        *[list(x) for x in zip(*selected)],
        K1.copy(),
        D1.copy(),
        K2.copy(),
        D2.copy(),
        size,
        flags=fisheye_flag("CALIB_FIX_INTRINSIC"),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )
    rms, _, _, _, _, R, T = values[:7]
    rectified = horizontal_rectification(K1, D1, K2, D2, size, R, T)
    model = Calibration(
        size,
        K1,
        D1,
        K2,
        D2,
        R,
        T,
        *rectified,
        (float(results[0][0]), float(results[1][0]), float(rms)),
    )
    model.check()
    return model, {
        "removed_training_corners": removed,
        "excluded_stereo_pairs": excluded_pairs,
        "stereo_points_per_pair": count,
        "training_pairs": len(observations),
    }


def export(model: Calibration, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    matrices = {k: v for k, v in asdict(model).items() if isinstance(v, np.ndarray)}
    np.savez(
        directory / "stereo_params.npz",
        **matrices,
        image_size=model.image_size,
        model="fisheye",
        rms=model.rms,
        length_unit="mm",
        translation_convention="X_right=R*X_left+T",
    )
    storage = cv2.FileStorage(str(directory / "stereo_params.yaml"), cv2.FILE_STORAGE_WRITE)
    try:
        for key, value in matrices.items():
            storage.write(key, value)
        storage.write("model", "fisheye")
        storage.write("distortion_model", "fisheye")
        storage.write("image_width", model.image_size[0])
        storage.write("image_height", model.image_size[1])
        storage.write("length_unit", "mm")
    finally:
        storage.release()
    for eye, K, D in (("left", model.K1, model.D1), ("right", model.K2, model.D2)):
        np.savez(
            directory / f"{eye}_fisheye.npz",
            mtx=K,
            dist=D,
            model="fisheye",
            image_size=model.image_size,
        )
