"""Serializable observations shared by collection, fitting, and replay."""

from concurrent.futures import wait
from dataclasses import dataclass

import cv2
import numpy as np

from .config import BoardConfig
from .detection import detect_charuco_robust, setup_charuco_detector


@dataclass
class Observation:
    name: str
    image_size: tuple[int, int]
    left_ids: np.ndarray
    left_points: np.ndarray
    right_ids: np.ndarray
    right_points: np.ndarray

    def common(self):
        ids, left, right = np.intersect1d(self.left_ids, self.right_ids, return_indices=True)
        return ids, self.left_points[left], self.right_points[right]

    def to_dict(self):
        return {
            "name": self.name,
            "image_size": list(self.image_size),
            **{
                key: getattr(self, key).tolist()
                for key in ("left_ids", "left_points", "right_ids", "right_points")
            },
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["name"],
            tuple(value["image_size"]),
            *[
                np.asarray(value[k], dtype=np.int32 if k.endswith("ids") else np.float64).reshape(
                    -1
                )
                if k.endswith("ids")
                else np.asarray(value[k], dtype=np.float64).reshape(-1, 2)
                for k in ("left_ids", "left_points", "right_ids", "right_points")
            ],
        )


class Detector:
    def __init__(self, config: BoardConfig, *, executor=None):
        self.config = config
        self.executor = executor
        self.board = config.create()
        self.detectors = [setup_charuco_detector(self.board) for _ in range(2)]

    def observe(self, frame, swap=False, *, full=True):
        height, width = frame.shape[:2]
        if width % 2:
            raise ValueError("Stereo frame width must be even.")
        views = [frame[:, : width // 2], frame[:, width // 2 :]]
        if swap:
            views.reverse()
        jobs = list(zip(views, self.detectors))
        if self.executor:
            futures = [self.executor.submit(self.observe_eye, view, detector, full=full)
                       for view, detector in jobs]
            # Join both eyes even if one fails; the next frame cannot reuse a busy detector.
            wait(futures)
            results = [future.result() for future in futures]
        else:
            results = [self.observe_eye(view, detector, full=full) for view, detector in jobs]
        arrays = [array for ids, corners, _ in results for array in (ids, corners)]
        return Observation("live", (width // 2, height), *arrays), [r[2] for r in results]

    def observe_eye(self, view, detector, *, full):
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        detection, count, markers, scale, variant = detect_charuco_robust(
            gray,
            self.board,
            detector,
            min_corners=4,
            scales=(1.0, 1.5, 2.0) if full else (1.0,),
        )
        ids = np.empty(0, dtype=np.int32) if detection is None else detection.ids
        corners = np.empty((0, 2)) if detection is None else detection.corners.reshape(-1, 2)
        square_px = []
        by_id = dict(zip(ids.tolist(), corners))
        stride = self.config.columns - 1
        for i, p in by_id.items():
            neighbours = [i + stride]
            if i % stride < stride - 1:
                neighbours.append(i + 1)
            square_px.extend(
                float(np.linalg.norm(p - by_id[j])) for j in neighbours if j in by_id
            )
        # Measure only the board region; surrounding texture must not dominate sharpness.
        roi = gray
        if len(corners):
            low = np.maximum(0, np.floor(corners.min(axis=0)).astype(int))
            high = np.minimum(
                [gray.shape[1], gray.shape[0]], np.ceil(corners.max(axis=0)).astype(int) + 1
            )
            roi = gray[low[1] : high[1], low[0] : high[0]]
        quality = {
            "corners": count,
            "markers": markers,
            "detector_scale": scale,
            "detector_variant": variant,
            "marker_corner_refinement": "none",
            "square_px": float(np.median(square_px)) if square_px else None,
            "sharpness": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        }
        return ids, corners, quality


def usable(observation: Observation, board: BoardConfig, minimum=12):
    ids, _, _ = observation.common()
    if len(ids) < minimum:
        return (
            False,
            f"Need {minimum} shared corners; seeing {len(ids)}. Show more of the board to both cameras.",
        )
    objects = board.create().getChessboardCorners()[ids, :2]
    if np.linalg.matrix_rank(objects - objects.mean(axis=0)) < 2:
        return False, "Corners form a line. Show another row of the board."
    return True, "Board detected in both cameras."
