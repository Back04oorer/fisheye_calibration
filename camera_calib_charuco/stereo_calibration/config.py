"""Physical board definitions; all lengths use millimetres."""

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardConfig:
    columns: int = 11
    rows: int = 8
    square_mm: float = 20.0
    marker_mm: float = 15.0
    dictionary: str = "DICT_5X5_100"
    legacy: bool = True
    marker_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.columns < 3 or self.rows < 3:
            raise ValueError("A ChArUco board needs at least 3 rows and 3 columns of squares.")
        if not (
            math.isfinite(self.square_mm)
            and math.isfinite(self.marker_mm)
            and 0 < self.marker_mm < self.square_mm
        ):
            raise ValueError("Marker length must be positive and smaller than the square length.")
        if not self.dictionary.startswith("DICT_") or not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")
        if self.marker_ids is not None:
            ids = tuple(self.marker_ids)
            count = self.columns * self.rows // 2
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
            if (len(ids) != count or len(set(ids)) != count
                    or any(type(i) is not int or i < 0 or i >= len(dictionary.bytesList) for i in ids)):
                raise ValueError("Marker IDs must be unique, in the dictionary, and match the board.")
            object.__setattr__(self, "marker_ids", ids)

    def create(self):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        args = [] if self.marker_ids is None else [np.asarray(self.marker_ids, dtype=np.int32)]
        board = cv2.aruco.CharucoBoard(
            (self.columns, self.rows), self.square_mm, self.marker_mm, dictionary, *args
        )
        board.setLegacyPattern(self.legacy)
        return board

    def geometry(self):
        board = self.create()
        markers = {
            int(i): np.asarray(p, dtype=np.float32)
            for i, p in zip(board.getIds().reshape(-1), board.getObjPoints())
        }
        return board, board.getChessboardCorners(), markers

    def to_dict(self):
        result = asdict(self)
        if self.marker_ids is None:
            result.pop("marker_ids")  # Keep existing session definitions byte-compatible.
        else:
            result["marker_ids"] = list(self.marker_ids)
        return result


PRESETS = {
    "existing-20mm": BoardConfig(),
    "a4-30mm": BoardConfig(columns=8, rows=6, square_mm=30.0, marker_mm=22.5),
}


@dataclass(frozen=True)
class CameraConfig:
    device: int | str = 0
    width: int = 2560
    height: int = 720
    fps: int = 30
    swap: bool = False

    def __post_init__(self):
        if self.width < 2 or self.width % 2 or self.height < 1 or self.fps < 1:
            raise ValueError("Side-by-side capture requires an even width and positive height/FPS.")

    @property
    def eye_size(self):
        return self.width // 2, self.height

    def to_dict(self):
        return asdict(self)
