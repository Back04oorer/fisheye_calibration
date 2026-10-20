"""Explainable movement goals and capture debounce, separate from accuracy checks.

The tolerances below are operator-interface heuristics, not accuracy guarantees.
They normalize motion by board size so changing preview resolution cannot change capture.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import MIN_FIT_VIEWS
from .validation import board_pose


@dataclass(frozen=True)
class Goal:
    key: str
    title: str
    instruction: str
    kind: str
    target: tuple[float, float] | None = None


# Radius in normalized left-image coordinates, also drawn by the preview renderer.
POSITION_RADIUS = 0.14
SCALE_NOVELTY = 0.14  # Difference in log apparent board scale.

POSITIONS = [
    ("centre", 0.5, 0.5),
    ("upper-left", 0.2, 0.2),
    ("top", 0.5, 0.2),
    ("upper-right", 0.8, 0.2),
    ("right", 0.8, 0.5),
    ("lower-right", 0.8, 0.8),
    ("bottom", 0.5, 0.8),
    ("lower-left", 0.2, 0.8),
    ("left", 0.2, 0.5),
]
TRAINING_GOALS = [
    Goal(
        name,
        name.replace("-", " ").title(),
        f"Move the pattern's centre anywhere inside the blue {name.replace('-', ' ')} area "
        "in the left view. Keep corners visible in both cameras; tilt or move closer around obstructions.",
        "position",
        (x, y),
    )
    for name, x, y in POSITIONS
] + [
    Goal(
        "near",
        "Come closer",
        "Bring the board closer until it appears about one-third larger.",
        "near",
    ),
    *[
        Goal(
            f"tilt-{edge}",
            f"Tilt {edge} edge",
            f"Tilt the board's printed {edge} edge toward the cameras.",
            edge,
        )
        for edge in ("left", "right", "top", "bottom")
    ],
]
VALIDATION_GOALS = [
    Goal(
        f"check-{name}",
        f"Check {name.replace('-', ' ')}",
        f"Fresh check: put the pattern's centre inside the blue {name.replace('-', ' ')} area "
        "in the left view, with corners visible in both cameras. "
        + (
            "Hold near your working distance."
            if i == 0
            else (
                "Use a different distance and a gentle tilt."
                if i % 2
                else "Change the board's tilt from the last view."
            )
        ),
        "position",
        (x, y),
    )
    for i, (name, x, y) in enumerate(POSITIONS)
]


def ready_to_fit(signatures, goals):
    """Pose diversity permits a numerical attempt, not a calibration pass."""
    return (
        len(signatures) >= MIN_FIT_VIEWS
        and "centre" in goals
        and bool({"tilt-left", "tilt-right"} & goals)
        and bool({"tilt-top", "tilt-bottom"} & goals)
    )


def describe(observation, board, model=None):
    ids, left, _ = observation.common()
    if len(ids) < 4:
        return None
    obj = board.create().getChessboardCorners().astype(np.float64)
    outline = np.array(
        [
            [0, 0, 0],
            [board.columns * board.square_mm, 0, 0],
            [board.columns * board.square_mm, board.rows * board.square_mm, 0],
            [0, board.rows * board.square_mm, 0],
        ],
        dtype=np.float64,
    )
    board_centre = outline.mean(axis=0)
    outline_visible = True
    tilt_x = tilt_y = None
    rotation = None
    translation = None
    if model is not None:
        r, t = board_pose(obj[ids], left, model.K1, model.D1)
        projected = cv2.fisheye.projectPoints(
            np.vstack([outline, board_centre]).reshape(1, -1, 3), r, t, model.K1, model.D1
        )[0].reshape(-1, 2)
        polygon, centre = projected[:4], projected[4]
        rotation = cv2.Rodrigues(r)[0]
        translation = t.reshape(3) / 1000
        outline_visible = bool(np.all((outline @ rotation.T + t.reshape(1, 3))[:, 2] > 0))
        # Positive x: printed right edge is closer; positive y: printed bottom edge is closer.
        tilt_x = math.degrees(math.atan2(-rotation[2, 0], np.linalg.norm(rotation[:2, 0])))
        tilt_y = math.degrees(math.atan2(-rotation[2, 1], np.linalg.norm(rotation[:2, 1])))
    else:
        H, _ = cv2.findHomography(obj[ids, :2], left, method=0)
        if H is None:
            return None
        polygon = cv2.perspectiveTransform(outline[None, :, :2], H).reshape(-1, 2)
        centre = cv2.perspectiveTransform(board_centre[None, None, :2], H).reshape(2)
    if not np.all(np.isfinite(polygon)) or not np.all(np.isfinite(centre)):
        return None
    tl, tr, br, bl = polygon
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left_edge, right_edge = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    footprint = polygon if outline_visible else cv2.convexHull(left.astype(np.float32))
    scale = math.sqrt(
        abs(cv2.contourArea(footprint.astype(np.float32))) / np.prod(observation.image_size)
    )
    # Perspective and fisheye projection are nonlinear: averaging projected
    # vertices does not give the image of the physical pattern centre.
    centre = centre / observation.image_size
    return {
        "x": float(centre[0]),
        "y": float(centre[1]),
        "scale": scale,
        "skew_x": math.log(max(right_edge, 1) / max(left_edge, 1)),
        "skew_y": math.log(max(bottom, 1) / max(top, 1)),
        "roll": math.atan2((tr - tl)[1], (tr - tl)[0]),
        "tilt_x": tilt_x,
        "tilt_y": tilt_y,
        "outline": polygon.tolist() if outline_visible else [],
        # Advisory display only; this monocular estimate does not qualify a
        # hand-eye capture or replace the independent stereo consistency check.
        "rotation_camera": rotation.tolist() if rotation is not None and outline_visible else None,
        "translation_camera_m": translation.tolist()
        if translation is not None and outline_visible
        else None,
        "pose_shared_corners": len(ids),
        "pose_layout_rank": int(np.linalg.matrix_rank(obj[ids, :2] - obj[ids, :2].mean(axis=0))),
    }


def different(a, b):
    if not a or not b:
        return True
    return (
        math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 0.07
        or abs(math.log(max(a["scale"], 0.001) / max(b["scale"], 0.001))) > SCALE_NOVELTY
        or abs(a["skew_x"] - b["skew_x"]) > 0.12
        or abs(a["skew_y"] - b["skew_y"]) > 0.12
        or abs(math.atan2(math.sin(a["roll"] - b["roll"]), math.cos(a["roll"] - b["roll"])))
        > math.radians(20)
        or (
            a["tilt_x"] is not None
            and b["tilt_x"] is not None
            and max(abs(a["tilt_x"] - b["tilt_x"]), abs(a["tilt_y"] - b["tilt_y"])) > 15
        )
    )


def achieved(signature, reference_scale, *, validation=False):
    if signature is None:
        return []
    goals = VALIDATION_GOALS if validation else TRAINING_GOALS
    result = []
    for goal in goals:
        if goal.kind == "position":
            good = (
                math.hypot(signature["x"] - goal.target[0], signature["y"] - goal.target[1])
                < POSITION_RADIUS
            )
        elif goal.kind == "near":
            good = reference_scale is not None and signature["scale"] >= reference_scale * 1.3
        else:
            axis = "x" if goal.kind in ("left", "right") else "y"
            value = signature[f"tilt_{axis}"]
            threshold = 15.0
            if value is None:
                value, threshold = signature[f"skew_{axis}"], 0.12
            good = value * (-1 if goal.kind in ("left", "top") else 1) > threshold
        if good:
            result.append(goal.key)
    return result


def capture_ready(signature, previous, last_capture, done_goals, reference_scale, *, validation):
    """The same pose/coverage decision drives capture and the live hold cue."""
    if signature is None or not all(different(signature, old) for old in previous):
        return False
    if not different(signature, last_capture):
        return False
    return not validation or bool(
        set(achieved(signature, reference_scale, validation=True)) - done_goals
    )


def live_adjustment(
    goal, signature, reference_scale, done_goals, *, ready=False, validation=False, previous=()
):
    """An advisory next action, without adding capture gates or pose requirements."""
    current = [signature["x"], signature["y"]] if signature else None
    if ready:
        return {"kind": "ready"}
    if goal is None:
        return {"kind": "unavailable"}
    matched = set(achieved(signature, reference_scale, validation=validation))
    if goal.kind == "position" and goal.key not in matched:
        return {"kind": "move", "target": goal.target, "current": current}
    if signature is None:
        return {"kind": "unavailable"}
    if goal.kind in ("left", "right", "top", "bottom") and goal.key not in matched:
        return {"kind": "tilt", "edge": goal.kind}
    if goal.kind == "near" and goal.key not in matched:
        return {"kind": "distance", "direction": "closer"}
    # Already in the requested area, but this pose duplicates a saved view.
    # Combine it with an outstanding tilt when useful, otherwise vary distance.
    if not validation:
        for missing in TRAINING_GOALS:
            if missing.kind in ("left", "right", "top", "bottom") and (
                missing.key not in done_goals and missing.key not in matched
            ):
                return {"kind": "tilt", "edge": missing.kind}
    # Choose the nearer end of the already-used scale range. This direction
    # stays consistent as the operator moves, even across the "near" goal size.
    bands = []
    for old in previous:
        if old and not different(dict(signature, scale=old["scale"]), old):
            scale = math.log(max(old["scale"], 0.001))
            bands.append((scale - SCALE_NOVELTY, scale + SCALE_NOVELTY))
    merged = []
    for low, high in sorted(bands):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low, high])
    current_scale = math.log(max(signature["scale"], 0.001))
    direction = "closer"
    for low, high in merged:
        if low <= current_scale <= high:
            direction = "farther" if current_scale - low < high - current_scale else "closer"
            break
    return {"kind": "distance", "direction": direction}


class Hold:
    """Accept one steady observation per new pose. Pause/detection loss clears the hold."""

    def __init__(self, seconds=0.7, maximum_gap=0.75):
        self.seconds = seconds
        self.maximum_gap = maximum_gap
        self.reset()

    def reset(self):
        self.anchor = None
        self.started = None
        self.last = None
        self.frames = 0
        self.best = None

    def update(self, observation, timestamp, frame, quality):
        if self.last is not None and timestamp <= self.last:
            self.reset()
            # A repeated/out-of-order image is not a new stability observation.
            return 0.0, None
        if self.last is not None and timestamp - self.last > self.maximum_gap:
            self.reset()
        if self.anchor is not None:
            for eye in ("left", "right"):
                ids, ai, bi = np.intersect1d(
                    getattr(self.anchor, f"{eye}_ids"),
                    getattr(observation, f"{eye}_ids"),
                    return_indices=True,
                )
                old = getattr(self.anchor, f"{eye}_points")[ai]
                new = getattr(observation, f"{eye}_points")[bi]
                diagonal = np.linalg.norm(np.ptp(old, axis=0)) if len(old) else 0
                if len(ids) < 12 or np.percentile(np.linalg.norm(new - old, axis=1), 90) > max(
                    1.5, 0.015 * diagonal
                ):
                    self.reset()
                    break
        if self.anchor is None:
            self.anchor, self.started = observation, timestamp
        self.last = timestamp
        self.frames += 1
        score = min(q["sharpness"] for q in quality)
        if self.best is None or score > self.best[0]:
            self.best = (score, frame.copy(), observation, quality)
        progress = min(1.0, (timestamp - self.started) / self.seconds)
        return progress, self.best if progress >= 1 and self.frames >= 3 else None
