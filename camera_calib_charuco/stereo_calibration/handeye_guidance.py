"""Small, camera-visible wrist-board suggestions; never robot motion commands.

Goals stay fixed between captures or an explicit new suggestion. Camera visibility
is checked at the current board centre; this is not arm IK or collision checking.
"""

from functools import lru_cache

import cv2
import numpy as np

from .guidance import POSITION_RADIUS
from .handeye import (
    L20_BOARD,
    POLICY,
    _rotation_support,
    checked_transform,
    difference,
    diversity,
    transform,
)
from .handeye_information import ViewInformation

GUIDE_POLICY = {
    "version": 8,
    "automatic_capture": "valid_stationary_novel_pose; suggested_pose_is_advisory",
    "position_radius": POSITION_RADIUS,
    "position_exit_radius": POSITION_RADIUS + 0.04,
    "position_anchor": "current_board_centre_when_local_goal_selected",
    "target_angle_tolerance_deg": 3.0,
    "target_separation_deg": 9.0,
    "local_rotation_step_deg": 12.0,
    "local_translation_step_m": 0.025,
    "translation_target_tolerance_m": 0.005,
    "maximum_remaining_translation_m": 0.04,
    "view_selection": "bootstrap_independent_two_axis_folds_then_expected_stereo_information_gain",
    "information_scope": "fixed_rigid_model_and_unit_pixel_noise; not_physical_accuracy",
    "maximum_remaining_rotation_deg": 15.0,
    "maximum_viewing_obliquity_deg": 60.0,
    "already_oblique_deg": 50.0,
    "oblique_suggestion_policy": "improve_view_before_coverage; endpoint_nonworsening; full_path_visibility",
    "image_margin_px": 8.0,
    "minimum_corner_separation_px": 8.0,
    "minimum_projected_area_px2": 400.0,
    "minimum_projected_axis_ratio": 0.2,
    "visibility_scope": "both_native_fisheye_eyes; sampled_view_path; not_robot_clearance_or_IK",
    "hold_seconds": 1.0,
    "hold_timing": "paired_pose_dwell_overlaps_strict_stationary_joint_window",
    "detection": "native_scale_guidance; multiscale_fallback_and_every_saved_frame",
    "maximum_detection_gap_s": 1.5,
    "hold_translation_m": 0.003,
    "hold_rotation_deg": 1.5,
}


def rotation(vector):
    return cv2.Rodrigues(np.radians(np.asarray(vector, dtype=float)))[0]


def angle(a, b):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(a.T @ b)[0])))


def requires_review(readiness):
    """Poor prediction is not a request for ever larger wrist excursions."""
    if not readiness or readiness.get("ready"):
        return False
    if (
        readiness.get("status") in ("prior_validation", "independent_validation")
        or (readiness.get("prior_validation") or {}).get("passed") is False
    ):
        return True
    if not readiness.get("coverage", {}).get("sufficient"):
        return False
    if readiness.get("status") == "accuracy":
        return True
    folds = readiness.get("leave_one_out", [])
    return (
        readiness.get("status") == "stability"
        and bool(folds)
        and all(row.get("coverage_sufficient", True) for row in folds)
    )


@lru_cache(maxsize=1)
def board_points():
    width, height = (
        L20_BOARD.columns * L20_BOARD.square_mm / 1000,
        L20_BOARD.rows * L20_BOARD.square_mm / 1000,
    )
    corners = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]])
    # Fisheye image edges are curved: test the full outline, not only vertices.
    outline = np.concatenate(
        [
            a[None] + np.linspace(0, 1, 17, endpoint=False)[:, None] * (b - a)[None]
            for a, b in zip(corners, np.roll(corners, -1, axis=0))
        ]
    )
    inner = np.asarray(L20_BOARD.create().getChessboardCorners(), dtype=float) / 1000
    return np.r_[outline, inner], inner, np.array([width / 2, height / 2, 0])


def visibility(pose, model, face_signs=(1, 1)):
    """Check observed board geometry in native left and right fisheye images."""
    if model is None:
        return {"feasible": False, "reason": "Calibrated stereo geometry is unavailable."}
    points, inner, centre = board_points()
    right = transform(model.R, np.asarray(model.T) / 1000) @ checked_transform(pose)
    eyes = []
    for name, T, K, D, sign in (
        ("left", pose, model.K1, model.D1, face_signs[0]),
        ("right", right, model.K2, model.D2, face_signs[1]),
    ):
        T = np.asarray(T)
        camera_points = points @ T[:3, :3].T + T[:3, 3]
        camera_centre = T[:3, :3] @ centre + T[:3, 3]
        if np.min(camera_points[:, 2]) <= 0.02:
            return {
                "feasible": False,
                "reason": f"The board would be behind or too close to the {name} camera.",
            }
        facing = float(sign * np.dot(T[:3, 2], camera_centre / np.linalg.norm(camera_centre)))
        if facing <= 0:
            return {
                "feasible": False,
                "reason": f"The printed face would turn away from the {name} camera.",
            }
        obliquity = float(np.degrees(np.arccos(np.clip(facing, -1, 1))))
        if obliquity > GUIDE_POLICY["maximum_viewing_obliquity_deg"]:
            return {
                "feasible": False,
                "reason": f"The board would be too edge-on in the {name} camera.",
            }
        pixels = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, D
        )[0].reshape(-1, 2)
        margin = GUIDE_POLICY["image_margin_px"]
        if (
            not np.isfinite(pixels).all()
            or np.any(pixels < margin)
            or np.any(pixels > np.asarray(model.image_size) - margin)
        ):
            return {
                "feasible": False,
                "reason": f"The board outline would leave the {name} camera view.",
            }
        inner_pixels = pixels[-len(inner) :]
        distances = np.linalg.norm(inner_pixels[:, None] - inner_pixels[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        separation = float(distances.min())
        area = float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))))
        singular = np.linalg.svd(inner_pixels - inner_pixels.mean(axis=0), compute_uv=False)
        ratio = float(singular[1] / singular[0]) if singular[0] else 0
        if (
            separation < GUIDE_POLICY["minimum_corner_separation_px"]
            or area < GUIDE_POLICY["minimum_projected_area_px2"]
            or ratio < GUIDE_POLICY["minimum_projected_axis_ratio"]
        ):
            return {
                "feasible": False,
                "reason": f"The board would be too small or compressed in the {name} camera.",
            }
        eyes.append(
            {
                "eye": name,
                "obliquity_deg": obliquity,
                "minimum_corner_separation_px": separation,
                "area_px2": area,
                "projected_axis_ratio": ratio,
            }
        )
    return {"feasible": True, "eyes": eyes}


def _face_signs(reference, model):
    """Keep the detected printed face convention, including OpenCV board axes."""
    _, _, centre = board_points()
    right = transform(model.R, np.asarray(model.T) / 1000) @ reference
    return tuple(
        1 if T[:3, 2] @ (T[:3, :3] @ centre + T[:3, 3]) >= 0 else -1 for T in (reference, right)
    )


def _about_centre(pose, target_rotation):
    _, _, centre = board_points()
    camera_centre = pose[:3, :3] @ centre + pose[:3, 3]
    return transform(target_rotation, camera_centre - target_rotation @ centre)


def _sequential_rotations(current, target):
    """Both UI intrinsic Euler orders, sampled at every 3 degrees or less.

    Local goals are small single-axis turns; slight operator deviations can
    introduce a second component. Validate both UI orders before showing them.
    """
    delta = current.T @ target
    for order in ("YXZ", "XYZ"):
        if order == "YXZ":
            angles = [
                np.arctan2(delta[0, 2], delta[2, 2]),
                np.arcsin(np.clip(-delta[1, 2], -1, 1)),
                np.arctan2(delta[1, 0], delta[1, 1]),
            ]
        else:
            angles = [
                np.arctan2(-delta[1, 2], delta[2, 2]),
                np.arcsin(np.clip(delta[0, 2], -1, 1)),
                np.arctan2(-delta[0, 1], delta[0, 0]),
            ]
        previous = current
        for axis, radians in zip(order, angles):
            vector = np.zeros(3)
            vector["XYZ".index(axis)] = np.degrees(radians)
            for fraction in np.linspace(0, 1, max(2, int(abs(vector).max() / 3) + 2)):
                yield previous @ rotation(vector * fraction)
            previous = previous @ rotation(vector)


def feasible_path(pose, target_rotation, model, face_signs):
    initial = visibility(pose, model, face_signs)
    if not initial["feasible"]:
        return initial
    if angle(pose[:3, :3], target_rotation) > GUIDE_POLICY["maximum_remaining_rotation_deg"] + 1e-6:
        return {
            "feasible": False,
            "reason": "The saved suggestion is now too far from your current pose. Request a new suggestion.",
        }
    final = visibility(_about_centre(pose, target_rotation), model, face_signs)
    if not final["feasible"]:
        return final
    if any(
        before["obliquity_deg"] >= GUIDE_POLICY["already_oblique_deg"]
        and after["obliquity_deg"] > before["obliquity_deg"] + 0.1
        for before, after in zip(initial["eyes"], final["eyes"])
    ):
        return {
            "feasible": False,
            "reason": "This direction would make an already oblique camera view worse.",
        }
    for R in _sequential_rotations(pose[:3, :3], target_rotation):
        result = visibility(_about_centre(pose, R), model, face_signs)
        if not result["feasible"]:
            return result
        # Non-worsening is an endpoint coaching heuristic, not a demand that
        # every tiny secondary Euler correction monotonically improves the
        # view. Both complete animation paths still obey all visibility limits.
    return {
        "feasible": True,
        "eyes": initial["eyes"],
        "target_eyes": final["eyes"],
        "checked_ui_orders": ["XYZ", "YXZ"],
    }


def feasible_translation(pose, target, model, face_signs):
    """Check a fixed-orientation board translation, not an arm trajectory."""
    if (
        np.linalg.norm(target[:3, 3] - pose[:3, 3])
        > GUIDE_POLICY["maximum_remaining_translation_m"]
    ):
        return {
            "feasible": False,
            "reason": "The position suggestion is now too far away. Request a new suggestion.",
        }
    if angle(pose[:3, :3], target[:3, :3]) > GUIDE_POLICY["target_angle_tolerance_deg"]:
        return {
            "feasible": False,
            "reason": "The board tilt changed during the position suggestion. Request a new suggestion.",
        }
    initial, final = visibility(pose, model, face_signs), visibility(target, model, face_signs)
    if not initial["feasible"]:
        return initial
    if not final["feasible"]:
        return final
    if any(
        a["obliquity_deg"] >= GUIDE_POLICY["already_oblique_deg"]
        and b["obliquity_deg"] > a["obliquity_deg"] + 0.1
        for a, b in zip(initial["eyes"], final["eyes"])
    ):
        return {"feasible": False, "reason": "This move would worsen an oblique view."}
    for fraction in np.linspace(0, 1, 9):
        point = pose.copy()
        point[:3, 3] += fraction * (target[:3, 3] - pose[:3, 3])
        check = visibility(point, model, face_signs)
        if not check["feasible"]:
            return check
    return {
        "feasible": True,
        "eyes": initial["eyes"],
        "target_eyes": final["eyes"],
        "checked_view_path": "straight_translation; fixed_orientation",
    }


def _summary(samples, current_camera_pose, stage):
    active = [s for s in samples if not s.get("excluded")]
    training = [s for s in active if s["role"] == "training"]
    role = "check" if stage in ("checking", "complete") else "training"
    phase = [s for s in active if s["role"] == role]
    reference = np.asarray(training[0]["T_camera_board"]) if training else None
    current = None if current_camera_pose is None else checked_transform(current_camera_pose)
    relative = (
        None
        if current is None
        else (np.eye(3) if reference is None else reference[:3, :3].T @ current[:3, :3])
    )
    vectors = (
        [
            np.degrees(
                cv2.Rodrigues(reference[:3, :3].T @ np.asarray(s["T_camera_board"])[:3, :3])[
                    0
                ].ravel()
            )
            for s in phase
        ]
        if reference is not None
        else []
    )
    return (
        active,
        training,
        phase,
        reference,
        current,
        {
            "role": role,
            "count": len(phase),
            "diversity": diversity([s["T_base_tool"] for s in phase]),
            "policy": dict(GUIDE_POLICY),
            "at_target": False,
            "current_rotation": None if relative is None else relative.tolist(),
            "target_rotation": None,
            "target_vector_deg": None,
            "angle_error_deg": None,
            "axis_span_deg": np.ptp(vectors, axis=0).tolist()
            if len(vectors) >= 2
            else [0.0, 0.0, 0.0],
        },
    )


def _validation_support(poses):
    """Coverage must survive removing a whole orientation, as in validation.

    In a rigid setup camera and tool relative rotations differ only by a
    constant change of coordinates, so this uses observed board poses without
    assuming an already accurate hand-eye solution.
    """
    poses, groups, _, _ = _rotation_support(poses)
    folds = [diversity([p for i, p in enumerate(poses) if i not in group]) for group in groups]
    required = np.degrees(POLICY["second_rotation_singular_value_rad"])
    return {
        "sufficient": len(groups) >= 4 and all(f["sufficient"] for f in folds),
        "score": (
            min((min(f["second_axis_rms_deg"] / required, 1) for f in folds), default=0.0),
            min(
                (
                    min(f["rotation_span_deg"] / POLICY["minimum_rotation_span_deg"], 1)
                    for f in folds
                ),
                default=0.0,
            ),
        ),
    }


class LocalGuidance:
    """One camera-checked goal per capture/explicit suggestion, never a chase."""

    def __init__(self):
        self.token = self.target = self.position = None
        self.translation_target = None
        self.selection = None
        self.serial = 0
        self.reason = None

    def guide(
        self,
        samples,
        current_camera_pose,
        stage,
        skip=0,
        *,
        model=None,
        readiness=None,
        signature=None,
    ):
        active, training, phase, reference, current, result = _summary(
            samples, current_camera_pose, stage
        )
        if stage == "complete":
            return {
                **result,
                "key": "done",
                "title": "Calibration verified",
                "instruction": "Coverage and independent prediction checks passed. Collection is complete.",
            }
        if requires_review(readiness):
            return {
                **result,
                "key": "review",
                "title": "Review prediction accuracy",
                "instruction": readiness["reason"]
                + " Keep a comfortable visible pose; larger tilts will not resolve this inconsistency.",
            }
        if not training:
            return {
                **result,
                "key": "reference",
                "title": "Start with a comfortable pose",
                "instruction": "Show the complete board to both cameras, then hold still.",
                "target_rotation": np.eye(3).tolist(),
                "target_vector_deg": [0, 0, 0],
                "at_target": current is not None,
                "angle_error_deg": 0 if current is not None else None,
            }
        token = (stage, skip, tuple(s["id"] for s in active))
        if token != self.token:
            self.token, self.target, self.position, self.reason = token, None, None, None
            self.translation_target = self.selection = None
            self.serial += 1
        if model is None or current is None:
            return {
                **result,
                "key": "unavailable",
                "title": "Waiting for stereo board geometry",
                "instruction": "Show the board clearly in both cameras before requesting a local suggestion.",
            }
        face_signs = _face_signs(reference, model)
        if self.target is None and self.reason is None:
            saved = [np.asarray(s["T_camera_board"])[:3, :3] for s in active]
            available = []
            phase_poses = [s["T_camera_board"] for s in phase]
            spread_before = diversity(phase_poses)
            support_before = _validation_support(phase_poses)
            information = ViewInformation(phase_poses, model)
            for axis in (1, 0, 2):
                for sign in (1, -1):
                    vector = np.zeros(3)
                    vector[axis] = sign * GUIDE_POLICY["local_rotation_step_deg"]
                    target = current[:3, :3] @ rotation(vector)
                    if any(angle(target, R) < GUIDE_POLICY["target_separation_deg"] for R in saved):
                        continue
                    feasibility = feasible_path(current, target, model, face_signs)
                    if not feasibility["feasible"]:
                        continue
                    improvements = [
                        before["obliquity_deg"] - after["obliquity_deg"]
                        for before, after in zip(feasibility["eyes"], feasibility["target_eyes"])
                        if before["obliquity_deg"] >= GUIDE_POLICY["already_oblique_deg"]
                    ]
                    view_improvement = min(improvements, default=0.0)
                    # A roll may add angular coverage while leaving an oblique
                    # view unchanged and fragile to corner-estimation jitter.
                    # Prefer an actual clearer view before asking for coverage.
                    if improvements and view_improvement <= 0.1:
                        continue
                    poses = [s["T_camera_board"] for s in phase] + [_about_centre(current, target)]
                    spread = diversity(poses)
                    second = spread["second_axis_rms_deg"]
                    required = np.degrees(POLICY["second_rotation_singular_value_rad"])
                    score = (
                        view_improvement,
                        min(second / required, 1),
                        min(spread["rotation_span_deg"] / POLICY["minimum_rotation_span_deg"], 1),
                        *(_validation_support(poses)["score"] if stage == "training" else (1, 1)),
                        information.gain(_about_centre(current, target)),
                        min(angle(target, R) for R in saved),
                    )
                    benefit = (
                        "Adds a less-sampled tilt direction for two-axis calibration."
                        if not spread_before["sufficient"]
                        else "Adds an independent tilt so validation can leave one orientation out."
                        if stage == "training" and not support_before["sufficient"]
                        else "Chosen to improve the information in the stereo calibration views."
                    )
                    available.append((score, target, None, benefit))
            # Pure translations cannot supply missing rotational observability.
            # Introduce them only after independent two-axis coverage survives
            # held-out orientation groups; otherwise translations can stall
            # validation while repeatedly observing the same essential tilt.
            if spread_before["sufficient"] and (
                stage != "training" or support_before["sufficient"]
            ):
                for axis in range(3):
                    for sign in (1, -1):
                        candidate = current.copy()
                        candidate[axis, 3] += sign * GUIDE_POLICY["local_translation_step_m"]
                        if any(
                            difference(candidate, s["T_camera_board"])[0]
                            < POLICY["pose_novelty_translation_m"]
                            and angle(candidate[:3, :3], np.asarray(s["T_camera_board"])[:3, :3])
                            < POLICY["pose_novelty_rotation_deg"]
                            for s in active
                        ):
                            continue
                        feasible = feasible_translation(current, candidate, model, face_signs)
                        if not feasible["feasible"]:
                            continue
                        improvements = [
                            a["obliquity_deg"] - b["obliquity_deg"]
                            for a, b in zip(feasible["eyes"], feasible["target_eyes"])
                            if a["obliquity_deg"] >= GUIDE_POLICY["already_oblique_deg"]
                        ]
                        improvement = min(improvements, default=0.0)
                        if improvements and improvement <= 0.1:
                            continue
                        gain = information.gain(candidate)
                        score = (improvement, 1, 1, 1, 1, gain, 0)
                        benefit = (
                            "Adds a different viewing distance after establishing two-axis tilt coverage."
                            if axis == 2
                            else "Adds a different board position after establishing two-axis tilt coverage."
                        )
                        available.append((score, candidate[:3, :3], candidate, benefit))
            available.sort(key=lambda row: row[0], reverse=True)
            if available:
                chosen = available[skip % len(available)]
                self.target, self.translation_target = chosen[1:3]
                self.selection = {
                    "method": GUIDE_POLICY["view_selection"],
                    "expected_information_gain_nats": chosen[0][-2],
                    "candidate_count": len(available),
                    "reason": chosen[3],
                    "robot_path_verified": False,
                }
                self.position = [signature["x"], signature["y"]] if signature else None
                if self.translation_target is not None:
                    p = (
                        self.translation_target[:3, :3] @ board_points()[2]
                        + self.translation_target[:3, 3]
                    )
                    pixel = cv2.fisheye.projectPoints(
                        p.reshape(1, 1, 3), np.zeros(3), np.zeros(3), model.K1, model.D1
                    )[0].ravel()
                    self.position = (pixel / np.asarray(model.image_size)).tolist()
            else:
                self.reason = "No useful small tilt keeps the whole board clear in both views. Keep a comfortable visible pose; reposition gently, then request a new suggestion."
        if self.target is None:
            return {
                **result,
                "key": "unavailable",
                "title": "No suitable local suggestion",
                "instruction": self.reason,
            }
        feasible = (
            feasible_translation(current, self.translation_target, model, face_signs)
            if self.translation_target is not None
            else feasible_path(current, self.target, model, face_signs)
        )
        if not feasible["feasible"]:
            return {
                **result,
                "key": "unavailable",
                "title": "Suggestion paused",
                "instruction": feasible["reason"],
            }
        target = reference[:3, :3].T @ self.target
        error = angle(current[:3, :3], self.target)
        result = {
            **result,
            "key": f"local-{self.serial}",
            "title": "Small local tilt",
            "instruction": "Follow this small board rotation while keeping the full board visible in both cameras.",
            "target_rotation": target.tolist(),
            "target_vector_deg": np.degrees(cv2.Rodrigues(target)[0].ravel()).tolist(),
            "angle_error_deg": error,
            "at_target": error <= GUIDE_POLICY["target_angle_tolerance_deg"],
            "position_target": self.position,
            "visibility": feasible,
            "selection": self.selection,
            "selection_reason": self.selection["reason"],
        }
        if self.translation_target is not None:
            translation_error = self.translation_target[:3, 3] - current[:3, 3]
            result.update(
                translation_target_camera_m=self.translation_target[:3, 3].tolist(),
                translation_error_camera_m=translation_error.tolist(),
                target_camera_pose=self.translation_target.tolist(),
                at_target=bool(
                    result["at_target"]
                    and np.linalg.norm(translation_error)
                    <= GUIDE_POLICY["translation_target_tolerance_m"]
                ),
            )
        return result


def guidance(samples, current_camera_pose, stage, skip=0, *, model=None, readiness=None):
    """One-shot offline helper; live runtime retains a LocalGuidance instance."""
    return LocalGuidance().guide(
        samples, current_camera_pose, stage, skip, model=model, readiness=readiness
    )


def pose_preview(samples, guide, signature, current_camera_pose=None):
    """Camera-view illustration, deliberately separate from capture readiness.

    The left-image pose can explain a movement even while stereo/arm quality
    checks block capture. Never feed it back into ``at_target`` or GuidedHold.
    """
    if not signature or guide["key"] in ("done", "manual", "review", "unavailable"):
        return None
    current = signature.get("rotation_camera")
    if current is None:
        return None
    current = np.asarray(current, dtype=float)
    if current.shape != (3, 3) or not np.all(np.isfinite(current)):
        return None
    first = next((s for s in samples if s["role"] == "training" and not s.get("excluded")), None)
    target = current
    if first and guide.get("target_rotation") is not None:
        reference = checked_transform(first["T_camera_board"])[:3, :3]
        target = reference @ np.asarray(guide["target_rotation"])
    view = np.eye(3)
    if current_camera_pose is not None:
        pose = checked_transform(current_camera_pose)
        _, _, centre = board_points()
        ray = pose[:3, :3] @ centre + pose[:3, 3]
        ray /= np.linalg.norm(ray)
        right = np.cross([0, 1, 0], ray)
        if np.linalg.norm(right) < 1e-8:
            return None
        right /= np.linalg.norm(right)
        view = np.column_stack((right, np.cross(ray, right), ray))
    return {
        "current_rotation": (view.T @ current).tolist(),
        "target_rotation": (view.T @ target).tolist(),
        "view_frame_camera": view.tolist(),
        "source": "left_camera_board_view",
        "scope": "Board-centred viewing-ray schematic; native fisheye projection remains in the camera view.",
    }


class PositionThenTilt:
    """Stereo's move/printed-edge cues; hysteresis prevents boundary flashing.

    The blue area remains around the operator's first captured position. It does
    not ask a wrist-mounted board to visit the full stereo lens-coverage grid.
    Missing detections preserve the illustration, but never qualify for capture.
    """

    def __init__(self):
        self.key = None
        self.positioned = False
        self.motion = None
        self.axis = None

    def apply(self, guide, signature, target=(0.5, 0.5)):
        g = dict(guide)
        if g["key"] != self.key:
            self.key, self.positioned = g["key"], False
            self.motion = self.axis = None
        if g["key"] in ("done", "manual", "review", "unavailable"):
            return {**g, "adjustment": {"kind": "unavailable"}}
        if g.get("translation_error_camera_m") is not None:
            delta = np.asarray(g["translation_error_camera_m"])
            tolerance = GUIDE_POLICY["translation_target_tolerance_m"]
            g["position_target"], g["position_radius"] = list(target), 0.025
            # Metric error drives the cue; a large image circle cannot declare
            # a 25 mm depth/position request complete without actually moving.
            if g["at_target"]:
                return {
                    **g,
                    "title": "Hold still",
                    "instruction": "New position reached. Hold steady.",
                    "adjustment": {"kind": "ready"},
                }
            if abs(delta[2]) > tolerance:
                direction = "farther" if delta[2] > 0 else "closer"
                return {
                    **g,
                    "title": f"Move the board {direction}",
                    "instruction": f"Move the whole board about {abs(delta[2]) * 1000:.0f} mm {direction}. Keep the same tilt and the full board visible.",
                    "adjustment": {"kind": "distance", "direction": direction},
                }
            return {
                **g,
                "title": "Move to the new position",
                "instruction": f"Shift the whole board about {np.linalg.norm(delta[:2]) * 1000:.0f} mm toward the blue dot. Keep the same tilt.",
                "adjustment": {
                    "kind": "move",
                    "target": list(target),
                    "current": [signature["x"], signature["y"]] if signature else None,
                },
            }
        current = [signature["x"], signature["y"]] if signature else None
        distance = None if current is None else float(np.linalg.norm(np.subtract(current, target)))
        if distance is not None:
            radius = GUIDE_POLICY["position_exit_radius"] if self.positioned else POSITION_RADIUS
            self.positioned = distance <= radius
        position_valid = self.positioned and distance is not None
        g["at_target"] = g["at_target"] and position_valid
        g["position_target"] = list(target)
        g["position_radius"] = POSITION_RADIUS
        if not self.positioned:
            g.update(
                title="Move into the blue area",
                instruction="Move the board centre into the blue area in the left view. Keep the full board visible in both cameras.",
                adjustment={"kind": "move", "target": list(target), "current": current},
            )
            return g
        if g["key"] == "reference":
            g.update(
                title="Hold your first pose",
                instruction="Keep the board still. This becomes the orientation reference.",
            )
            self.motion = {"kind": "ready"}
        elif g["current_rotation"] is not None:
            # Correct from the CURRENT printed axes, not an animation that always
            # assumes the wrist has returned to the first captured orientation.
            delta = np.degrees(
                cv2.Rodrigues(
                    np.asarray(g["current_rotation"]).T @ np.asarray(g["target_rotation"])
                )[0].ravel()
            )
            largest = int(np.argmax(np.abs(delta)))
            if self.axis is None or abs(delta[largest]) > abs(delta[self.axis]) * 1.4 + 2:
                self.axis = largest
            if not g["at_target"]:
                value = delta[self.axis]
                if self.axis == 2:
                    self.motion = {
                        "kind": "roll",
                        "direction": "clockwise" if value > 0 else "counterclockwise",
                    }
                else:
                    edges = (("top", "bottom"), ("right", "left"))
                    self.motion = {"kind": "tilt", "edge": edges[self.axis][0 if value > 0 else 1]}
        motion = self.motion or {"kind": "unavailable"}
        g["adjustment"] = motion
        if motion["kind"] == "tilt":
            edge = motion["edge"]
            g.update(
                title=f"Tilt {edge} edge",
                instruction=f"Bring the printed {edge} edge toward the cameras. Keep the board in this area.",
            )
        elif motion["kind"] == "roll":
            direction = motion["direction"]
            g.update(
                title=f"Rotate {direction}",
                instruction="Turn the board in its own plane, as viewed by the cameras. Keep its centre in this area.",
            )
        if g["at_target"]:
            g.update(
                title="Hold still",
                instruction="Position and tilt reached. Hold steady for capture.",
            )
        return g


class GuidedHold:
    """Fresh, qualifying detections of one stationary pose; polling cannot advance it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.started = self.last = self.pose = self.key = None
        self.progress = 0.0

    def update(self, timestamp, key, eligible, pose):
        if not eligible or pose is None:
            self.reset()
            return 0.0
        if self.last is not None and timestamp <= self.last:
            self.reset()
            return 0.0
        restart = self.last is None or key != self.key
        if not restart:
            translation, rotation_deg = difference(self.pose, pose)
            restart = (
                timestamp - self.last > GUIDE_POLICY["maximum_detection_gap_s"]
                or translation > GUIDE_POLICY["hold_translation_m"]
                or rotation_deg > GUIDE_POLICY["hold_rotation_deg"]
            )
        if restart:
            self.started, self.key, self.pose = timestamp, key, checked_transform(pose).copy()
        self.last = timestamp
        self.progress = min(1.0, (timestamp - self.started) / GUIDE_POLICY["hold_seconds"])
        return self.progress
