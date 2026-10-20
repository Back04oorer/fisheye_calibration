"""Fixed-camera / wrist-board calibration. T_a_b maps coordinates from b into a.

All public transforms use metres, camera observations use native pixels, and
OpenCV board geometry uses mm. No robot or camera ownership belongs here.
"""

import hashlib
import heapq
import itertools
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares

from .config import BoardConfig
from .observations import usable
from .validation import board_pose, view_metrics
from .handeye_kinematics import calibrated_joint_angle, decode_arm_model
from .handeye_configuration import ReferenceGravityCorrection
from .handeye_refinement import refinement_delta

L20_BOARD = BoardConfig(4, 5, 25.0, 20.0, "DICT_4X4_50", False, tuple(range(20, 30)))
MIN_TRAIN = 3
MIN_CHECK = 3
COVERAGE_METHOD = "complete_link_rotation_support_v1"
FIT_METHOD = "fixed_stereo_native_reprojection_v1"
CROSS_VALIDATION_METHOD = "observable_distinct_pose_folds_v1"
# Operational capture/conditioning policies, not claims of physical accuracy.
POLICY = {
    "minimum_training_poses": MIN_TRAIN,
    "minimum_check_poses": MIN_CHECK,
    "minimum_cross_validation_poses": MIN_TRAIN + 1,
    "native_reprojection_rms_px": 2.0,
    "capture_stereo_rms_px": 3.0,
    "check_reprojection_rms_px": 3.0,
    "pose_novelty_translation_m": 0.015,
    "pose_novelty_rotation_deg": 8.0,
    "minimum_rotation_span_deg": 25.0,
    "second_rotation_singular_value_rad": 0.15,
    "feedback_max_age_s": 0.25,
    "pair_max_skew_s": 0.08,
    "stationary_window_s": 0.7,
    "stationary_joint_range_rad": 0.006,
    "camera_max_age_s": 0.25,
}


class CaptureBlocked(ValueError):
    """A failed quality gate with diagnostics, including on rejected frames."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


def transform(R, t):
    result = np.eye(4)
    result[:3, :3] = R
    result[:3, 3] = np.asarray(t).reshape(3)
    return checked_transform(result)


def checked_transform(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("Expected a finite, proper rigid transform in metres.")
    return value


def inverse(T):
    T = checked_transform(T)
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def difference(a, b):
    a, b = checked_transform(a), checked_transform(b)
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0]))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(angle)


def novel(pose, previous):
    return all(
        t >= POLICY["pose_novelty_translation_m"] or r >= POLICY["pose_novelty_rotation_deg"]
        for t, r in (difference(pose, p) for p in previous)
    )


def _rotation_support(poses):
    """Deterministic complete-link orientation support, never sample weights.

    Use the existing rotation-novelty scale, not floating-point equality, to
    group measurement jitter and repeated views. Complete-link bounds each
    cluster's full diameter, preventing a chain of small movements from merging
    a broad sweep. A minimax medoid depends on geometric extent, not the number
    of duplicate observations. All observations remain in residual evaluation.
    """
    poses = [checked_transform(p) for p in poses]
    by_rotation = {}
    for index, pose in enumerate(poses):
        by_rotation.setdefault(tuple(pose[:3, :3].ravel()), []).append(index)
    keys = sorted(by_rotation)
    if not keys:
        return poses, [], [], 0
    rotations = np.asarray(keys).reshape(-1, 3, 3)
    traces = np.einsum("aij,bij->ab", rotations, rotations)
    distances = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))
    np.fill_diagonal(distances, 0)
    clusters = {(i,) for i in range(len(keys))}
    candidates = [
        (float(distances[i, j]), (i,), (j,)) for i, j in itertools.combinations(range(len(keys)), 2)
    ]
    heapq.heapify(candidates)
    limit = POLICY["pose_novelty_rotation_deg"]
    while candidates:
        distance, a, b = heapq.heappop(candidates)
        if a not in clusters or b not in clusters:
            continue
        if distance > limit:
            break
        clusters.remove(a)
        clusters.remove(b)
        merged = tuple(sorted(a + b))
        for other in clusters:
            pair = sorted((merged, other))
            value = float(distances[np.ix_(merged, other)].max())
            heapq.heappush(candidates, (value, *pair))
        clusters.add(merged)
    groups, representatives = [], []
    for cluster in sorted(clusters):
        medoid = min(cluster, key=lambda i: (float(distances[i, list(cluster)].max()), keys[i]))
        groups.append(sorted(index for i in cluster for index in by_rotation[keys[i]]))
        representatives.append(poses[by_rotation[keys[medoid]][0]])
    return poses, groups, representatives, len(keys)


def diversity(poses):
    """Equal-weight independent orientation support, plus translation extent.

    Singular values describe RMS pairwise rotation excitation in the base
    frame. Their 0.15 rad floor is an operational conditioning rule, not a
    physical accuracy guarantee. Repeated/noisy views share an orientation
    support cluster and cannot amplify or dilute another tilt by their count.
    """
    poses, groups, representatives, unique_count = _rotation_support(poses)
    rotations = [
        cv2.Rodrigues(b[:3, :3] @ a[:3, :3].T)[0].ravel()
        for a, b in itertools.combinations(representatives, 2)
    ]
    sv = (
        np.linalg.svd(rotations, compute_uv=False) / np.sqrt(len(rotations))
        if rotations
        else np.array([])
    )
    span = max((np.degrees(np.linalg.norm(r)) for r in rotations), default=0.0)
    translation_span = max(
        (np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in itertools.combinations(poses, 2)),
        default=0.0,
    )
    sufficient = (
        len(groups) >= MIN_TRAIN
        and span >= POLICY["minimum_rotation_span_deg"]
        and len(sv) >= 2
        and sv[1] >= POLICY["second_rotation_singular_value_rad"]
    )
    return {
        "sufficient": bool(sufficient),
        "rotation_span_deg": float(span),
        "rotation_singular_values_rad": sv.tolist(),
        "rotation_rms_singular_values_rad": sv.tolist(),
        "rotation_singular_values_normalized": True,
        "coverage_method": COVERAGE_METHOD,
        "rotation_support_count": len(groups),
        "rotation_support_diameter_deg": POLICY["pose_novelty_rotation_deg"],
        "rotation_support_sizes": [len(group) for group in groups],
        "rotation_conditioning_ratio": float(sv[1] / sv[0]) if len(sv) >= 2 and sv[0] else 0.0,
        "second_axis_rms_deg": float(np.degrees(sv[1])) if len(sv) >= 2 else 0.0,
        "translation_span_m": float(translation_span),
        "unique_rotation_count": unique_count,
        "sample_count": len(poses),
    }


def camera_board_pose(observation, model, board=L20_BOARD):
    if tuple(observation.image_size) != tuple(model.image_size):
        raise ValueError("Native camera resolution differs from the frozen stereo calibration.")
    ok, reason = usable(observation, board, minimum=(board.columns - 1) * (board.rows - 1))
    if not ok:
        raise ValueError(reason)
    metrics = view_metrics(observation, model, board)
    left, right = (metrics[f"{eye}_reprojection_rms_px"] for eye in ("left", "right"))
    if (
        max(metrics[f"{eye}_reprojection_rms_px"] for eye in ("left", "right"))
        > POLICY["native_reprojection_rms_px"]
    ):
        raise CaptureBlocked(
            "reprojection",
            "Board reprojection is unstable. "
            f"Left {left:.2f} px / right {right:.2f} px; "
            f"limit {POLICY['native_reprojection_rms_px']:.1f} px. Improve sharpness or visibility.",
            reprojection_px=[left, right],
        )
    if (
        max(metrics[f"{d}_rms_px"] for d in ("left_to_right", "right_to_left"))
        > POLICY["capture_stereo_rms_px"]
    ):
        lr, rl = (
            metrics[f"{direction}_rms_px"] for direction in ("left_to_right", "right_to_left")
        )
        raise CaptureBlocked(
            "stereo",
            "The two camera views disagree. "
            f"L→R {lr:.2f} px / R→L {rl:.2f} px; "
            f"limit {POLICY['capture_stereo_rms_px']:.1f} px. Check board flatness and stereo calibration.",
            reprojection_px=[left, right],
            stereo_reprojection_px=[lr, rl],
        )
    obj = board.create().getChessboardCorners()[observation.left_ids].astype(float)
    r, t = board_pose(obj, observation.left_points, model.K1, model.D1)
    return transform(cv2.Rodrigues(r)[0], t / 1000), metrics


def solve(samples, model=None):
    """Park seed, optionally refined against both native stereo images.

    G_i = T_base_tool, C_i = T_camera_board. Solve X = T_base_camera,
    then Z = T_tool_board from G_i Z = X C_i. Check samples are NEVER fitted.
    """
    train = [s for s in samples if s["role"] == "training" and not s.get("excluded")]
    return _solve_batches([train], model=model)[0]


def _solve_batches(groups, model=None, *, cross_validation=False):
    """One isolated solver startup for a fit and all cross-validation folds."""
    jobs = []
    for index, train in enumerate(groups):
        jobs.append(_solver_job(train, require_capture_coverage=not (cross_validation and index)))
    if hasattr(cv2, "calibrateHandEye"):
        results = []
        for B, C, _spread in jobs:
            r, t = cv2.calibrateHandEye(
                [b[:3, :3] for b in B],
                [b[:3, 3] for b in B],
                [c[:3, :3] for c in C],
                [c[:3, 3] for c in C],
                method=cv2.CALIB_HAND_EYE_PARK,
            )
            results.append({"R": r, "t": t, "opencv": cv2.__version__})
    else:
        # Some OpenCV 5 wheels omit these bindings. Keep the camera environment
        # unchanged and call the pinned official 4.x solver in its own process.
        python = Path(__file__).resolve().parents[1] / "envs/handeye-opencv4/bin/python"
        if not python.is_file():
            raise ValueError("Install the isolated hand-eye solver; see README Hand–eye mode.")
        worker = Path(__file__).with_name("handeye_solver.py")
        completed = subprocess.run(
            [str(python), str(worker)],
            input=json.dumps(
                {
                    "batches": [
                        {"B": [b.tolist() for b in B], "C": [c.tolist() for c in C]}
                        for B, C, _spread in jobs
                    ]
                }
            ),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            raise ValueError("Hand-eye solver failed: " + completed.stderr[-1000:])
        results = json.loads(completed.stdout)["results"]
    seeds = [_solution(train, job, result) for train, job, result in zip(groups, jobs, results)]
    if model is None:
        return seeds
    return [_refine_solution(train, seed, model) for train, seed in zip(groups, seeds)]


def _refine_solution(train, seed, model):
    """Jointly fit X and Z to every training corner in both raw camera images.

    G_i Z = X C_i; predict C_i = inv(X) G_i Z. Only X and Z change:
    intrinsics, distortion, stereo extrinsics, board size and measured FK stay
    frozen. Linear least squares gives every corner the same weight. Callers
    construct each cross-validation fold before initialization/refinement.
    This is the fixed-camera analogue of Tabb & Ahmad Yousef's fixed-intrinsic
    reprojection objective (2017, doi:10.1007/s00138-017-0841-7).
    """
    objects = L20_BOARD.create().getChessboardCorners().astype(float) / 1000
    rig_R, rig_t = model.R, np.asarray(model.T).reshape(3) / 1000
    records = []
    for sample in train:
        obs = sample["observation"]
        if tuple(obs["image_size"]) != tuple(model.image_size):
            raise ValueError("Training observation resolution differs from frozen stereo geometry.")
        eyes = []
        for eye, K, D in (("left", model.K1, model.D1), ("right", model.K2, model.D2)):
            ids = np.asarray(obs[f"{eye}_ids"], dtype=int)
            pixels = np.asarray(obs[f"{eye}_points"], dtype=float)
            if (
                ids.ndim != 1
                or len(ids) < 4
                or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0)
                or np.any(ids >= len(objects))
                or pixels.shape != (len(ids), 2)
                or not np.isfinite(pixels).all()
            ):
                raise ValueError("Invalid training corner correspondences.")
            eyes.append((objects[ids], pixels, K, D))
        records.append((checked_transform(sample["T_base_tool"]), eyes))

    X0, Z0 = (checked_transform(seed[key]) for key in ("T_base_camera_left_raw", "T_tool_board"))

    def unpack(parameters):
        transforms = []
        for values, initial in ((parameters[:6], X0), (parameters[6:], Z0)):
            delta = np.eye(4)
            delta[:3, :3] = cv2.Rodrigues(values[:3])[0]
            delta[:3, 3] = values[3:]
            transforms.append(delta @ initial)
        return transforms

    def residuals(parameters, *, require_front=False):
        X, Z = unpack(parameters)
        invX = np.eye(4)
        invX[:3, :3], invX[:3, 3] = X[:3, :3].T, -X[:3, :3].T @ X[:3, 3]
        errors = []
        for G, eyes in records:
            C = invX @ G @ Z
            for eye, (obj, pixels, K, D) in enumerate(eyes):
                camera = obj @ C[:3, :3].T + C[:3, 3]
                if eye:
                    camera = camera @ rig_R.T + rig_t
                if require_front and np.min(camera[:, 2]) <= 0:
                    raise ValueError("Refined board lies behind a camera.")
                projected = cv2.fisheye.projectPoints(
                    camera.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D
                )[0].reshape(-1, 2)
                errors.append((projected - pixels).ravel())
        result = np.concatenate(errors)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite stereo hand-eye refinement residuals.")
        return result

    start = np.zeros(12)
    initial_error = residuals(start, require_front=True)
    fit = least_squares(
        residuals,
        start,
        loss="linear",
        x_scale="jac",
        max_nfev=150,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-8,
    )
    final_error = residuals(fit.x, require_front=True)
    if not fit.success or final_error @ final_error > initial_error @ initial_error + 1e-8:
        raise ValueError("Stereo hand-eye refinement did not converge to a lower residual.")
    X, Z = (checked_transform(T) for T in unpack(fit.x))
    pose_errors = [
        difference(np.asarray(s["T_base_tool"]) @ Z, X @ np.asarray(s["T_camera_board"]))
        for s in train
    ]
    return {
        **seed,
        "method": "OpenCV PARK seed; joint fixed-stereo native-pixel least squares",
        "fit_method": FIT_METHOD,
        "T_base_camera_left_raw": X.tolist(),
        "T_tool_board": Z.tolist(),
        "training_translation_rms_mm": float(
            np.sqrt(np.mean([e[0] ** 2 for e in pose_errors])) * 1000
        ),
        "training_rotation_rms_deg": float(np.sqrt(np.mean([e[1] ** 2 for e in pose_errors]))),
        "refinement": {
            "scipy_version": scipy.__version__,
            "loss": "linear",
            "parameters": 12,
            "fitted": ["T_base_camera_left_raw", "T_tool_board"],
            "initial_rms_px": float(np.sqrt(np.mean(initial_error.reshape(-1, 2) ** 2) * 2)),
            "final_rms_px": float(np.sqrt(np.mean(final_error.reshape(-1, 2) ** 2) * 2)),
            "corner_count": len(final_error) // 2,
            "evaluations": int(fit.nfev),
        },
    }


def motion_consistency(samples):
    """Rotation angle is invariant under conjugation; no fitted X/Z is needed.

    This diagnostic cannot distinguish mount flex, kinematic error and camera
    pose bias and therefore does not invent a separate acceptance threshold.
    """
    pairs = []
    for a, b in itertools.combinations(samples, 2):
        robot = difference(a["T_base_tool"], b["T_base_tool"])[1]
        camera = difference(a["T_camera_board"], b["T_camera_board"])[1]
        pairs.append(
            {
                "ids": [a["id"], b["id"]],
                "robot_rotation_deg": robot,
                "board_rotation_deg": camera,
                "difference_deg": abs(camera - robot),
            }
        )
    pairs.sort(key=lambda row: row["difference_deg"], reverse=True)
    return {
        "max_rotation_disagreement_deg": pairs[0]["difference_deg"] if pairs else 0.0,
        "worst_pairs": pairs[:5],
        "pair_count": len(pairs),
        "interpretation": "Diagnostic only; check mounting, robot kinematics and camera pose bias.",
    }


def _solver_job(train, *, require_capture_coverage=True):
    if len(train) < MIN_TRAIN:
        raise ValueError(
            f"At least {MIN_TRAIN} distinct poses are needed to observe two rotation axes."
        )
    G = [checked_transform(s["T_base_tool"]) for s in train]
    C = [checked_transform(s["T_camera_board"]) for s in train]
    spread = diversity(G)
    if require_capture_coverage and not spread["sufficient"]:
        raise ValueError(
            "More wrist rotation is needed about two different axes; translations alone cannot solve hand–eye."
        )
    if not require_capture_coverage:
        # OpenCV needs at least two non-parallel rotation axes. The full run
        # retains the stricter capture-coverage policy; a withheld fold should
        # be judged by its predictions, not required to reproduce the complete
        # acquisition envelope. Reject numerically near-parallel folds before
        # Park's solve. Whole orientation clusters still leave together.
        values = spread["rotation_rms_singular_values_rad"]
        if (
            spread["rotation_support_count"] < MIN_TRAIN
            or len(values) < 2
            or values[1] <= max(1e-6, values[0] * 1e-3)
        ):
            raise ValueError("Withheld fold needs at least two non-parallel rotation axes.")
    B = [inverse(g) for g in G]
    return B, C, spread


def _solution(train, job, result):
    B, C, spread = job
    G = [inverse(b) for b in B]
    X = transform(result["R"], result["t"])
    mounts = [inverse(g) @ X @ c for g, c in zip(G, C)]
    u, _, vt = np.linalg.svd(sum(m[:3, :3] for m in mounts))
    R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    Z = transform(R, np.mean([m[:3, 3] for m in mounts], axis=0))
    residuals = [difference(g @ Z, X @ c) for g, c in zip(G, C)]
    return {
        "method": "OpenCV calibrateHandEye PARK; inverse follower poses (eye-to-hand)",
        "solver_opencv_version": result["opencv"],
        "length_unit": "m",
        "convention": "T_a_b maps b coordinates into a",
        "T_base_camera_left_raw": X.tolist(),
        "T_tool_board": Z.tolist(),
        "training_ids": [s["id"] for s in train],
        "diversity": spread,
        "coverage_method": COVERAGE_METHOD,
        "cross_validation_method": CROSS_VALIDATION_METHOD,
        "training_translation_rms_mm": float(
            np.sqrt(np.mean(np.square([r[0] * 1000 for r in residuals])))
        ),
        "training_rotation_rms_deg": float(np.sqrt(np.mean(np.square([r[1] for r in residuals])))),
        "status": "needs_independent_checks",
        "physical_accuracy_verified": False,
    }


def check_sample(sample, solution, model, board=L20_BOARD):
    X, Z = (
        checked_transform(solution["T_base_camera_left_raw"]),
        checked_transform(solution["T_tool_board"]),
    )
    predicted = inverse(X) @ checked_transform(sample["T_base_tool"]) @ Z
    observed = checked_transform(sample["T_camera_board"])
    t, r = difference(predicted, observed)
    stereo = transform(model.R, model.T / 1000)
    corners = board.create().getChessboardCorners().astype(float)
    errors = {}
    for eye, K, D, pose in (
        ("left", model.K1, model.D1, predicted),
        ("right", model.K2, model.D2, stereo @ predicted),
    ):
        obs = sample["observation"]
        obj = corners[np.asarray(obs[f"{eye}_ids"], dtype=int)]
        if ((pose[:3, :3] @ (obj / 1000).T).T + pose[:3, 3])[:, 2].min() <= 0:
            raise ValueError("Predicted board lies behind a camera.")
        pixels = cv2.fisheye.projectPoints(
            obj.reshape(-1, 1, 3), cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3] * 1000, K, D
        )[0].reshape(-1, 2)
        errors[eye + "_rms_px"] = float(
            np.sqrt(np.mean(np.sum((pixels - np.asarray(obs[f"{eye}_points"])) ** 2, axis=1)))
        )
    return {
        "id": sample["id"],
        **errors,
        "translation_mm": t * 1000,
        "rotation_deg": r,
        "passed": all(
            np.isfinite(v) and v <= POLICY["check_reprojection_rms_px"] for v in errors.values()
        ),
    }


def _evaluate(sample, solution, model):
    try:
        return check_sample(sample, solution, model)
    except (ValueError, cv2.error) as error:
        return {"id": sample["id"], "passed": False, "error": str(error)}


def _max_reprojection(results):
    if not results or any("error" in result for result in results):
        return None
    value = max(result[f"{eye}_rms_px"] for result in results for eye in ("left", "right"))
    return float(value) if np.isfinite(value) else None


def _prior_validation_report(evidence, solution, model):
    """Previously failed views remain held out after an unchanged-mount restart."""
    if evidence is None:
        return None
    if evidence.get("mount_unchanged") is not True or not evidence.get("samples"):
        raise ValueError("Prior validation evidence is missing or its mounting is incompatible.")
    results = []
    for sample in evidence["samples"]:
        result = _evaluate(sample, solution, model)
        result["source_run_id"] = sample.get("source_run_id", evidence["source_run_id"])
        results.append(result)
    passed = all(result["passed"] for result in results)
    return {
        "checks": results,
        "passed": passed,
        "max_rms_px": _max_reprojection(results),
        "reason": (
            "Previously failed views now meet the same pixel budget; fresh validation is still required."
            if passed
            else "Previously failed views still disagree with this fit. Archiving them does not resolve the error; review the measurements and model."
        ),
        "source_run_id": evidence["source_run_id"],
        "evidence_sha256": evidence.get("sha256"),
        "threshold_px": POLICY["check_reprojection_rms_px"],
        "scope": "Archived held-out regression evidence; never used to fit a calibration.",
    }


def training_assessment(samples, model, prior_validation=None):
    """Readiness is coverage AND image prediction accuracy, never a pose quota.

    Full-fit errors and leave-one-distinct-pose-out predictions must meet the
    same native-pixel budget as subsequent independent checks. Each held-out
    prediction is made by a fit that never saw that image or camera-board pose.
    Whole orientation-support clusters leave together, so jitter or translated
    repeats cannot masquerade as an independent held-out wrist tilt.
    Current-run independent checks never enter this assessment. Separately
    supplied prior failures are evaluated only after fitting, never fitted.

    The three-pose OpenCV observability floor implies four distinct poses for
    cross-validation; collecting that number alone never grants readiness.
    """
    train = [s for s in samples if s["role"] == "training" and not s.get("excluded")]
    coverage = diversity([s["T_base_tool"] for s in train])
    out = {
        "ready": False,
        "reason": "Move through wrist tilts about two different axes; more copies of one pose do not add coverage.",
        "coverage": coverage,
        "solution": None,
        "fit_max_rms_px": None,
        "leave_one_out_max_rms_px": None,
        "fit": [],
        "leave_one_out": [],
        "threshold_px": POLICY["check_reprojection_rms_px"],
        "sample_count": len(train),
        "status": "coverage",
        "coverage_method": COVERAGE_METHOD,
        "cross_validation_method": CROSS_VALIDATION_METHOD,
        "rotation_support_diameter_deg": POLICY["pose_novelty_rotation_deg"],
        "prior_validation": None,
        "fit_method": FIT_METHOD,
        "motion_consistency": motion_consistency(train),
        "cross_validation_coverage_sufficient": False,
    }
    if not coverage["sufficient"]:
        return out
    _, support, _, _ = _rotation_support([s["T_base_tool"] for s in train])
    groups = [[train[index] for index in group] for group in support]
    folds, held_out = [], []
    for group in groups:
        held_ids = {id(s) for s in group}
        fold = [s for s in train if id(s) not in held_ids]
        try:
            _solver_job(fold, require_capture_coverage=False)
        except ValueError as error:
            out["leave_one_out"].extend(
                {"id": s["id"], "passed": False, "error": str(error), "coverage_sufficient": False}
                for s in group
            )
        else:
            folds.append(fold)
            held_out.append(group)
    try:
        solutions = _solve_batches([train] + folds, model=model, cross_validation=True)
    except (ValueError, cv2.error, subprocess.SubprocessError) as error:
        out.update(status="solver", reason=f"Calibration cannot be fitted reliably: {error}")
        return out
    out["solution"] = solutions[0]
    out["prior_validation"] = _prior_validation_report(prior_validation, solutions[0], model)
    out["fit"] = [_evaluate(sample, solutions[0], model) for sample in train]
    out["fit_max_rms_px"] = _max_reprojection(out["fit"])
    for group, solution in zip(held_out, solutions[1:]):
        out["leave_one_out"].extend(_evaluate(sample, solution, model) for sample in group)
    out["leave_one_out_max_rms_px"] = _max_reprojection(out["leave_one_out"])
    out["cross_validation_coverage_sufficient"] = len(groups) >= MIN_TRAIN + 1 and all(
        r.get("coverage_sufficient", True) for r in out["leave_one_out"]
    )
    # A first solvable fit can depend on one indispensable tilt. Keep its
    # residuals visible, but collect redundant two-axis coverage before using
    # that provisional fit to halt collection. No accuracy threshold changes.
    if not out["cross_validation_coverage_sufficient"]:
        reason = (
            "Add different wrist tilts: removing a saved orientation still loses "
            "the two-axis coverage needed to validate this provisional fit."
        )
        worst = out["fit_max_rms_px"]
        if worst is not None and worst > out["threshold_px"]:
            reason += (
                f" Current worst fit error is {worst:.2f} px, above the "
                f"{out['threshold_px']:.1f} px limit; this warning remains until "
                "the fit and held-out predictions pass."
            )
        out.update(status="stability", reason=reason)
    elif not all(result["passed"] for result in out["fit"]):
        worst = out["fit_max_rms_px"]
        detail = (
            f"Worst native-camera error is {worst:.2f} px"
            if worst is not None
            else "Image prediction failed"
        )
        out.update(
            status="accuracy",
            reason=(
                f"{detail}; the budget is {out['threshold_px']:.1f} px. "
                "Robot and observed board rotations disagree by up to "
                f"{out['motion_consistency']['max_rotation_disagreement_deg']:.2f}°. "
                "Review mounting, measured pose pairing, arm kinematics and the camera model before continuing."
            ),
        )
    elif not all(result["passed"] for result in out["leave_one_out"]):
        worst = out["leave_one_out_max_rms_px"]
        detail = (
            f"Held-out training prediction reaches {worst:.2f} px"
            if worst is not None
            else "Held-out training prediction failed"
        )
        out.update(
            status="stability",
            reason=f"{detail}; the budget is {out['threshold_px']:.1f} px. Add useful orientation coverage and review inconsistent measurements.",
        )
    elif out["prior_validation"] is not None and not out["prior_validation"]["passed"]:
        out.update(status="prior_validation", reason=out["prior_validation"]["reason"])
    else:
        out.update(
            ready=True,
            status="ready",
            reason="Two-axis coverage and held-out image predictions are consistent. Continue to fresh views to verify the frozen calibration.",
        )
    return out


def validation_report(samples, solution, model, prior_validation=None):
    # Hiding a failed check cannot validate the same frozen candidate. Restarted
    # attempts must also pass hash-verified prior evidence from the parent run.
    all_checks = [s for s in samples if s["role"] == "check"]
    checks = [s for s in all_checks if not s.get("excluded")]
    results = []
    for sample in all_checks:
        result = _evaluate(sample, solution, model)
        if not sample.get("excluded") or not result["passed"]:
            results.append(result)
    spread = diversity([s["T_base_tool"] for s in checks])
    passed = len(checks) >= MIN_CHECK and spread["sufficient"] and all(r["passed"] for r in results)
    prior = _prior_validation_report(prior_validation, solution, model)
    passed = passed and (prior is None or prior["passed"])
    maximum = _max_reprojection(results)
    if prior is not None and not prior["passed"]:
        reason = prior["reason"]
    elif any(not result["passed"] for result in results):
        reason = "A fresh view disagrees with this calibration. Review the failed measurement before starting a new attempt."
    elif not spread["sufficient"]:
        reason = "Continue through fresh wrist tilts about two axes to verify the calibration across orientations."
    else:
        reason = "Fresh views cover two rotation axes and meet the native-pixel accuracy budget."
    return {
        "passed": bool(passed),
        "checks": results,
        "diversity": spread,
        "coverage": spread,
        "coverage_method": COVERAGE_METHOD,
        "required_checks": MIN_CHECK,
        "required_checks_is_observability_floor": False,
        "required_checks_basis": "Coverage policy: fresh views spanning two rotation axes; no transform is estimated from these checks.",
        "reason": reason,
        "max_rms_px": maximum,
        "threshold_px": POLICY["check_reprojection_rms_px"],
        "prior_validation": prior,
        "scope": "Independent geometric consistency in native pixels; physical task accuracy and palm mounting transform are not certified.",
    }


class ArmKinematics:
    """Measured-state FK, not command validation. Never clamp encoder angles."""

    def __init__(self, path, base="base_link", tip="link7"):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        urdf, self.joint_angle_maps, configuration, self.raw_joint_refinement = decode_arm_model(raw, with_refinement=True)
        self.configuration_correction = ReferenceGravityCorrection(configuration) if configuration is not None else None
        self.model_format = ("arm_kinematic_model.v3" if self.raw_joint_refinement is not None else
                             "arm_kinematic_model.v2" if configuration is not None else
                             "arm_kinematic_model.v1" if self.joint_angle_maps else "urdf")
        tree = ET.fromstring(urdf)
        by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
        chain, link = [], tip
        while link != base:
            if link not in by_child or len(chain) >= len(by_child):
                raise ValueError("URDF does not contain the requested base-to-tool chain.")
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.names = [j.get("name") for j in self.chain if j.get("type") != "fixed"]
        self.base, self.tip = base, tip

    def pose(self, names, positions):
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("Joint feedback names/positions are invalid.")
        q = dict(zip(names, positions))
        if any(n not in q or not np.isfinite(q[n]) for n in self.names):
            raise ValueError("Measured follower feedback must contain every arm joint.")
        correction = self.configuration_correction
        required = correction.names if correction is not None and correction.target_joint in self.names else self.names
        if any(n not in q or not np.isfinite(q[n]) for n in required):
            raise ValueError("Static wrist correction requires complete measured arm feedback.")
        mapped = {
            name: calibrated_joint_angle(name, q[name], self.joint_angle_maps[name])
            if name in self.joint_angle_maps else q[name]
            for name in required
        }
        if correction is not None and correction.target_joint in self.names:
            mapped[correction.target_joint] += correction.delta(mapped)
        if self.raw_joint_refinement is not None and self.raw_joint_refinement['target_joint'] in self.names:
            mapped[self.raw_joint_refinement['target_joint']] += refinement_delta(self.raw_joint_refinement, q)
        out = np.eye(4)
        for j in self.chain:
            origin = j.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rot = np.eye(3)
            for axis, angle in zip(np.eye(3), rpy):
                rot = cv2.Rodrigues(axis * angle)[0] @ rot
            out = out @ transform(rot, xyz)
            if j.get("type") == "fixed":
                continue
            if j.get("type") not in ("revolute", "continuous"):
                raise ValueError("Only fixed/revolute arm chains are supported.")
            axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            value = mapped[j.get("name")]
            out = out @ transform(
                cv2.Rodrigues(axis / np.linalg.norm(axis) * value)[0], np.zeros(3)
            )
        return checked_transform(out)

    def limit_diagnostics(self, names, positions):
        """Report nominal-envelope excursions without altering valid measured FK.

        Command limits remain in the robot controller. Measured feedback can
        exceed them (e.g. tracking error); clipping would corrupt calibration.
        """
        q = dict(zip(names, positions))
        result = []
        for joint in self.chain:
            limit = joint.find("limit")
            if joint.get("type") != "revolute" or limit is None:
                continue
            value = float(q[joint.get("name")])
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            if value < lower or value > upper:
                result.append(
                    {
                        "joint": joint.get("name"),
                        "measured_rad": value,
                        "lower_rad": lower,
                        "upper_rad": upper,
                        "outside_by_rad": max(lower - value, value - upper),
                        "action": "recorded_unmodified; no robot command",
                    }
                )
        return result
