"""Per-arm calibrated geometry for raw encoder feedback.

Joint-zero corrections live in URDF joint origins. Controller commands and
reported encoder values retain their original hardware convention. Each run
freezes the complete model; a corrected right arm never changes the left arm.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from .handeye_configuration import configuration_parameters, checked_configuration
from .handeye_refinement import checked_refinement

SCHEMA = "arm_joint_zero_calibration.v1"
ANGLE_PROFILE_SCHEMA = "arm_kinematic_calibration.v2"
MODEL_SCHEMA = "arm_kinematic_model.v1"
CONFIGURATION_PROFILE_SCHEMA = "arm_configuration_calibration.v3"
CONFIGURATION_MODEL_SCHEMA = "arm_kinematic_model.v2"
REFINEMENT_MODEL_SCHEMA = "arm_kinematic_model.v3"
REFINEMENT_PROFILE_SCHEMA = "arm_configuration_calibration.v4"


def checked_angle_maps(maps, urdf):
    """Validate a monotonic empirical map and its measured qualification interval."""
    if not isinstance(maps, dict) or not maps:
        raise ValueError("A calibrated joint model must name its angle maps.")
    joints = {j.get("name"): j for j in ET.fromstring(urdf).findall("joint")}
    result = {}
    for name, spec in maps.items():
        if name not in joints or joints[name].get("type") not in ("revolute", "continuous"):
            raise ValueError("Angle calibration names an unknown revolute joint.")
        if not isinstance(spec, dict) or spec.get("model") != "relative_quadratic_v1":
            raise ValueError("Unsupported joint-angle calibration model.")
        coefficients, domain = spec.get("coefficients_rad"), spec.get("qualified_range_rad")
        if not isinstance(coefficients, list) or len(coefficients) != 2 or not isinstance(domain, list) or len(domain) != 2:
            raise ValueError("Angle calibration needs two coefficients and a qualified interval.")
        values = [spec.get("reference_rad"), spec.get("scale_rad"), *coefficients, *domain]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
            raise ValueError("Joint-angle calibration values must be finite radians.")
        reference, scale = values[:2]
        if scale <= 0 or domain[0] >= domain[1] or not domain[0] <= reference <= domain[1]:
            raise ValueError("Invalid joint-angle calibration interval or scale.")
        x = (np.array(domain) - reference) / scale
        derivative = 1 + coefficients[0] / scale + 2 * coefficients[1] * x / scale
        if np.min(derivative) <= 0:
            raise ValueError("Joint-angle calibration must stay monotonic over its qualified interval.")
        result[name] = {
            "model": "relative_quadratic_v1", "reference_rad": float(reference),
            "scale_rad": float(scale), "coefficients_rad": list(map(float, coefficients)),
            "qualified_range_rad": list(map(float, domain)),
        }
    return result


def calibrated_joint_model(raw, offsets, maps, configuration=None, refinement=None):
    """Self-contained model: corrected URDF plus explicit raw-to-model angle maps.

    JSON is intentional: a nonlinear encoder map cannot be represented by a
    standard URDF. Consumers must use ArmKinematics or implement this map.
    """
    urdf = calibrated_urdf(raw, offsets)
    data = {
        "schema": MODEL_SCHEMA,
        "urdf_xml": urdf.decode("utf-8"),
        "urdf_sha256": hashlib.sha256(urdf).hexdigest(),
        "joint_angle_maps": checked_angle_maps(maps, urdf),
        "input_convention": "Raw measured joint angles in radians; maps affect FK only.",
    }
    if configuration is not None:
        config = configuration_parameters(configuration)
        reference = calibrated_urdf(raw, config["reference_zero_offsets_rad"])
        config.update(reference_urdf_xml=reference.decode("utf-8"),
                      reference_urdf_sha256=hashlib.sha256(reference).hexdigest())
        data["configuration_correction"] = checked_configuration(config, urdf)
        data["schema"] = CONFIGURATION_MODEL_SCHEMA
    if refinement is not None:
        if configuration is None:
            raise ValueError('Wrist refinement requires its parent configuration correction')
        data['raw_joint_refinement'] = checked_refinement(refinement, urdf)
        data['schema'] = REFINEMENT_MODEL_SCHEMA
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")


def decode_arm_model(raw, *, with_configuration=False, with_refinement=False):
    """Read ordinary URDFs or hash-bound, self-contained calibrated models."""
    if not raw.lstrip().startswith(b"{"):
        return (raw, {}, None, None) if with_refinement else (raw, {}, None) if with_configuration else (raw, {})
    data = json.loads(raw)
    if data.get("schema") not in (MODEL_SCHEMA, CONFIGURATION_MODEL_SCHEMA, REFINEMENT_MODEL_SCHEMA) or not isinstance(data.get("urdf_xml"), str):
        raise ValueError("Unsupported calibrated arm model.")
    urdf = data["urdf_xml"].encode("utf-8")
    if hashlib.sha256(urdf).hexdigest() != data.get("urdf_sha256"):
        raise ValueError("Calibrated arm geometry does not match its fingerprint.")
    maps = checked_angle_maps(data.get("joint_angle_maps"), urdf)
    configuration = None
    if data["schema"] in (CONFIGURATION_MODEL_SCHEMA, REFINEMENT_MODEL_SCHEMA):
        if not (with_configuration or with_refinement):
            raise ValueError("This arm model requires its static configuration correction.")
        configuration = checked_configuration(data.get("configuration_correction"), urdf)
    elif data.get("configuration_correction") is not None:
        raise ValueError("Static configuration corrections require a version-2 arm model.")
    refinement = None
    if data['schema'] == REFINEMENT_MODEL_SCHEMA:
        if not with_refinement:
            raise ValueError('This arm model requires its raw-joint refinement')
        refinement = checked_refinement(data.get('raw_joint_refinement'), urdf)
    elif data.get('raw_joint_refinement') is not None:
        raise ValueError('Raw-joint refinement requires a version-3 arm model')
    return (urdf, maps, configuration, refinement) if with_refinement else (urdf, maps, configuration) if with_configuration else (urdf, maps)


def calibrated_joint_angle(name, value, spec):
    lower, upper = spec["qualified_range_rad"]
    # Numerical roundoff only; do not silently clamp or extrapolate a fitted map.
    if value < lower - 1e-12 or value > upper + 1e-12:
        raise ValueError(
            f"{name} is outside the software-calibrated interval "
            f"({np.degrees(lower):.2f}° to {np.degrees(upper):.2f}°). "
            "Use the validated wrist range or requalify the arm model. Raw feedback is unchanged."
        )
    x = (value - spec["reference_rad"]) / spec["scale_rad"]
    a, b = spec["coefficients_rad"]
    return value + a * x + b * x * x


def calibrated_urdf(raw, offsets):
    """Bake q_model = q_encoder + offset into revolute joint origins."""
    tree = ET.fromstring(raw)
    joints = {j.get("name"): j for j in tree.findall("joint")}
    if not isinstance(offsets, dict) or not offsets:
        raise ValueError("Joint-zero calibration must name its offsets in radians.")
    for name, offset in offsets.items():
        if name not in joints or joints[name].get("type") not in ("revolute", "continuous"):
            raise ValueError(f"Joint-zero calibration names an unknown revolute joint: {name}.")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not np.isfinite(offset)
        ):
            raise ValueError("Joint-zero offsets must be finite numbers in radians.")
        joint = joints[name]
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        if axis.shape != (3,) or rpy.shape != (3,) or np.linalg.norm(axis) == 0:
            raise ValueError("URDF joint geometry is invalid.")
        rotation = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(
            axis / np.linalg.norm(axis) * offset
        )
        origin.set("rpy", " ".join(format(v, ".17g") for v in rotation.as_euler("xyz")))
    return ET.tostring(tree, encoding="utf-8", xml_declaration=True)


def select_arm_model(source, side, profile_directory):
    """Load only a qualified, hash-bound profile for the selected follower."""
    source, root = Path(source).resolve(), Path(profile_directory).resolve()
    if side not in ("left", "right"):
        raise ValueError("Choose the left or right follower arm.")
    profile_path = root / side / "profile.json"
    if not profile_path.exists():
        return source, None
    profile = json.loads(profile_path.read_text())
    raw = source.read_bytes()
    if profile.get("schema") not in (SCHEMA, ANGLE_PROFILE_SCHEMA, CONFIGURATION_PROFILE_SCHEMA, REFINEMENT_PROFILE_SCHEMA) or profile.get("side") != side:
        raise ValueError("Arm geometry profile does not match the selected follower.")
    if profile.get("source_urdf_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Arm geometry source changed; requalify its joint-zero calibration.")
    if (profile.get("qualification") or {}).get("status") != "independent_checks_passed":
        raise ValueError("Arm geometry profile has not passed independent checks.")
    model = (profile_path.parent / profile["model_file"]).resolve()
    if not model.is_relative_to(profile_path.parent) or not model.is_file():
        raise ValueError("Calibrated arm model is missing or outside its profile directory.")
    if profile["schema"] in (ANGLE_PROFILE_SCHEMA, CONFIGURATION_PROFILE_SCHEMA, REFINEMENT_PROFILE_SCHEMA):
        configuration = profile.get("configuration_correction")
        if (profile["schema"] in (CONFIGURATION_PROFILE_SCHEMA, REFINEMENT_PROFILE_SCHEMA)) != (configuration is not None):
            raise ValueError("Static correction and profile schema disagree.")
        refinement = profile.get('raw_joint_refinement')
        if (profile['schema'] == REFINEMENT_PROFILE_SCHEMA) != (refinement is not None):
            raise ValueError('Wrist refinement and profile schema disagree')
        expected = calibrated_joint_model(raw, profile["joint_zero_offsets_rad"], profile.get("joint_angle_maps"), configuration, refinement)
    else:
        if profile.get("joint_angle_maps"):
            raise ValueError("Nonlinear angle maps require a version-2 arm profile.")
        expected = calibrated_urdf(raw, profile["joint_zero_offsets_rad"])
    if (
        hashlib.sha256(model.read_bytes()).hexdigest() != profile.get("model_sha256")
        or model.read_bytes() != expected
    ):
        raise ValueError("Calibrated arm model does not match its recorded zero corrections.")
    return model, {
        **profile,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }
