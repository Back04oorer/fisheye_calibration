"""Static, fixed-payload empirical FK correction; never a motor command.

The gravity-shaped basis is a compact configuration feature. Its fitted
coefficients are not measurements of joint stiffness or a hardware diagnosis.
"""
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

CONFIGURATION_MODEL = "relative_gravity_basis_v1"


def configuration_parameters(spec):
    if not isinstance(spec, dict) or spec.get("model") != CONFIGURATION_MODEL:
        raise ValueError("Unsupported static configuration correction.")
    result = {"model": CONFIGURATION_MODEL}
    for key in ("target_joint", "base_frame", "tip_frame"):
        value = spec.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("Configuration correction needs named joints and frames.")
        result[key] = value
    for key in ("coefficients_rad", "reference_basis", "gravity_vector_base"):
        value = spec.get(key)
        if (not isinstance(value, list) or len(value) != 3 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not np.isfinite(v) for v in value)):
            raise ValueError("Configuration correction needs finite three-vectors.")
        result[key] = list(map(float, value))
    if not np.isclose(np.linalg.norm(result["gravity_vector_base"]), 1., atol=1e-10):
        raise ValueError("Configuration gravity direction must be a unit vector.")
    offsets = spec.get("reference_zero_offsets_rad")
    if (not isinstance(offsets, dict) or not offsets or
            any(not isinstance(n, str) or isinstance(v, bool) or
                not isinstance(v, (int, float)) or not np.isfinite(v)
                for n, v in offsets.items())):
        raise ValueError("Configuration reference offsets must be finite radians.")
    result["reference_zero_offsets_rad"] = dict(offsets)
    return result


def checked_configuration(spec, output_urdf):
    result = configuration_parameters(spec)
    xml = spec.get("reference_urdf_xml")
    if not isinstance(xml, str) or hashlib.sha256(xml.encode()).hexdigest() != spec.get("reference_urdf_sha256"):
        raise ValueError("Configuration reference geometry does not match its fingerprint.")
    target = result["target_joint"]
    output = {j.get("name"): j for j in ET.fromstring(output_urdf).findall("joint")}
    if target not in output or output[target].get("type") not in ("revolute", "continuous"):
        raise ValueError("Configuration correction names an unknown revolute joint.")
    result.update(reference_urdf_xml=xml, reference_urdf_sha256=spec["reference_urdf_sha256"])
    # Validate the reference chain at the model boundary, including its target.
    ReferenceGravityCorrection(result)
    return result


class ReferenceGravityCorrection:
    """Cached reference-chain geometry, evaluated using already mapped angles."""
    def __init__(self, spec):
        self.target_joint = spec["target_joint"]
        self.coefficients = np.asarray(spec["coefficients_rad"], dtype=float)
        self.reference_basis = np.asarray(spec["reference_basis"], dtype=float)
        self.gravity = np.asarray(spec["gravity_vector_base"], dtype=float)
        tree = ET.fromstring(spec["reference_urdf_xml"])
        by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
        chain, link = [], spec["tip_frame"]
        while link != spec["base_frame"]:
            if link not in by_child or len(chain) >= len(by_child):
                raise ValueError("Configuration reference chain is invalid.")
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain, self.names = [], []
        for joint in reversed(chain):
            kind, name = joint.get("type"), joint.get("name")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError("Unsupported configuration reference joint.")
            origin = joint.find("origin")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            if rpy.shape != (3,) or not np.isfinite(rpy).all():
                raise ValueError("Invalid configuration reference rotation.")
            rotation = Rotation.from_euler("xyz", rpy).as_matrix()
            axis = None
            if kind != "fixed":
                node = joint.find("axis")
                axis = np.fromstring(node.get("xyz", "0 0 1"), sep=" ") if node is not None else np.array([0., 0., 1.])
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                    raise ValueError("Invalid configuration reference axis.")
                axis = axis / np.linalg.norm(axis)
                self.names.append(name)
            self.chain.append((name, rotation, axis))
        if self.target_joint not in self.names:
            raise ValueError("Configuration target is absent from its reference chain.")

    def delta(self, mapped_values):
        if any(n not in mapped_values or not np.isfinite(mapped_values[n]) for n in self.names):
            raise ValueError("Static wrist correction requires complete measured arm feedback.")
        rotation, target_axis = np.eye(3), None
        for name, origin, axis in self.chain:
            rotation = rotation @ origin
            if axis is None:
                continue
            if name == self.target_joint:
                target_axis = rotation @ axis
            rotation = rotation @ Rotation.from_rotvec(axis * mapped_values[name]).as_matrix()
        basis = np.array([target_axis @ np.cross(rotation[:, j], self.gravity) for j in range(3)])
        return float(self.coefficients @ (basis - self.reference_basis))
