import copy
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stereo_calibration.handeye import ArmKinematics
from stereo_calibration.handeye_kinematics import (
    CONFIGURATION_PROFILE_SCHEMA,
    calibrated_joint_model,
    decode_arm_model,
    select_arm_model,
)

SOURCE = b"""<robot name="test"><link name="base_link"/><link name="link1"/><link name="link7"/>
<joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/>
<origin xyz="0 0 .1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
<joint name="joint2" type="revolute"><parent link="link1"/><child link="link7"/>
<origin xyz="0 0 .1" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="-2" upper="2"/></joint></robot>"""
OFFSETS = {"joint1": 0.2, "joint2": 0.1}
MAPS = {
    "joint2": {
        "model": "relative_quadratic_v1",
        "reference_rad": 0.0,
        "scale_rad": 1.0,
        "coefficients_rad": [0.02, 0.01],
        "qualified_range_rad": [-0.9, 0.9],
    }
}
CONFIG = {
    "model": "relative_gravity_basis_v1",
    "target_joint": "joint2",
    "base_frame": "base_link",
    "tip_frame": "link7",
    "reference_zero_offsets_rad": {"joint1": 0.2},
    "coefficients_rad": [0.0, 0.07, 0.04],
    "reference_basis": [0.0, -1.0, 0.0],
    "gravity_vector_base": [0.0, 0.0, -1.0],
}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.json"
    path.write_bytes(calibrated_joint_model(SOURCE, OFFSETS, MAPS, CONFIG))
    return path


def test_static_basis_matches_analytic_rotation_without_changing_feedback(model):
    k = ArmKinematics(model)
    for raw in np.random.default_rng(12).uniform(-0.8, 0.8, (25, 2)):
        before = raw.copy()
        mapped = raw[1] + 0.02 * raw[1] + 0.01 * raw[1] ** 2
        delta = 0.07 * (1 - np.cos(mapped)) + 0.04 * np.sin(mapped)
        expected = Rotation.from_euler("z", raw[0] + 0.2) * Rotation.from_euler(
            "x", mapped + 0.1 + delta
        )
        actual = k.pose(k.names, raw)
        np.testing.assert_allclose(actual[:3, :3], expected.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(actual[:3, 3], [0, 0, 0.2], atol=1e-12)
        np.testing.assert_array_equal(raw, before)


def test_snapshot_and_legacy_reader_behavior(model, tmp_path):
    raw = model.read_bytes()
    copied = tmp_path / "other.json"
    copied.write_bytes(raw)
    np.testing.assert_array_equal(
        ArmKinematics(model).pose(["joint1", "joint2"], [0.1, 0.3]),
        ArmKinematics(copied).pose(["joint1", "joint2"], [0.1, 0.3]),
    )
    with pytest.raises(ValueError, match="requires its static"):
        decode_arm_model(raw)
    old = calibrated_joint_model(SOURCE, OFFSETS, MAPS)
    assert len(decode_arm_model(old)) == 2


def test_upstream_pose_and_angle_domain_remain_explicit(model):
    k = ArmKinematics(model, tip="link1")
    np.testing.assert_allclose(
        k.pose(["joint1"], [0.4])[:3, :3], Rotation.from_euler("z", 0.6).as_matrix()
    )
    with pytest.raises(ValueError, match="outside the software-calibrated interval"):
        ArmKinematics(model).pose(["joint1", "joint2"], [0.0, 1.0])


@pytest.mark.parametrize(
    "change",
    ["nan", "boolean", "gravity", "target", "chain", "reference_hash", "schema", "missing"],
)
def test_invalid_snapshot_is_rejected(model, change):
    d = json.loads(model.read_bytes())
    c = d["configuration_correction"]
    if change == "nan":
        c["coefficients_rad"][0] = float("nan")
    elif change == "boolean":
        c["reference_basis"][0] = True
    elif change == "gravity":
        c["gravity_vector_base"] = [0, 0, 0]
    elif change == "target":
        c["target_joint"] = "missing"
    elif change == "chain":
        c["tip_frame"] = "missing"
    elif change == "reference_hash":
        c["reference_urdf_xml"] += " "
    elif change == "schema":
        d["schema"] = "arm_kinematic_model.v1"
    elif change == "missing":
        d.pop("configuration_correction")
    model.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ArmKinematics(model)


def test_profile_binds_source_side_and_configuration(model, tmp_path):
    source = tmp_path / "source.urdf"
    source.write_bytes(SOURCE)
    root = tmp_path / "profiles"
    folder = root / "right"
    folder.mkdir(parents=True)
    raw = model.read_bytes()
    (folder / "arm_model.json").write_bytes(raw)
    data = {
        "schema": CONFIGURATION_PROFILE_SCHEMA,
        "side": "right",
        "source_urdf_sha256": hashlib.sha256(SOURCE).hexdigest(),
        "model_file": "arm_model.json",
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "joint_zero_offsets_rad": OFFSETS,
        "joint_angle_maps": MAPS,
        "configuration_correction": copy.deepcopy(CONFIG),
        "qualification": {"status": "independent_checks_passed"},
    }
    manifest = folder / "profile.json"
    manifest.write_text(json.dumps(data))
    selected, _ = select_arm_model(source, "right", root)
    assert selected.read_bytes() == raw
    assert select_arm_model(source, "left", root) == (source, None)
    data["configuration_correction"]["coefficients_rad"][0] += 0.01
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        select_arm_model(source, "right", root)
