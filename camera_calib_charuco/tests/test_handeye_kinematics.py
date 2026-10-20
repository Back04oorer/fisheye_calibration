import hashlib
import json

import numpy as np
import pytest

from stereo_calibration.handeye import ArmKinematics
from stereo_calibration.handeye_kinematics import SCHEMA, calibrated_urdf, select_arm_model


def source_model(tmp_path):
    path = tmp_path / "source.urdf"
    path.write_text("""<robot name="test"><link name="base_link"/><link name="link1"/>
      <link name="link7"/><joint name="joint1" type="revolute">
      <parent link="base_link"/><child link="link1"/><origin xyz="0 0 .1" rpy=".2 -.3 .4"/>
      <axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
      <joint name="joint2" type="revolute"><parent link="link1"/><child link="link7"/>
      <origin xyz=".1 -.3 .2" rpy="1.5707963267948966 0 .7"/>
      <axis xyz="1 0 0"/><limit lower="-2" upper="2"/></joint></robot>""")
    return path


def profile(tmp_path, source, offsets):
    root = tmp_path / "profiles"
    folder = root / "right"
    folder.mkdir(parents=True)
    model = calibrated_urdf(source.read_bytes(), offsets)
    (folder / "arm_model.urdf").write_bytes(model)
    manifest = {
        "schema": SCHEMA,
        "side": "right",
        "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "joint_zero_offsets_rad": offsets,
        "model_file": "arm_model.urdf",
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "qualification": {"status": "independent_checks_passed"},
    }
    (folder / "profile.json").write_text(json.dumps(manifest))
    return root, folder, manifest


def test_calibrated_model_preserves_raw_feedback_convention(tmp_path):
    source = source_model(tmp_path)
    offsets = {"joint1": 0.1, "joint2": -0.07}
    root, _, _ = profile(tmp_path, source, offsets)
    selected, metadata = select_arm_model(source, "right", root)
    nominal, calibrated = ArmKinematics(source), ArmKinematics(selected)
    for q in np.random.default_rng(40).uniform(-1, 1, (20, 2)):
        raw = q.copy()
        np.testing.assert_allclose(
            calibrated.pose(nominal.names, raw),
            nominal.pose(nominal.names, q + [0.1, -0.07]),
            atol=1e-12,
        )
        np.testing.assert_array_equal(raw, q)
    assert metadata["joint_zero_offsets_rad"] == offsets
    assert select_arm_model(source, "left", root) == (source, None)


@pytest.mark.parametrize("change", ["source", "model", "unqualified", "side", "escape"])
def test_invalid_or_stale_profile_is_rejected(tmp_path, change):
    source = source_model(tmp_path)
    root, folder, manifest = profile(tmp_path, source, {"joint2": 0.1})
    if change == "source":
        source.write_text(source.read_text().replace('xyz=".1 -.3 .2"', 'xyz=".1 -.4 .2"'))
    elif change == "model":
        (folder / "arm_model.urdf").write_bytes(source.read_bytes())
    elif change == "unqualified":
        manifest["qualification"]["status"] = "pending"
    elif change == "side":
        manifest["side"] = "left"
    elif change == "escape":
        manifest["model_file"] = "../../source.urdf"
    (folder / "profile.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        select_arm_model(source, "right", root)


@pytest.mark.parametrize(
    "offsets", [{"missing": 0.1}, {"joint2": float("nan")}, {"joint2": True}, {}]
)
def test_invalid_zero_corrections_are_rejected(tmp_path, offsets):
    with pytest.raises(ValueError):
        calibrated_urdf(source_model(tmp_path).read_bytes(), offsets)


def angle_map():
    return {"joint2": {"model": "relative_quadratic_v1", "reference_rad": 0.,
                       "scale_rad": 1., "coefficients_rad": [.02, .01],
                       "qualified_range_rad": [-.9, .9]}}


def test_nonlinear_model_profile_preserves_raw_values_and_snapshots(tmp_path):
    from stereo_calibration.handeye_kinematics import ANGLE_PROFILE_SCHEMA, calibrated_joint_model
    source = source_model(tmp_path)
    root = tmp_path / "profiles"
    folder = root / "right"
    folder.mkdir(parents=True)
    offsets = {"joint2": -.07}
    maps = angle_map()
    raw = calibrated_joint_model(source.read_bytes(), offsets, maps)
    path = folder / "arm_model.json"
    path.write_bytes(raw)
    manifest = dict(schema=ANGLE_PROFILE_SCHEMA, side="right", model_file=path.name,
                    source_urdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    joint_zero_offsets_rad=offsets, joint_angle_maps=maps,
                    model_sha256=hashlib.sha256(raw).hexdigest(),
                    qualification={"status": "independent_checks_passed"})
    (folder / "profile.json").write_text(json.dumps(manifest))
    selected, _ = select_arm_model(source, "right", root)
    calibrated, nominal = ArmKinematics(selected), ArmKinematics(source)
    for q in np.random.default_rng(8).uniform(-.8, .8, (20, 2)):
        before = q.copy()
        corrected = q.copy()
        corrected[1] += -.07 + .02 * q[1] + .01 * q[1] ** 2
        np.testing.assert_allclose(calibrated.pose(nominal.names, q), nominal.pose(nominal.names, corrected), atol=1e-12)
        np.testing.assert_array_equal(q, before)
    copied = tmp_path / "snapshot.json"
    copied.write_bytes(raw)
    assert ArmKinematics(copied).sha256 == calibrated.sha256
    with pytest.raises(ValueError, match="software-calibrated interval"):
        calibrated.pose(nominal.names, [0, 1])
    # Controller limits remain raw-angle diagnostics, not a transformed/clamped value.
    assert calibrated.limit_diagnostics(nominal.names, [0, 2.5])[0]["measured_rad"] == 2.5
    manifest["joint_angle_maps"]["joint2"]["coefficients_rad"][0] += .01
    (folder / "profile.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        select_arm_model(source, "right", root)


@pytest.mark.parametrize("change", ["nonfinite", "boolean", "reversed", "nonmonotonic", "unknown", "scale", "model"])
def test_invalid_nonlinear_models_are_rejected(tmp_path, change):
    from stereo_calibration.handeye_kinematics import calibrated_joint_model
    source = source_model(tmp_path)
    maps = angle_map()
    spec = maps["joint2"]
    if change == "nonfinite": spec["coefficients_rad"][0] = float("nan")
    elif change == "boolean": spec["coefficients_rad"][0] = True
    elif change == "reversed": spec["qualified_range_rad"] = [.9, -.9]
    elif change == "nonmonotonic": spec["coefficients_rad"] = [-2, 0]
    elif change == "unknown": maps["missing"] = maps.pop("joint2")
    elif change == "scale": spec["scale_rad"] = 0
    elif change == "model": spec["model"] = "unsupported"
    with pytest.raises(ValueError):
        calibrated_joint_model(source.read_bytes(), {"joint2": .1}, maps)


def test_embedded_geometry_tampering_is_rejected(tmp_path):
    from stereo_calibration.handeye_kinematics import calibrated_joint_model
    source = source_model(tmp_path)
    data = json.loads(calibrated_joint_model(source.read_bytes(), {"joint2": .1}, angle_map()))
    data["urdf_xml"] = data["urdf_xml"].replace('xyz=".1 -.3 .2"', 'xyz=".2 -.3 .2"')
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint"):
        ArmKinematics(path)
