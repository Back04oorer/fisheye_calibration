import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stereo_calibration.handeye import ArmKinematics
from stereo_calibration.handeye_kinematics import calibrated_joint_model, decode_arm_model
from test_handeye_configuration import SOURCE, OFFSETS, MAPS, CONFIG

SPEC = dict(model='raw_joint_polynomial_v1', target_joint='joint2',
            joint_names=['joint1','joint2'],reference_rad=[0,0],scale_rad=[1,1],
            observed_range_rad=[[-.8,.8],[-.8,.8]],powers=[[1,1],[0,2]],
            coefficients_rad=[.08,-.03])


def test_raw_feedback_basis_is_applied_once_after_parent_corrections(tmp_path):
    path=tmp_path/'model.json'
    path.write_bytes(calibrated_joint_model(SOURCE,OFFSETS,MAPS,CONFIG,SPEC))
    k=ArmKinematics(path)
    for q in np.random.default_rng(7).uniform(-.8,.8,(30,2)):
        before=q.copy()
        mapped=q[1]+.02*q[1]+.01*q[1]**2
        gravity=.07*(1-np.cos(mapped))+.04*np.sin(mapped)
        extra=.08*q[0]*q[1]-.03*q[1]**2
        expected=Rotation.from_euler('z',q[0]+.2)*Rotation.from_euler('x',mapped+.1+gravity+extra)
        np.testing.assert_allclose(k.pose(k.names,q)[:3,:3],expected.as_matrix(),atol=1e-12)
        np.testing.assert_array_equal(q,before)
    with pytest.raises(ValueError,match='requires its raw-joint'):
        decode_arm_model(path.read_bytes(),with_configuration=True)
    with pytest.raises(ValueError,match='no extrapolation'):
        k.pose(k.names,[.81,0])
    # Upstream links do not depend on the downstream wrist refinement.
    assert ArmKinematics(path,tip='link1').pose(['joint1'],[.2]).shape==(4,4)


@pytest.mark.parametrize('field,value', [('scale_rad',[0,1]),('coefficients_rad',[float('nan'),0]),
    ('powers',[[0,1.5],[1,1]]),('observed_range_rad',[[.8,-.8],[-.8,.8]]),('target_joint','missing')])
def test_invalid_refinement_is_rejected(field,value):
    spec={**SPEC,field:value}
    with pytest.raises(ValueError):calibrated_joint_model(SOURCE,OFFSETS,MAPS,CONFIG,spec)


def test_observed_mean_boundary_accepts_one_encoder_tick_without_clamping():
    from stereo_calibration.handeye_refinement import refinement_delta, FEEDBACK_ROUNDING_RAD
    q = .8 + .9 * FEEDBACK_ROUNDING_RAD
    assert refinement_delta(SPEC, {'joint1': q, 'joint2': .4}) == pytest.approx(.08*q*.4-.03*.4**2)
    with pytest.raises(ValueError, match='no extrapolation'):
        refinement_delta(SPEC, {'joint1': .8+1.1*FEEDBACK_ROUNDING_RAD, 'joint2': .4})
    with pytest.raises(ValueError, match='no extrapolation'):
        refinement_delta(SPEC, {'joint1': -.8-1.1*FEEDBACK_ROUNDING_RAD, 'joint2': .4})
