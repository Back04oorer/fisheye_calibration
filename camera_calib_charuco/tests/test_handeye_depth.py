"""Offline diagnostic tests: synthetic curved target and held-out validation.

Use /usr/bin/python3 -m pytest on this host (SciPy is intentionally separate
from the running camera environment).
"""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
import cv2

spec = importlib.util.spec_from_file_location('depth_diagnostic', Path(__file__).parents[1]/'scripts/diagnose_handeye_depth.py')
depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth)


def synthetic(noise=0):
    rng = np.random.default_rng(247)
    k = np.array([[600., 0, 640], [0, 600., 400], [0, 0, 1.]])
    cal = dict(K1=k, K2=k, D1=np.array([-.03, .01, 0, 0]), D2=np.array([-.02, .005, 0, 0]),
               R=cv2.Rodrigues(np.array([.002, -.003, .001]))[0], T=np.array([-80., 0, 0]))
    engine = depth.StereoRecovery(cal, 12)
    x = depth.transform(np.array([.1, -.2, .15, .3, -.15, .1]))
    mount = depth.transform(np.array([.2, .1, -.1, .14, -.02, .03]))
    points = np.array([[.025*(i%3+1), .025*(i//3+1), .003*((i%3)-1)**2] for i in range(12)])
    tool_points = depth.apply(mount, points)
    samples = []
    for i in range(18):
        camera_board = depth.transform(np.r_[rng.uniform(-.5, .5, 3), rng.uniform(-.1, .1, 2), rng.uniform(.45, .7)])
        g = x @ camera_board @ np.linalg.inv(mount)
        observation = {}
        for eye, pose, ik, dk in [('left', camera_board, 'K1', 'D1'), ('right', engine.stereo@camera_board, 'K2', 'D2')]:
            uv = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), cv2.Rodrigues(pose[:3,:3])[0], pose[:3,3], cal[ik], cal[dk])[0].reshape(-1,2)
            observation[eye+'_points'] = (uv+rng.normal(0, noise, uv.shape)).tolist()
            observation[eye+'_ids'] = list(range(12))
        samples.append(dict(id=str(i), role='training' if i<12 else 'check', T_base_tool=g.tolist(), observation=observation))
    initial = depth.transform(depth.parameters(x)+np.array([.02, -.01, .015, .01, -.02, .01]))
    data = dict(board=dict(columns=4, rows=5), calibration=cal, samples=samples,
                policy=dict(check_reprojection_rms_px=3., minimum_check_poses=5),
                solution=dict(T_base_camera_left_raw=initial.tolist()))
    return data, x, tool_points


def test_exact_curved_target_recovers_metric_shape_and_camera():
    data, camera, points = synthetic()
    report, clouds = depth.recover(data)
    assert clouds.shape == (12, 12, 3)
    np.testing.assert_allclose(report['T_base_camera_left_raw'], camera, atol=1e-6)
    np.testing.assert_allclose(report['points_tool_m'], points, atol=1e-6)
    assert abs(report['plane_peak_to_peak_mm']-3) < 1e-4
    assert report['held_out_pass_count'] == report['held_out_total'] == 6
    assert report['applied'] is False


def test_subpixel_noise_still_recovers_and_checks():
    data, camera, points = synthetic(noise=.08)
    report, _ = depth.recover(data)
    fitted = np.array(report['T_base_camera_left_raw'])
    assert np.linalg.norm(fitted[:3, 3]-camera[:3, 3]) < .001
    assert np.max(np.linalg.norm(np.asarray(report['points_tool_m'])-points, axis=1)) < .001
    assert report['held_out_pass_count'] == 6


def test_check_errors_are_never_fitted_away_and_inputs_are_preserved():
    data, _, _ = synthetic(noise=.08)
    baseline, _ = depth.recover(data)
    broken = copy.deepcopy(data)
    for s in broken['samples'][12:]:
        s['observation']['left_points'] = (np.array(s['observation']['left_points'])+[10, 0]).tolist()
    original = copy.deepcopy(broken['samples'])
    report, _ = depth.recover(broken)
    assert report['T_base_camera_left_raw'] == baseline['T_base_camera_left_raw']
    assert report['points_tool_m'] == baseline['points_tool_m']
    assert report['held_out_pass_count'] == 0
    assert broken['samples'] == original
    assert set(report['training_ids']).isdisjoint(report['held_out_check_ids'])


def test_incomplete_correspondences_are_rejected():
    data, _, _ = synthetic()
    data['samples'][0]['observation']['right_ids'].pop()
    with pytest.raises(ValueError, match='all corner IDs'):
        depth.recover(data)
