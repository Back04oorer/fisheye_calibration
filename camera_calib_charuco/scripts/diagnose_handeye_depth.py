"""Offline stereo shape / hand-eye recovery experiment; never applies calibration.

Run with a Python containing NumPy, OpenCV fisheye and SciPy. Units: metres in
transforms / reconstructed points, native pixels in observations. The camera
model and measured robot poses stay fixed. Only training images enter the fit.
Objective reference: https://github.com/koide3/st_handeye_graph (method inspiration,
not copied code). Here the target's individual 3D corners are also unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def transform(v):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(v[:3])[0]
    result[:3, 3] = v[3:6]
    return result


def parameters(t):
    return np.r_[cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]]


def apply(t, points):
    return points @ t[:3, :3].T + t[:3, 3]


class StereoRecovery:
    def __init__(self, calibration, count):
        self.cal = {key: np.asarray(value, dtype=float) for key, value in calibration.items()}
        self.stereo = np.eye(4)
        self.stereo[:3, :3] = self.cal['R']
        self.stereo[:3, 3] = self.cal['T'].ravel() / 1000  # Saved stereo baseline is mm.
        self.count = count

    def triangulate(self, sample):
        obs = sample['observation']
        # Recovery needs complete correspondences; never silently reassign IDs.
        if obs['left_ids'] != list(range(self.count)) or obs['right_ids'] != list(range(self.count)):
            raise ValueError('Depth recovery requires all corner IDs in ascending order in both eyes.')
        rays = []
        for eye, k, d in [('left', 'K1', 'D1'), ('right', 'K2', 'D2')]:
            pixels = np.array(obs[eye + '_points'], dtype=float).reshape(-1, 1, 2)
            rays.append(cv2.fisheye.undistortPoints(pixels, self.cal[k], self.cal[d]).reshape(-1, 2).T)
        h = cv2.triangulatePoints(np.eye(3, 4), self.stereo[:3], *rays)
        points = (h[:3] / h[3]).T
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0) or np.any(apply(self.stereo, points)[:, 2] <= 0):
            raise ValueError('Stereo reconstruction contains invalid or behind-camera points.')
        return points

    def residual(self, v, samples):
        camera_base = np.linalg.inv(transform(v[:6]))
        points = v[6:].reshape(self.count, 3)
        errors = []
        for sample in samples:
            pose = camera_base @ np.asarray(sample['T_base_tool'])
            for eye, ep, k, d in [('left', pose, 'K1', 'D1'), ('right', self.stereo @ pose, 'K2', 'D2')]:
                ids = np.asarray(sample['observation'][eye + '_ids'])
                predicted = cv2.fisheye.projectPoints(
                    points[ids].reshape(-1, 1, 3), cv2.Rodrigues(ep[:3, :3])[0],
                    ep[:3, 3], self.cal[k], self.cal[d])[0].reshape(-1, 2)
                errors.append((predicted - np.asarray(sample['observation'][eye + '_points'])).ravel())
        return np.concatenate(errors)

    def fit(self, samples, initial_camera):
        train = [s for s in samples if s['role'] == 'training' and not s.get('excluded')]
        if len(train) < 12:
            raise ValueError('At least 12 training poses are required for this diagnostic.')
        clouds = [self.triangulate(s) for s in train]
        initial_points = np.mean([apply(np.linalg.inv(s['T_base_tool']) @ initial_camera, p)
                                  for s, p in zip(train, clouds)], axis=0)
        fit = least_squares(self.residual, np.r_[parameters(initial_camera), initial_points.ravel()],
                            args=(train,), x_scale='jac', max_nfev=200,
                            ftol=1e-10, xtol=1e-10, gtol=1e-8)
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError('Offline shape recovery did not converge.')
        return fit, np.asarray(clouds)


def recover(data):
    count = (data['board']['columns'] - 1) * (data['board']['rows'] - 1)
    engine = StereoRecovery(data['calibration'], count)
    samples = [s for s in data['samples'] if not s.get('excluded')]
    fit, clouds = engine.fit(samples, np.asarray(data['solution']['T_base_camera_left_raw']))
    x, points = transform(fit.x[:6]), fit.x[6:].reshape(count, 3)
    _, _, vt = np.linalg.svd(points - points.mean(axis=0))
    departure = (points - points.mean(axis=0)) @ vt[-1] * 1000
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    distances = np.asarray([[np.linalg.norm(p[i] - p[j]) for i, j in pairs] for p in clouds]) * 1000
    threshold = data['policy']['check_reprojection_rms_px']
    rows = []
    for sample in samples:
        residual = engine.residual(fit.x, [sample]).reshape(2, count, 2)
        rms = np.sqrt(np.mean(np.sum(residual**2, axis=2), axis=1))
        recovered = apply(np.linalg.inv(sample['T_base_tool']) @ x, engine.triangulate(sample))
        rows.append({'id': sample['id'], 'role': sample['role'],
                     'left_rms_px': float(rms[0]), 'right_rms_px': float(rms[1]),
                     'passes_pixel_check': bool(max(rms) <= threshold),
                     'corner_position_rms_mm': float(np.sqrt(np.mean(np.sum((recovered-points)**2, axis=1))) * 1000)})
    sv = np.linalg.svd(fit.jac, compute_uv=False)
    checks = [r for r in rows if r['role'] == 'check']
    report = {
        'scope': 'Offline diagnostic only; not applied; original observations and failed checks retained.',
        'method': 'Calibrated stereo triangulation, then joint native-pixel least squares of camera transform and tool-frame 3D corners.',
        'training_ids': [s['id'] for s in samples if s['role'] == 'training'],
        'held_out_check_ids': [s['id'] for s in samples if s['role'] == 'check'],
        'T_base_camera_left_raw': x.tolist(), 'points_tool_m': points.tolist(),
        'plane_departure_mm': departure.tolist(),
        'plane_rms_mm': float(np.sqrt(np.mean(departure**2))),
        'plane_peak_to_peak_mm': float(np.ptp(departure)),
        'shape_distance_std_median_mm': float(np.median(np.std(distances, axis=0))),
        'shape_distance_std_max_mm': float(np.max(np.std(distances, axis=0))),
        'shape_scope': 'Detected inner corners only; repeatability is not absolute accuracy. Assumes frozen stereo calibration and rigid attachment.',
        'jacobian_condition': float(sv[0]/sv[-1]),
        'check_threshold_px': threshold,
        'held_out_pass_count': sum(r['passes_pixel_check'] for r in checks),
        'held_out_total': len(checks), 'minimum_check_count': data['policy']['minimum_check_poses'],
        'rows': rows, 'physical_accuracy_verified': False, 'applied': False,
    }
    return report, clouds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the source run.')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve().parent):
        parser.error('Output must be outside the original run directory.')
    raw = args.run.read_bytes()
    data = json.loads(raw)
    report, clouds = recover(data)
    report.update(input_sha256=hashlib.sha256(raw).hexdigest(),
                  calibration_sha256=data['calibration_sha256'], source_run=str(args.run.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(args.output/'points.npz', points_tool_m=report['points_tool_m'], training_clouds_camera_m=clouds)
    print(json.dumps({k: report[k] for k in ['plane_rms_mm', 'plane_peak_to_peak_mm', 'held_out_pass_count', 'held_out_total', 'applied']}))


if __name__ == '__main__':
    main()
