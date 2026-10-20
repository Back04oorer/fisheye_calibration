"""Bounded, empirical raw-joint correction for static FK. No motor commands."""
import xml.etree.ElementTree as ET
import numpy as np

# Nero joint feedback is quantized in 0.001 degrees. Observed limits come from
# stationary frame averages, so an endpoint reading may differ by one tick.
# This only covers representation rounding; never clamp or alter raw angles.
FEEDBACK_ROUNDING_RAD = float(np.radians(.001))


def checked_refinement(spec, urdf):
    if not isinstance(spec, dict) or spec.get('model') != 'raw_joint_polynomial_v1':
        raise ValueError('Unsupported wrist refinement')
    joints = {j.get('name') for j in ET.fromstring(urdf).findall('joint')
              if j.get('type') in ('revolute', 'continuous')}
    names = spec.get('joint_names')
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not set(names) <= joints or spec.get('target_joint') not in joints):
        raise ValueError('Invalid refinement joints')
    result = {k: spec[k] for k in ('model', 'joint_names', 'target_joint')}
    shapes = {'reference_rad': (len(names),), 'scale_rad': (len(names),),
              'observed_range_rad': (len(names), 2)}
    for key, shape in shapes.items():
        value = np.asarray(spec.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid finite refinement ' + key)
        result[key] = value.tolist()
    if np.any(np.array(result['scale_rad']) <= 0):
        raise ValueError('Refinement scales must be positive')
    limits = np.array(result['observed_range_rad'])
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Invalid refinement interval')
    powers = np.asarray(spec.get('powers'), dtype=float)
    coefficients = np.asarray(spec.get('coefficients_rad'), dtype=float)
    if (powers.ndim != 2 or powers.shape[1] != len(names) or not len(powers)
            or coefficients.shape != (len(powers),) or not np.isfinite(coefficients).all()
            or not np.isfinite(powers).all() or np.any(powers != np.floor(powers))
            or np.any(powers < 0) or np.any(powers.sum(1) > 4)):
        raise ValueError('Invalid refinement polynomial')
    result.update(powers=powers.astype(int).tolist(), coefficients_rad=coefficients.tolist())
    return result


def refinement_delta(spec, raw):
    try:
        q = np.array([raw[n] for n in spec['joint_names']], dtype=float)
    except KeyError as error:
        raise ValueError('Wrist refinement needs complete raw joint feedback') from error
    limits = np.asarray(spec['observed_range_rad'])
    tolerance = FEEDBACK_ROUNDING_RAD + 1e-12
    if (not np.isfinite(q).all() or np.any(q < limits[:, 0]-tolerance)
            or np.any(q > limits[:, 1]+tolerance)):
        raise ValueError('Outside the observed wrist-refinement workspace; no extrapolation')
    x = (q-np.array(spec['reference_rad']))/np.array(spec['scale_rad'])
    return float(np.prod(x[None, :] ** np.array(spec['powers']), axis=1)
                 @ np.array(spec['coefficients_rad']))
