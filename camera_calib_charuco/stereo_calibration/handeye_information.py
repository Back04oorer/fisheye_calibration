"""Expected stereo information for advisory view selection, not robot planning.

The fixed-rig X/Z reprojection Jacobian is evaluated in virtual camera/board
frames (X=Z=I, G=C). These are constant changes of base/tool coordinates for a
consistent rigid setup. Log-determinant information gains are invariant under
the corresponding nonsingular parameter basis change. A fixed tiny ridge is
used only for numerical rank deficiency during bootstrap. Pixel noise is set
to one; robot/model error is deliberately NOT claimed to be covered by this
conditional ranking. See Yang, Rebello & Waslander (2023), arXiv:2303.06766.
"""

import cv2
import numpy as np

from .handeye import L20_BOARD, checked_transform


def view_information(pose, model):
    C = checked_transform(pose)
    obj = L20_BOARD.create().getChessboardCorners().astype(float) / 1000
    camera = obj @ C[:3, :3].T + C[:3, 3]
    rig_t = np.asarray(model.T).reshape(3) / 1000

    def pixels(points):
        views = []
        for p, K, D in (
            (points, model.K1, model.D1),
            (points @ model.R.T + rig_t, model.K2, model.D2),
        ):
            if p[:, 2].min() <= 0:
                raise ValueError("Information candidate lies behind a camera.")
            views.append(
                cv2.fisheye.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[
                    0
                ].ravel()
            )
        return np.concatenate(views)

    # Dimensionless parameters: radians and translations in units of 0.1 m.
    # Centre differences avoid a numerical preference for a perturbation sign.
    columns = []
    step = 1e-5
    for parameter in range(12):
        axis = np.eye(3)[parameter % 3]
        pair = []
        for sign in (1, -1):
            if parameter < 3:
                p = camera @ cv2.Rodrigues(-sign * step * axis)[0].T
            elif parameter < 6:
                p = camera - sign * step * 0.1 * axis
            elif parameter < 9:
                p = (obj @ cv2.Rodrigues(sign * step * axis)[0].T) @ C[:3, :3].T + C[:3, 3]
            else:
                p = (obj + sign * step * 0.1 * axis) @ C[:3, :3].T + C[:3, 3]
            pair.append(pixels(p))
        columns.append((pair[0] - pair[1]) / (2 * step))
    J = np.column_stack(columns)
    return J.T @ J


class ViewInformation:
    """One fixed prior per suggestion; repeated polling never adds evidence."""

    def __init__(self, poses, model):
        self.model = model
        self.prior = sum((view_information(pose, model) for pose in poses), np.zeros((12, 12)))
        self.prior += np.eye(12) * 1e-6
        self.logdet = np.linalg.slogdet(self.prior)[1]

    def gain(self, pose):
        total = self.prior + view_information(pose, self.model)
        sign, logdet = np.linalg.slogdet(total)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Invalid view-information matrix.")
        return float((logdet - self.logdet) / 2)
