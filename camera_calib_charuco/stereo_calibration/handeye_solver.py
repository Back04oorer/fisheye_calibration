"""Isolated official OpenCV solver; JSON in/out, no camera or robot access."""

import json
import sys

import cv2
import numpy as np


def solve(value):
    B, C = [np.asarray(value[k], dtype=float) for k in ("B", "C")]
    if B.shape != C.shape or B.ndim != 3 or B.shape[1:] != (4, 4) or len(B) < 3:
        raise ValueError("Expected paired 4x4 transforms.")
    if not np.isfinite(B).all() or not np.isfinite(C).all():
        raise ValueError("Transforms must be finite.")
    R, t = cv2.calibrateHandEye(
        list(B[:, :3, :3]),
        list(B[:, :3, 3]),
        list(C[:, :3, :3]),
        list(C[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    return {"R": R.tolist(), "t": t.tolist(), "opencv": cv2.__version__}


def main():
    value = json.load(sys.stdin)
    result = (
        {"results": [solve(v) for v in value["batches"]]} if "batches" in value else solve(value)
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
