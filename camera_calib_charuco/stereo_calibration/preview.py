"""Preview rendering is presentation only; saved images are never annotated or resized."""

import cv2
import numpy as np

from .guidance import POSITION_RADIUS


class Preview:
    def __init__(self):
        self.model = None
        self.maps = None

    def render(self, frame, observation, signature, goal, camera, model=None):
        width = frame.shape[1] // 2
        views = [frame[:, :width].copy(), frame[:, width:].copy()]
        if camera.swap:
            views.reverse()
        raw = [view.copy() for view in views]
        if goal is not None and goal.target:
            # Goals are evaluated in the left image. Equal pixels in the two images
            # do not identify a common physical position, so do not duplicate this cue.
            target = tuple(np.rint(np.asarray(goal.target) * camera.eye_size).astype(int))
            axes = tuple(
                np.maximum(1, np.rint(np.asarray(camera.eye_size) * POSITION_RADIUS)).astype(int)
            )
            overlay = raw[0].copy()
            cv2.ellipse(overlay, target, axes, 0, 0, 360, (245, 190, 30), -1)
            cv2.addWeighted(overlay, 0.12, raw[0], 0.88, 0, dst=raw[0])
            cv2.ellipse(raw[0], target, axes, 0, 0, 360, (245, 190, 30), 3, cv2.LINE_AA)
        common, _, _ = observation.common()
        for view, eye in zip(raw, ("left", "right")):
            for point, corner_id in zip(
                getattr(observation, f"{eye}_points"), getattr(observation, f"{eye}_ids")
            ):
                colour = (120, 215, 70) if corner_id in common else (0, 190, 255)
                cv2.circle(view, tuple(np.rint(point).astype(int)), 4, colour, 2)
        if signature:
            polygon = np.rint(signature["outline"]).astype(np.int32)
            if len(polygon):
                cv2.polylines(raw[0], [polygon], True, (240, 195, 60), 2)
            centre = tuple(
                np.rint(np.asarray([signature["x"], signature["y"]]) * camera.eye_size).astype(int)
            )
            cv2.drawMarker(raw[0], centre, (255, 255, 255), cv2.MARKER_CROSS, 32, 3, cv2.LINE_AA)
        result = {"raw": self.encode(np.hstack(raw))}
        if model is not None:
            if self.model is not model:
                self.maps = [
                    cv2.fisheye.initUndistortRectifyMap(K, D, R, P, camera.eye_size, cv2.CV_16SC2)
                    for K, D, R, P in (
                        (model.K1, model.D1, model.R1, model.P1),
                        (model.K2, model.D2, model.R2, model.P2),
                    )
                ]
                self.model = model
            corrected = [
                cv2.remap(view, *maps, cv2.INTER_LINEAR) for view, maps in zip(views, self.maps)
            ]
            for view in corrected:
                for y in range(40, view.shape[0], 40):
                    cv2.line(view, (0, y), (view.shape[1], y), (100, 185, 110), 1)
            result["rectified"] = self.encode(np.hstack(corrected))
        return result

    @staticmethod
    def encode(image):
        if image.shape[1] > 1600:
            image = cv2.resize(image, (1600, round(image.shape[0] * 1600 / image.shape[1])))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise OSError("Preview encoding failed.")
        return encoded.tobytes()
