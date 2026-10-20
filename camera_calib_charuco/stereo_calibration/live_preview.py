"""Render the newest camera image independently of the measurement worker."""

import threading
import time

import cv2
import numpy as np

from .guidance import POSITION_RADIUS
from .preview import Preview


class LivePreview:
    def __init__(self):
        self.lock = threading.Lock()
        self.map_key = None
        self.cache_key = None
        self.encoded = None

    def image(self, source, camera, model, target, mode, after=-1):
        with source.condition:
            packet = source.latest
        if packet is None or packet.sequence == after or source.error or source.stopping.is_set():
            return None
        with self.lock:
            key = (source, packet.sequence, camera.swap, id(model), target, mode)
            if self.cache_key != key:
                image = self.render(packet.image, camera, model, target, mode)
                self.encoded = Preview.encode(image)
                self.cache_key = key
            return (packet.sequence, self.encoded,
                    max(0, time.monotonic() - packet.timestamp), packet.timestamp)

    def render(self, image, camera, model, target, mode):
        """Resize/remap for display only; no detected corners are borrowed from old frames."""
        width, height = camera.eye_size
        size = (min(width, 800), max(1, round(height * min(width, 800) / width)))
        views = [image[:, :width], image[:, width:]]
        if camera.swap:
            views.reverse()
        if mode == 'rectified':
            if model is None:
                raise ValueError('Fit a calibration before viewing rectified video.')
            if self.map_key != (id(model), size):
                sx, sy = size[0] / width, size[1] / height
                # Match resize's pixel-centre convention and preserve the fitted field of view.
                resize = np.array([[sx, 0, (sx - 1) / 2],
                                   [0, sy, (sy - 1) / 2], [0, 0, 1]])
                self.maps = [cv2.fisheye.initUndistortRectifyMap(
                    K, D, R, resize @ P, size, cv2.CV_16SC2
                ) for K, D, R, P in (
                    (model.K1, model.D1, model.R1, model.P1),
                    (model.K2, model.D2, model.R2, model.P2),
                )]
                self.map_key = (id(model), size)
                self.map_model = model  # Retain identity while the cached maps are in use.
            views = [cv2.remap(view, *maps, cv2.INTER_LINEAR)
                     for view, maps in zip(views, self.maps)]
            for view in views:
                spacing = max(1, round(40 * size[1] / height))
                for y in range(spacing, size[1], spacing):
                    cv2.line(view, (0, y), (size[0], y), (100, 185, 110), 1)
        else:
            views = [cv2.resize(view, size, interpolation=cv2.INTER_AREA) for view in views]
            if target:
                centre = tuple(np.rint(np.asarray(target) * size).astype(int))
                axes = tuple(np.maximum(1, np.rint(np.asarray(size) * POSITION_RADIUS)).astype(int))
                overlay = views[0].copy()
                cv2.ellipse(overlay, centre, axes, 0, 0, 360, (245, 190, 30), -1)
                cv2.addWeighted(overlay, .12, views[0], .88, 0, dst=views[0])
                cv2.ellipse(views[0], centre, axes, 0, 0, 360, (245, 190, 30), 1, cv2.LINE_AA)
        return np.hstack(views)
