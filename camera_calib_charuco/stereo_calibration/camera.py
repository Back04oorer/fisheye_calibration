"""One owner per camera; consumers receive only the most recent frame."""

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float
    image: np.ndarray
    record: dict | None = None


class CameraSource:
    def __init__(self, config):
        self.config = config
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.error = None
        self.actual = {}
        self.thread = threading.Thread(target=self._run, name="camera-owner", daemon=True)

    def start(self):
        self.thread.start()

    def publish(self, frame):
        with self.condition:
            self.latest = frame
            self.condition.notify_all()

    def next(self, after=-1, timeout=0.25):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.stopping.is_set()
                    or self.error
                    or (self.latest is not None and self.latest.sequence != after)
                ),
                timeout,
            )
            return (
                self.latest if self.latest is not None and self.latest.sequence != after else None
            )

    def _run(self):
        camera = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        try:
            if not camera.isOpened():
                raise OSError(
                    "Cannot open camera. Check the selected device and close other camera apps."
                )
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.actual = {
                "backend": camera.getBackendName(),
                "fps": camera.get(cv2.CAP_PROP_FPS),
                "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            sequence = 0
            while not self.stopping.is_set():
                ok, image = camera.read()
                if not ok:
                    raise OSError(
                        "Camera disconnected or stopped producing frames. Captures are preserved."
                    )
                if image.shape[:2] != (self.config.height, self.config.width):
                    raise ValueError(
                        f"Camera returned {image.shape[1]}×{image.shape[0]}; requested "
                        f"{self.config.width}×{self.config.height}. Select a supported mode."
                    )
                self.publish(Frame(sequence, time.monotonic(), image))
                sequence += 1
        except Exception as error:
            logging.getLogger(__name__).exception("Calibration worker operation failed")
            self.error = str(error)
        finally:
            camera.release()
            with self.condition:
                self.condition.notify_all()

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError("The previous camera is still stopping. Please retry shortly.")


class ReplaySource(CameraSource):
    def __init__(self, session):
        super().__init__(session.camera)
        self.session = session
        self.playing = threading.Event()
        self.playing.set()

    def _run(self):
        captures = [x for x in self.session.data["captures"] if not x.get("undone")]
        if not captures:
            self.error = "This session contains no saved frames to replay."
            return
        sequence = 0
        while not self.stopping.is_set():
            if not self.playing.wait(0.1):
                continue
            record = captures[sequence % len(captures)]
            image = cv2.imread(str(self.session.path / record["frame"]))
            if image is None:
                self.error = f"Cannot read saved frame {record['frame']}"
                return
            self.publish(Frame(sequence, time.monotonic(), image, record))
            sequence += 1
            self.stopping.wait(1.0)
