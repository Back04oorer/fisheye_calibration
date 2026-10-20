"""Live video must not wait for detection or misrepresent its pixel coordinates."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from stereo_calibration.camera import CameraSource, Frame
from stereo_calibration.config import BoardConfig, CameraConfig
from stereo_calibration.live_preview import LivePreview
from stereo_calibration.observations import Detector


def test_latest_video_advances_without_detection_and_preserves_raw_images():
    camera = CameraConfig(width=1920, height=1200)
    source = CameraSource(camera)
    video = LivePreview()
    raw = np.zeros((1200, 1920, 3), np.uint8)
    raw[:, :960] = (20, 50, 210)
    raw[:, 960:] = (210, 50, 20)
    original = raw.copy()
    source.publish(Frame(7, time.monotonic(), raw))
    first = video.image(source, camera, None, (.7, .7), 'raw')
    assert first[0] == 7 and first[2] >= 0
    assert video.image(source, camera, None, None, 'raw', after=7) is None
    source.publish(Frame(13, time.monotonic(), raw))
    newest = video.image(source, camera, None, None, 'raw', after=7)
    assert newest[0] == 13
    decoded = cv2.imdecode(np.frombuffer(newest[1], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (1000, 1600, 3)
    assert decoded[100, 100, 2] > decoded[100, 100, 0]
    swapped = video.render(raw, CameraConfig(width=1920, height=1200, swap=True),
                           None, None, 'raw')
    assert swapped[100, 100, 0] > swapped[100, 100, 2]
    np.testing.assert_array_equal(raw, original)
    source.close()
    assert video.image(source, camera, None, None, 'raw') is None


def test_display_rectification_preserves_projection_and_pixel_centres():
    camera = CameraConfig(width=3840, height=1200)
    K = np.array([[500., 0, 940], [0, 505., 590], [0, 0, 1]])
    D = np.array([-.01, .002, 0, 0])
    P = np.column_stack([K, [0, 0, 0]])
    model = SimpleNamespace(K1=K, K2=K, D1=D, D2=D, R1=np.eye(3), R2=np.eye(3), P1=P, P2=P)
    video = LivePreview()
    raw = np.zeros((1200, 3840, 3), np.uint8)
    output = video.render(raw, camera, model, None, 'rectified')
    assert output.shape == (500, 1600, 3)
    # Independently project selected display pixel centres back into the fisheye sensor.
    x, y = np.meshgrid(np.array([5, 200, 400, 795]), np.array([5, 240, 495]))
    display = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    original = (display + .5) / (800 / 1920) - .5
    rays = np.column_stack([(original[:, 0] - K[0, 2]) / K[0, 0],
                            (original[:, 1] - K[1, 2]) / K[1, 1]])
    expected = cv2.fisheye.distortPoints(rays.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    mx, my = cv2.convertMaps(*video.maps[0], cv2.CV_32FC1)
    actual = np.column_stack([mx[y.ravel(), x.ravel()], my[y.ravel(), x.ravel()]])
    np.testing.assert_allclose(actual, expected, atol=1 / 32)
    np.testing.assert_array_equal(model.P1, P)


@pytest.mark.parametrize('full', [False, True])
def test_concurrent_eyes_preserve_ids_coordinates_and_quality(full):
    gray = cv2.imread(str(Path(__file__).parent / 'data/charuco_marker_recovery.png'))
    frame = np.hstack([gray, np.full_like(gray, 255)])
    board = BoardConfig()
    serial = Detector(board)
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = Detector(board, executor=pool)
        for swap in (False, True):
            expected, quality = serial.observe(frame, swap, full=full)
            actual, threaded_quality = concurrent.observe(frame, swap, full=full)
            assert actual.to_dict() == expected.to_dict()
            assert quality == threaded_quality
