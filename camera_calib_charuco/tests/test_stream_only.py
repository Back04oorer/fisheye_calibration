"""Stream-only suspends both detector owners, never the camera or feedback."""
import copy
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_workspace import idle_camera  # noqa: F401
from test_handeye_retry import runtime as retry_runtime  # noqa: F401

from stereo_calibration.camera import Frame
from stereo_calibration.config import BoardConfig, CameraConfig
from stereo_calibration.controller import Controller
from stereo_calibration.session import Session
from stereo_calibration.web import create_app
from stereo_calibration.workspace import Workspace


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.005)


def controller_fixture(tmp_path):
    session = Session.create(tmp_path/'session', BoardConfig(), CameraConfig(width=320, height=120))
    return Controller(session=session)


def test_http_suspends_stereo_detection_but_camera_pixels_keep_advancing(tmp_path, idle_camera, monkeypatch):
    c = controller_fixture(tmp_path)
    calls = []
    def process(packet):
        calls.append(packet.sequence)
        c.sequence = packet.sequence
        c.inspection = {'frame_id': packet.sequence}
    monkeypatch.setattr(c, 'process', process)
    before = (c.session.path/'session.json').read_bytes()
    with TestClient(create_app(c), base_url='http://127.0.0.1:8765') as client:
        revision = client.get('/api/state').json()['workspace_id']
        def command(action):
            return client.post('/api/control', json={'action': action, 'workspace_id': revision})
        image = np.zeros((120, 320, 3), np.uint8)
        c.source.publish(Frame(1, time.monotonic(), image))
        wait_until(lambda: calls == [1])
        # Ordinary pause still permits inspection, as before this change.
        assert command('pause').status_code == 200
        c.source.publish(Frame(2, time.monotonic(), image))
        wait_until(lambda: calls == [1, 2])
        state = command('stream_only').json()
        assert state['stream_only'] and state['paused']
        assert state['frame_id'] == -1 and state['inspection'] is None
        owner = c.source
        for frame in (3, 4):
            image = np.full_like(image, frame)
            timestamp = time.monotonic()
            c.source.publish(Frame(frame, timestamp, image))
            response = client.get('/api/live.png', params={'workspace_id': revision})
            assert int(response.headers['X-Frame-Id']) == frame
            assert float(response.headers['X-Frame-Timestamp-Monotonic']) == timestamp
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(response.content, np.uint8), 1), image)
        time.sleep(.08)
        assert calls == [1, 2] and c.source is owner and owner.thread.is_alive()
        assert command('resume_detection').json()['paused']
        wait_until(lambda: calls[-1] == 4)
        assert command('stream_only').status_code == 200
        state = command('resume').json()
        assert not state['stream_only'] and not state['paused']
        assert client.post('/api/control', json={'action': 'stream_only', 'workspace_id': 'old'}).status_code == 409
    assert (c.session.path/'session.json').read_bytes() == before


def test_active_handeye_suspends_without_closing_feedback_or_restoring_old_detector(tmp_path, idle_camera, monkeypatch):
    c = controller_fixture(tmp_path)
    workspace = Workspace(controller=c, output=tmp_path)
    h = workspace.handeye
    calls, stereo_calls, closed = [], [], []
    monkeypatch.setattr(c, 'process', lambda packet: stereo_calls.append(packet.sequence))
    h.controller = c
    h.active = c.handeye_active = True
    h.revision = workspace.revision
    h.feed = SimpleNamespace(status=lambda: {'connected': True, 'age_s': 0.0},
                             close=lambda: closed.append(True))
    h.data = dict(stage='complete', samples=[], side='right', solution=None, report={'passed': True})
    original = copy.deepcopy(h.data)
    def process(packet, detector):
        with h.lock:
            calls.append(packet.sequence)
            h.measurement = {'frame_id': packet.sequence, 'ready': True,
                             'measured_monotonic': time.monotonic()}
    monkeypatch.setattr(h, '_process', process)
    h.thread = threading.Thread(target=h._run, daemon=True)
    with TestClient(create_app(workspace), base_url='http://127.0.0.1:8765') as client:
        h.thread.start()
        pixels = np.zeros((120, 320, 3), np.uint8)
        c.source.publish(Frame(1, time.monotonic(), pixels))
        wait_until(lambda: calls == [1])
        response = client.post('/api/control', json={'action': 'stream_only', 'workspace_id': workspace.revision})
        assert response.status_code == 200, response.text
        assert h.active and c.handeye_active and not closed
        assert h.measurement is None and not h.pending and not h.auto_capture
        c.source.publish(Frame(2, time.monotonic(), pixels))
        time.sleep(.15)
        assert calls == [1] and stereo_calls == []
        assert h.data == original and h.feed.status()['connected']
        # Finishing the run must not silently start the other board detector.
        h.close()
        c.source.publish(Frame(3, time.monotonic(), pixels))
        time.sleep(.08)
        assert stereo_calls == [] and c.stream_only
        response = client.post('/api/control', json={'action': 'resume_detection', 'workspace_id': workspace.revision})
        assert response.status_code == 200
        wait_until(lambda: 3 in stereo_calls)


def test_suspension_acknowledges_only_after_inflight_handeye_result_is_invalidated(tmp_path, idle_camera, monkeypatch):
    c = controller_fixture(tmp_path)
    workspace = Workspace(controller=c, output=tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    h = workspace.handeye
    h.controller = c
    h.active = c.handeye_active = True
    def inflight():
        with c.detection_lock:
            entered.set()
            assert release.wait(2)
            h.measurement = {'frame_id': 1}
    worker = threading.Thread(target=inflight)
    workspace.start()
    try:
        worker.start()
        assert entered.wait(1)
        def suspend():
            workspace.command('stream_only', revision=workspace.revision)
            finished.set()
        request = threading.Thread(target=suspend)
        request.start()
        assert not finished.wait(.05)
        release.set()
        assert finished.wait(2)
        worker.join(); request.join()
        assert h.measurement is None and c.stream_only
    finally:
        release.set()
        workspace.close()


def test_constructor_stream_only_and_explicit_resume_preserve_results(tmp_path):
    c = Controller(output=tmp_path, stream_only=True)
    try:
        assert c.state()['stream_only'] and c.paused
        c._command('resume_detection', {})
        assert not c.stream_only and c.paused
        c.inspection_log = SimpleNamespace(active=True)
        with pytest.raises(ValueError, match='Stop diagnostic recording'):
            c._command('stream_only', {})
        assert not c.stream_only
    finally:
        c.inspection_log = None
        c.close()


def test_handeye_retry_explicitly_resumes_detection_and_still_requires_fresh_capture(retry_runtime):
    h, samples, clock = retry_runtime
    h.controller.stream_only = True
    def resume():
        h.controller.stream_only = False
    h.controller.resume_detection = resume
    h.suspend_detection()
    result = h.command('retry_capture', 'workspace')
    assert not h.controller.stream_only
    assert not result['capture_ready'] and result['measurement'] is None
    assert result['pending'] and not h.data['samples']


def test_stream_only_reports_camera_metadata_opened_after_initial_publication(tmp_path, monkeypatch):
    c = Controller(output=tmp_path, stream_only=True)
    try:
        assert c.state()['camera_actual'] == {}
        # Model the asynchronous camera owner's successful open. No detector
        # processing or new publication is needed to expose actual resolution.
        c.source.actual = {'backend': 'V4L2', 'width': 3840, 'height': 1200, 'fps': 30.0}
        def forbidden():
            raise AssertionError('Reading camera metadata must not republish the full calibration')
        monkeypatch.setattr(c, 'publish', forbidden)
        state = c.state()
        assert state['camera_actual'] == c.source.actual
        assert state['stream_only'] and state['frame_id'] == -1
        assert state['detection_fps'] == 0.0
        state['camera_actual']['fps'] = -1
        assert c.state()['camera_actual']['fps'] == 30.0
    finally:
        c.close()
