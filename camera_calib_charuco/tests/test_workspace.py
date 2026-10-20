"""Session switching must preserve data and separate camera/frame ownership."""

import json
import threading
import time
from concurrent.futures import Future
from typing import ClassVar

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stereo_calibration.camera import CameraSource, Frame
from stereo_calibration.config import BoardConfig, CameraConfig
from stereo_calibration.session import Session
from stereo_calibration.web import create_app
from stereo_calibration.workspace import Workspace


@pytest.fixture
def idle_camera(monkeypatch):
    class IdleCamera(CameraSource):
        owners: ClassVar[set] = set()
        maximum_owners = 0
        owners_lock = threading.Lock()

        def _run(self):
            with self.owners_lock:
                self.owners.add(self)
                type(self).maximum_owners = max(self.maximum_owners, len(self.owners))
            try:
                self.stopping.wait()
            finally:
                with self.owners_lock:
                    self.owners.remove(self)

    monkeypatch.setattr('stereo_calibration.controller.CameraSource', IdleCamera)
    return IdleCamera


def saved_session(path, board=None, camera=None):
    session = Session.create(path, board or BoardConfig(), camera or CameraConfig())
    session.close()
    return session.path


def test_native_snapshot_preserves_pixels_timestamp_and_workspace(tmp_path, idle_camera):
    workspace = Workspace(output=tmp_path, camera=CameraConfig(swap=True))
    with TestClient(create_app(workspace), base_url='http://127.0.0.1:8765') as client:
        params = {'workspace_id': workspace.revision}
        assert client.get('/api/live.png', params=params).status_code == 204
        workspace.switch()
        params = {'workspace_id': workspace.revision}
        active = workspace.controller
        pixels = np.random.default_rng(17).integers(0, 256, (720, 2560, 3), dtype=np.uint8)
        timestamp = time.monotonic()
        active.source.publish(Frame(42, timestamp, pixels))
        response = client.get('/api/live.png', params=params)
        assert response.status_code == 200
        assert response.headers['X-Frame-Id'] == '42'
        assert float(response.headers['X-Frame-Timestamp-Monotonic']) == timestamp
        assert response.headers['X-Workspace-Id'] == workspace.revision
        assert response.headers['X-Image-Order'] == 'camera-stream'
        decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, pixels)
        assert client.get('/api/live.png', params={**params, 'after': 42}).status_code == 204
        assert client.get('/api/live.png', params={'workspace_id': 'stale'}).status_code == 409
        assert idle_camera.maximum_owners == 1


def test_live_jpeg_timestamp_belongs_to_returned_pixels(tmp_path, idle_camera, monkeypatch):
    workspace = Workspace(output=tmp_path)
    with TestClient(create_app(workspace), base_url='http://127.0.0.1:8765') as client:
        workspace.switch()
        active=workspace.controller
        first=np.full((720,2560,3),40,np.uint8)
        later=np.full_like(first,210)
        stamp=time.monotonic()-.05
        active.source.publish(Frame(42,stamp,first))
        render=active.video.render
        def publish_while_encoding(*args,**kwargs):
            active.source.publish(Frame(43,stamp+.04,later))
            return render(*args,**kwargs)
        monkeypatch.setattr(active.video,'render',publish_while_encoding)
        response=client.get('/api/live.jpg',params={'workspace_id':workspace.revision})
        assert response.status_code==200
        assert response.headers['X-Frame-Id']=='42'
        assert float(response.headers['X-Frame-Timestamp-Monotonic'])==stamp
        decoded=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
        assert abs(float(decoded.mean())-40)<1
        assert float(response.headers['X-Frame-Age-Ms'])>=50


def test_picker_switches_saved_new_and_replay_without_overwriting_files(tmp_path, idle_camera):
    first = saved_session(tmp_path / 'first')
    second = saved_session(tmp_path / 'second', BoardConfig(columns=8, rows=6),
                           CameraConfig(width=3840, height=1200))
    snapshots = {p: (p / 'session.json').read_bytes() for p in (first, second)}
    workspace = Workspace(output=tmp_path)
    with TestClient(create_app(workspace), base_url='http://127.0.0.1:8765') as client:
        initial = client.get('/api/state').json()
        assert initial['phase'] == 'picker'
        assert not idle_camera.owners
        listed = client.get('/api/sessions').json()['sessions']
        assert {row['path'] for row in listed} == {str(first), str(second)}

        def switch(path=None, replay=False):
            revision = client.get('/api/state').json()['workspace_id']
            response = client.post('/api/sessions/open', json={
                'path': str(path) if path else None, 'replay': replay, 'workspace_id': revision,
            })
            assert response.status_code == 200, response.text
            return response.json()

        opened = switch(first)
        assert opened['session'] == str(first) and opened['paused']
        # A delayed frame request/command from another session must never match
        # a recycled frame sequence or start collection in the new session.
        workspace.controller.previews[0] = {'raw': b'old-frame'}
        changed = switch(second)
        assert changed['workspace_id'] != opened['workspace_id']
        assert changed['board']['columns'] == 8 and changed['camera']['width'] == 3840
        assert changed['saved_views'] == [] and changed['frame_id'] == -1
        assert client.get('/api/frame.jpg', params={
            'frame_id': 0, 'workspace_id': opened['workspace_id']}).status_code == 409
        assert client.get('/api/live.jpg', params={
            'workspace_id': opened['workspace_id']}).status_code == 409
        assert client.get('/api/live.jpg', params={
            'workspace_id': changed['workspace_id']}).status_code == 204
        assert client.post('/api/control', json={
            'action': 'resume', 'workspace_id': opened['workspace_id']}).status_code == 409
        assert client.post('/api/control', json={'action': 'resume'}).status_code == 422
        with pytest.raises(ValueError, match='already open'):
            Session(second)
        unlocked = Session(first)
        unlocked.close()

        new = switch()
        assert new['phase'] == 'ready' and new['session'] is None
        assert new['camera']['width'] == 3840  # Keep selected hardware mode.
        assert new['training_count'] == 0 and not new['has_model']
        assert len(list(tmp_path.glob('*/session.json'))) == 2  # No empty saved session.
        replay = switch(first, replay=True)
        assert replay['replay'] and replay['phase'] == 'replay'
        assert client.post('/api/control', json={
            'action': 'fit', 'workspace_id': replay['workspace_id']}).status_code == 409
        switch(second)
    assert not idle_camera.owners and idle_camera.maximum_owners == 1
    for path, contents in snapshots.items():
        assert (path / 'session.json').read_bytes() == contents


def test_bad_locked_or_busy_destination_preserves_active_session(tmp_path, idle_camera):
    first = saved_session(tmp_path / 'first')
    locked = Session.create(tmp_path / 'locked', BoardConfig(), CameraConfig())
    corrupt = tmp_path / 'corrupt'
    corrupt.mkdir()
    (corrupt / 'session.json').write_text('{')
    workspace = Workspace(output=tmp_path)
    workspace.start()
    try:
        opened = workspace.switch(path=first)
        current = workspace.controller
        for target in (tmp_path / 'missing', corrupt, locked.path):
            with pytest.raises((ValueError, OSError)):
                workspace.switch(path=target)
            assert workspace.controller is current
            assert workspace.state()['workspace_id'] == opened['workspace_id']
            assert not current.stop.is_set()
        assert any(row['error'] for row in workspace.sessions()['sessions'])
        current.fit_future = Future()
        with pytest.raises(ValueError, match='fit is still running'):
            workspace.switch()
        assert workspace.controller is current and not current.stop.is_set()
        current.fit_future = None
        # A fresh-session transition releases the old writer after its last work.
        workspace.switch()
        reopened = Session(first)
        reopened.close()
    finally:
        workspace.close()
        locked.close()
    assert json.loads((first / 'session.json').read_text())['captures'] == []
