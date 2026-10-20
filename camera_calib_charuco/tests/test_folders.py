"""Folder navigation preserves the selected workflow and locates retained logs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from stereo_calibration.config import BoardConfig, CameraConfig
from stereo_calibration.folders import open_directory
from stereo_calibration.session import Session
from stereo_calibration.web import create_app
from stereo_calibration.workspace import Workspace


def test_folder_navigation_and_migrated_logs_do_not_switch_sessions(tmp_path, monkeypatch):
    root = tmp_path / 'sessions'
    session = Session.create(root / 'saved', BoardConfig(), CameraConfig())
    path = session.path
    session.close()
    original = (path / 'session.json').read_bytes()
    logs = path / 'diagnostics' / 'audit' / 'run'
    logs.mkdir(parents=True)
    # A self-contained session may have been moved since its log was recorded.
    (logs / 'readings.jsonl').write_text(json.dumps({
        'type': 'metadata', 'session': '/old/location', 'utc': '2026-10-10T12:00:00Z',
    }) + '\n')
    unrelated = tmp_path / 'output' / 'other-session'
    unrelated.mkdir(parents=True)
    (unrelated / 'readings.jsonl').write_text(json.dumps({
        'type': 'metadata', 'session': '/other-session', 'utc': '2026-10-11T12:00:00Z',
    }))
    corrupt = path / 'diagnostics' / 'interrupted'
    corrupt.mkdir()
    (corrupt / 'readings.jsonl').write_text('null\n')
    opened = []
    monkeypatch.setattr('stereo_calibration.workspace.open_directory',
                        lambda target: opened.append(Path(target)) or target)
    workspace = Workspace(output=root)
    with TestClient(create_app(workspace), base_url='http://127.0.0.1:8765') as client:
        revision = workspace.revision
        response = client.post('/api/folders/open', json={
            'kind': 'session', 'path': str(path), 'workspace_id': revision,
        })
        assert response.status_code == 200 and opened == [path]
        assert workspace.controller is None and workspace.revision == revision
        assert client.post('/api/folders/open', json={
            'kind': 'session', 'path': str(tmp_path), 'workspace_id': revision,
        }).status_code == 409
        state = workspace.switch(path=path, replay=True)
        assert client.get('/api/folders').json()['logs'] == str(logs)
        assert client.post('/api/folders/open', json={
            'kind': 'logs', 'workspace_id': revision,
        }).status_code == 409
        assert client.post('/api/folders/open', json={
            'kind': 'logs', 'workspace_id': state['workspace_id'],
        }).status_code == 200
        assert opened[-1] == logs
        assert workspace.state()['workspace_id'] == state['workspace_id']
    assert (path / 'session.json').read_bytes() == original


def test_desktop_opener_handles_paths_and_reports_failure(tmp_path, monkeypatch):
    path = tmp_path / 'board logs $(literal)'
    path.mkdir()
    calls = []
    monkeypatch.setattr('stereo_calibration.folders.shutil.which', lambda name: '/usr/bin/gio')

    def run(command, **options):
        calls.append(command)
        assert 'shell' not in options
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr('stereo_calibration.folders.subprocess.run', run)
    assert open_directory(path) == path
    assert calls == [['/usr/bin/gio', 'open', path.as_uri()]]
    monkeypatch.setattr('stereo_calibration.folders.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=1, stderr='No display'))
    with pytest.raises(OSError, match='No display'):
        open_directory(path)
    with pytest.raises(ValueError, match='does not exist'):
        open_directory(tmp_path / 'missing')
