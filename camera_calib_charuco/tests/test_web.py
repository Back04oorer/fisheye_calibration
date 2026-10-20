import time

import numpy as np
from fastapi.testclient import TestClient
from synthetic import observations

from stereo_calibration.config import CameraConfig
from stereo_calibration.controller import Controller
from stereo_calibration.session import Session
from stereo_calibration.web import create_app


def test_replay_http_flow_is_readonly_and_serves_matching_frame(tmp_path):
    board, views, *_ = observations()
    session = Session.create(tmp_path / "recorded", board, CameraConfig())
    session.append(
        np.zeros((720, 2560, 3), np.uint8),
        views[0],
        role="training",
        frame_id=1,
        goals=[],
        signature=None,
        quality=[],
    )
    path = session.path
    session.close()
    before = (path / "session.json").read_bytes()
    controller = Controller(session=Session(path, readonly=True), replay=True)
    with TestClient(create_app(controller), base_url="http://127.0.0.1:8765") as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        deadline = time.monotonic() + 3
        while (state := client.get("/api/state").json())["frame_id"] < 0:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        response = client.get('/api/frame.jpg', params={
            'frame_id': state['frame_id'], 'workspace_id': state['workspace_id']})
        assert response.status_code == 200
        assert response.headers["X-Frame-Id"] == str(state["frame_id"])
        assert response.content.startswith(b"\xff\xd8")
        inspection = state["inspection"]
        assert inspection["frame_id"] == state["frame_id"]
        assert inspection["source"] == "recorded"
        assert len(inspection["shared_ids"]) == 70
        assert inspection["alignment"] is None  # Counts cannot imply calibration quality.
        saved = state["saved_views"]
        assert len(saved) == 1 and saved[0]["status"] == "training"
        capture_url = f"/api/captures/{saved[0]['id']}"
        assert client.get(capture_url).content.startswith(b"\xff\xd8")
        assert (
            client.get(capture_url + "?thumbnail=false").content
            == (path / "frames" / f"{saved[0]['id']}.png").read_bytes()
        )
        assert client.get("/api/captures/unknown").status_code == 404
        def control(action):
            return client.post('/api/control', json={
                'action': action, 'workspace_id': state['workspace_id']})

        assert control("pause").json()["paused"]
        assert not control("resume").json()["paused"]
        assert control("fit").status_code == 409
        board_response = client.get("/api/board.pdf?preset=a4-30mm")
        assert board_response.content.startswith(b"%PDF")
    assert (path / "session.json").read_bytes() == before
