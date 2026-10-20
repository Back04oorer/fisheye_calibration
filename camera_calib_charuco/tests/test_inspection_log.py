import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import observations

from stereo_calibration.calibration import Calibration
from stereo_calibration.camera import Frame
from stereo_calibration.config import CameraConfig
from stereo_calibration.controller import Controller
from stereo_calibration.inspection_log import InspectionLog
from stereo_calibration.rectification import horizontal_rectification
from stereo_calibration.session import Session


@pytest.mark.parametrize("save_images", [False, True])
def test_recording_keeps_every_processed_detection_and_stops_without_mutating_session(
    tmp_path, save_images
):
    board, views, K, D, R, T = observations(1)
    view = views[0]
    model = Calibration(
        view.image_size,
        K,
        D,
        K,
        D,
        R,
        T,
        *horizontal_rectification(K, D, K, D, view.image_size, R, T),
        (0, 0, 0),
    )
    session = Session.create(tmp_path / "session", board, CameraConfig())
    session.accept_fit({"capture_ids": [], "model": model.to_dict(), "diagnostics": {}})
    session.close()
    original = (session.path / "session.json").read_bytes()
    controller = Controller(
        session=Session(session.path, readonly=True),
        replay=True,
        inspection_log="auto" if save_images else tmp_path / "inspection",
        inspection_images=save_images,
    )
    frame = np.random.default_rng(7).integers(0, 256, (720, 2560, 3), dtype=np.uint8)
    try:
        controller.process(Frame(1, 1.0, frame, {"observation": view.to_dict(), "quality": []}))
        view.left_ids = view.left_ids[:0]
        view.left_points = view.left_points[:0]
        view.right_ids = view.right_ids[:0]
        view.right_points = view.right_points[:0]
        empty = {"observation": view.to_dict(), "quality": []}
        controller.process(Frame(5, 2.0, frame, empty))
        with pytest.raises(ValueError, match="Stop diagnostic recording"):
            controller._command("more", {})
        path = controller.inspection_log.path / "readings.jsonl"
        if save_images:
            assert path.is_relative_to(session.path / "diagnostics" / "runs")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["frame_id"] for row in rows[1:]] == [1, 5]
        assert len(rows[1]["inspection"]["shared_ids"]) == 70
        assert rows[2]["inspection"]["shared_ids"] == []
        assert rows[0]["model"] == json.loads(json.dumps(model.to_dict()))
        if save_images:
            for row in rows[1:]:
                saved_image = cv2.imread(str(path.parent / row["raw_image"]))
                np.testing.assert_array_equal(saved_image, frame)
            assert len(list((path.parent / "frames").glob("*.png"))) == 2
        else:
            assert not (path.parent / "frames").exists()
            assert all("raw_image" not in row for row in rows)
        controller._command("stop_inspection", {})
        saved = path.read_bytes()
        controller.process(Frame(9, 3.0, frame, empty))
        assert controller.image(9, "raw")
        assert path.read_bytes() == saved
        assert controller.inspection_log.status()["frames"] == 2
        assert not controller.inspection_log.active
        if save_images:
            assert len(list((path.parent / "frames").glob("*.png"))) == 2
    finally:
        controller.close()
    assert (session.path / "session.json").read_bytes() == original


def test_image_storage_failure_is_visible_and_does_not_claim_a_saved_frame(tmp_path, monkeypatch):
    log = InspectionLog(tmp_path / "inspection", {}, save_images=True)
    original_open = Path.open

    def fail_image_write(path, *args, **kwargs):
        if path.suffix == ".png":
            raise OSError("No space left on device")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_image_write)
    packet = Frame(12, 3.0, np.zeros((20, 40, 3), dtype=np.uint8))
    try:
        log.frame(packet, {}, [])
        assert not log.active
        assert log.frames == 0
        assert "No space left on device" in log.status()["error"]
        rows = [json.loads(line) for line in (log.path / "readings.jsonl").read_text().splitlines()]
        assert rows[-1]["type"] == "recording_error"
        assert rows[-1]["frame_id"] == 12
        assert not any(row["type"] == "frame" for row in rows)
    finally:
        log.close()
