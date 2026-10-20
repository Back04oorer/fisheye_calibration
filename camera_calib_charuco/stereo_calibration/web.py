"""Local HTTP interface. Camera and workflow ownership remain in the controller."""

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import PRESETS
from .workspace import Workspace

STATIC = Path(__file__).parent / "static"


@lru_cache(maxsize=64)
def capture_thumbnail(path):
    # Original captures are immutable. Decode at reduced size, off the camera thread.
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None:
        raise HTTPException(404, "Saved image not found")
    height, width = image.shape[:2]
    image = cv2.resize(image, (512, max(1, round(height * 512 / width))))
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class BoardSettings(BaseModel):
    columns: int = 11
    rows: int = 8
    square_mm: float = 20.0
    marker_mm: float = 15.0
    dictionary: str = "DICT_5X5_100"
    legacy: bool = True


class CameraSettings(BaseModel):
    device: int | str = 0
    width: int = 2560
    height: int = 720
    fps: int = 30
    swap: bool = False


class SetupRequest(BaseModel):
    board: BoardSettings
    camera: CameraSettings
    workspace_id: str


class ControlRequest(BaseModel):
    action: Literal[
        "start", "pause", "resume", "undo", "fit", "more", "reconnect", "stop_inspection",
        "stream_only", "resume_detection"
    ]
    workspace_id: str


class SessionRequest(BaseModel):
    path: str | None = None
    replay: bool = False
    workspace_id: str


class FolderRequest(BaseModel):
    kind: Literal["session", "sessions", "diagnostics", "logs"]
    path: str | None = None
    workspace_id: str


class HandEyeStart(BaseModel):
    side: Literal["left", "right"] = "right"
    workspace_id: str
    run_id: str | None = None


class HandEyeControl(BaseModel):
    action: Literal[
        "capture",
        "retry_capture",
        "exclude_failed_training",
        "solve",
        "undo",
        "pause",
        "folder",
        "auto_start",
        "auto_stop",
        "guide_next",
    ]
    workspace_id: str


def create_app(controller, port=8765):
    workspace = (
        controller
        if isinstance(controller, Workspace)
        else Workspace(output=controller.output, controller=controller)
    )

    @asynccontextmanager
    async def lifespan(_app):
        workspace.start()
        try:
            yield
        finally:
            workspace.close()

    app = FastAPI(title="Stereo calibration", lifespan=lifespan, docs_url=None, redoc_url=None)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{host}" for host in hosts}

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.headers.get("host") not in hosts or (
            request.headers.get("origin") and request.headers["origin"] not in origins
        ):
            return JSONResponse(
                {"detail": "This app accepts local, same-origin requests only."}, status_code=403
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/hand-eye")
    def handeye_page():
        return FileResponse(STATIC / "handeye.html")

    @app.get("/api/handeye/state")
    def handeye_state():
        return workspace.handeye.state()

    @app.get("/api/handeye/runs")
    def handeye_runs():
        return workspace.handeye.runs()

    @app.post("/api/handeye/start")
    def handeye_start(selection: HandEyeStart):
        try:
            return workspace.handeye.begin(selection.side, selection.workspace_id, selection.run_id)
        except (ValueError, OSError, TimeoutError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/handeye/control")
    def handeye_control(command: HandEyeControl):
        try:
            return workspace.handeye.command(command.action, command.workspace_id)
        except (ValueError, OSError, TimeoutError, cv2.error) as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/handeye/frame.jpg")
    def handeye_frame(workspace_id: str, frame_id: int):
        with workspace.access(workspace_id):
            h = workspace.handeye
            with h.lock:
                if h.revision != workspace_id or frame_id not in h.previews:
                    raise HTTPException(404, "Measured frame is no longer available.")
                return Response(
                    h.previews[frame_id],
                    media_type="image/jpeg",
                    headers={"X-Frame-Id": str(frame_id)},
                )

    @app.get("/api/handeye/board.png")
    def handeye_board():
        from .handeye import L20_BOARD

        image = L20_BOARD.create().generateImage((240, 300), marginSize=0)
        return Response(cv2.imencode(".png", image)[1].tobytes(), media_type="image/png")

    @app.get("/api/handeye/artifacts/{filename}")
    def handeye_artifact(filename: str, workspace_id: str, run_id: str):
        allowed = {"result.json", "handeye.npz", "run.json", "events.jsonl", "arm_model.json", "arm_model.urdf"}
        with workspace.access(workspace_id):
            h = workspace.handeye
            with h.lock:
                if (
                    filename not in allowed
                    or not h.path
                    or h.path.name != run_id
                    or h.revision != workspace_id
                    or not (h.path / filename).is_file()
                ):
                    raise HTTPException(404, "Artifact not available for this run.")
                return FileResponse(h.path / filename, filename=filename)

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @app.get("/api/state")
    def state():
        return workspace.state()

    @app.get("/api/sessions")
    def sessions():
        return workspace.sessions()

    @app.get("/api/folders")
    def folders():
        return workspace.folders()

    @app.post("/api/folders/open")
    def open_folder(selection: FolderRequest):
        try:
            return workspace.open_folder(
                selection.kind, path=selection.path, revision=selection.workspace_id
            )
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/sessions/open")
    def open_session(selection: SessionRequest):
        if selection.replay and not selection.path:
            raise HTTPException(422, "Select a saved session to replay.")
        try:
            return workspace.switch(
                path=selection.path, replay=selection.replay, revision=selection.workspace_id
            )
        except (ValueError, OSError, TimeoutError, KeyError, TypeError) as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/setup")
    def setup():
        devices = []
        for device in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                devices.append(
                    {"index": int(device.name[5:]), "name": (device / "name").read_text().strip()}
                )
            except (OSError, ValueError):
                continue
        return {"presets": {k: v.to_dict() for k, v in PRESETS.items()}, "devices": devices}

    @app.post("/api/configure")
    def configure(settings: SetupRequest):
        try:
            return workspace.command(
                "configure", settings.model_dump(), revision=settings.workspace_id
            )
        except (ValueError, OSError, TimeoutError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/control")
    def control(command: ControlRequest):
        try:
            return workspace.command(command.action, revision=command.workspace_id)
        except (ValueError, OSError, TimeoutError) as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/frame.jpg")
    def frame(frame_id: int, workspace_id: str, mode: Literal["raw", "rectified"] = "raw"):
        try:
            image = workspace.image(frame_id, mode, revision=workspace_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if image is None:
            raise HTTPException(404, "This frame is no longer available.")
        return Response(image, media_type="image/jpeg", headers={"X-Frame-Id": str(frame_id)})

    @app.get("/api/live.jpg")
    def live_frame(workspace_id: str, mode: Literal["raw", "rectified"] = "raw", after: int = -1):
        try:
            frame = workspace.live_image(mode, after, revision=workspace_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if frame is None:
            return Response(status_code=204)
        sequence, image, age, timestamp = frame
        return Response(
            image,
            media_type="image/jpeg",
            headers={
                "X-Frame-Id": str(sequence),
                "X-Frame-Age-Ms": f"{age * 1000:.1f}",
                "X-Frame-Timestamp-Monotonic": repr(timestamp),
            },
        )

    @app.get("/api/live.png")
    def native_frame(workspace_id: str, after: int = -1):
        """Lossless native pixels for diagnostics, in original camera stream order."""
        try:
            packet = workspace.native_frame(after, revision=workspace_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if packet is None:
            return Response(status_code=204)
        ok, encoded = cv2.imencode(".png", packet.image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise HTTPException(500, "Cannot encode the camera frame.")
        return Response(encoded.tobytes(), media_type="image/png", headers={
            "X-Frame-Id": str(packet.sequence),
            "X-Frame-Timestamp-Monotonic": repr(packet.timestamp),
            "X-Workspace-Id": workspace_id,
            "X-Image-Order": "camera-stream",
        })

    @app.get("/api/board.png")
    def board(preset: str | None = None, pattern_only: bool = False):
        if preset is not None and preset not in PRESETS:
            raise HTTPException(404, "Unknown board preset")
        config = PRESETS[preset] if preset else workspace.board
        margin = 0 if pattern_only else 20
        image = config.create().generateImage(
            (config.columns * 60 + margin * 2, config.rows * 60 + margin * 2), marginSize=margin
        )
        return Response(cv2.imencode(".png", image)[1].tobytes(), media_type="image/png")

    @app.get("/api/captures/{capture_id}")
    def saved_capture(capture_id: str, thumbnail: bool = True):
        with workspace.access() as active:
            session = active.session if active else None
            if session is None or capture_id not in {
                view["id"] for view in active.state()["saved_views"]
            }:
                raise HTTPException(404, "Capture not found")
            path = session.path / "frames" / f"{capture_id}.png"
        if thumbnail:
            return Response(capture_thumbnail(path), media_type="image/jpeg")
        return FileResponse(path, media_type="image/png", filename=path.name)

    @app.get("/api/board.pdf")
    def printable_board(preset: str = "existing-20mm"):
        from .boards import pdf

        if preset not in PRESETS:
            raise HTTPException(404, "Unknown board preset")
        return Response(
            pdf(PRESETS[preset]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="charuco-{preset}.pdf"'},
        )

    @app.get("/api/artifacts/{candidate}/{filename}")
    def artifact(candidate: str, filename: str):
        allowed = {
            "stereo_params.npz",
            "stereo_params.yaml",
            "report.json",
            "candidate.json",
            "left_fisheye.npz",
            "right_fisheye.npz",
        }
        with workspace.access() as active:
            session = active.session if active else None
            if (
                session is None
                or filename not in allowed
                or candidate not in {c["id"] for c in session.data["candidates"]}
            ):
                raise HTTPException(404, "Artifact not found")
            path = session.path / "results" / candidate / filename
        if not path.is_file():
            raise HTTPException(404, "Artifact has not been generated yet")
        return FileResponse(path, filename=filename)

    return app
