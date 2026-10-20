"""Session discovery and serialized ownership of one active calibration workflow."""

import json
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import BoardConfig, CameraConfig
from .controller import Controller
from .folders import inspection_folders, open_directory
from .handeye_runtime import HandEye
from .session import Session


class Workspace:
    def __init__(self, board=None, camera=None, *, output="sessions", controller=None):
        self.board_defaults = board or BoardConfig()
        self.camera_defaults = camera or CameraConfig()
        self.output = Path(output).resolve()
        self.controller = controller
        self.roots = {self.output}
        if controller and controller.session:
            self.roots.add(controller.session.path.parent)
        self.lock = threading.RLock()
        self.revision = uuid.uuid4().hex
        self.started = False
        self.handeye = HandEye(self)

    def start(self):
        with self.lock:
            self.started = True
            if self.controller:
                self.controller.start()

    def close(self):
        with self.lock:
            self.handeye.close()
            if self.controller:
                self.controller.close()
                self.controller = None
            self.started = False

    @contextmanager
    def access(self, revision=None):
        with self.lock:
            if revision is not None and revision != self.revision:
                raise ValueError("Session changed. Refresh the current session before continuing.")
            yield self.controller

    @property
    def board(self):
        with self.lock:
            return self.controller.board if self.controller else self.board_defaults

    def state(self):
        with self.lock:
            value = self.controller.state() if self.controller else {
                "phase": "picker", "session": None, "error": None, "frame_id": -1,
                "board": self.board_defaults.to_dict(), "camera": self.camera_defaults.to_dict(),
            }
            value["workspace_id"] = self.revision
            return value

    def command(self, action, payload=None, *, revision=None):
        with self.access(revision) as controller:
            if action in ("stream_only", "resume_detection"):
                if controller is None:
                    raise ValueError("Choose a saved session or start a new session first.")
                controller.command(action, payload)
                if action == "stream_only":
                    self.handeye.suspend_detection()
                elif self.handeye.active:
                    with self.handeye.lock:
                        self.handeye.message = "Checking fresh board and measured arm feedback."
                return self.state()
            if self.handeye.active:
                raise ValueError("Pause hand–eye mode before changing stereo collection or camera settings.")
            if controller is None:
                raise ValueError("Choose a saved session or start a new session first.")
            controller.command(action, payload)
            return self.state()

    def image(self, frame_id, mode, *, revision=None):
        with self.access(revision) as controller:
            return controller.image(frame_id, mode) if controller else None

    def live_image(self, mode, after=-1, *, revision=None):
        with self.access(revision) as controller:
            return controller.live_image(mode, after) if controller else None

    def native_frame(self, after=-1, *, revision=None):
        """Read the existing source without changing collection or image ordering."""
        with self.access(revision) as controller:
            return controller.source.next(after=after, timeout=0) if controller else None

    def folders(self):
        with self.lock:
            active = self.controller
            session = active.session.path if active and active.session else None
            recording = active.inspection_log if active else None
            current = recording.path if recording else None
            logs = inspection_folders(session, self.output.parent / "output", current)
            return {
                "sessions": str(self.output), "sessions_exists": self.output.is_dir(),
                "session": str(session) if session else None,
                "diagnostics": str(session / "diagnostics")
                    if session and (session / "diagnostics").is_dir() else None,
                "logs": str(current) if current else logs[0]["path"] if logs else None,
            }

    def open_folder(self, kind, *, path=None, revision=None):
        with self.access(revision):
            folders = self.folders()
            target = folders[kind]
            if kind == "session" and path:
                selected = str(Path(path).expanduser().resolve())
                known = {entry["path"] for entry in self.sessions()["sessions"]}
                if selected not in known and selected != target:
                    raise ValueError("That session is not in the session picker.")
                target = selected
            elif path:
                raise ValueError("Choose a folder using its session or log control.")
            if not target:
                raise ValueError("No saved folder is available for this selection yet.")
            return {"path": str(open_directory(target))}

    def sessions(self):
        with self.lock:
            paths = {p.parent.resolve() for root in self.roots for p in root.glob("*/session.json")}
            rows = []
            for path in paths:
                try:
                    data = json.loads((path / "session.json").read_text())
                    if data["version"] != 1:
                        raise ValueError("Unsupported session format")
                    board = BoardConfig(**data["board"])
                    camera = CameraConfig(**data["camera"])
                    candidate = next((c for c in data["candidates"]
                                      if c["id"] == data["current_candidate"]), None)
                    rows.append({
                        "path": str(path), "name": path.name, "created_at": data["created_at"],
                        "phase": data["phase"], "captures": len(data["captures"]),
                        "passed": bool(candidate and (candidate.get("report") or {}).get("passed")),
                        "board": board.to_dict(), "camera": camera.to_dict(), "error": None,
                    })
                except (OSError, ValueError, KeyError, TypeError) as error:
                    rows.append({"path": str(path), "name": path.name, "created_at": "",
                                 "error": f"Cannot read session: {error}"})
            return {"sessions": sorted(rows, key=lambda r: r["created_at"], reverse=True),
                    "output": str(self.output)}

    def switch(self, *, path=None, replay=False, revision=None):
        """Prepare the destination before releasing the current writer and camera.

        A fresh controller owns all transient state (fit worker, hold, renderer,
        frame IDs and coverage). A revision prevents delayed browser requests
        from acting on or displaying another session with recycled frame IDs.
        """
        with self.access(revision) as previous:
            if self.handeye.active:
                raise ValueError("Pause hand–eye mode before switching stereo sessions.")
            target = Path(path).expanduser().resolve() if path else None
            if (previous and previous.session and target == previous.session.path
                    and replay == previous.replay):
                return self.state()
            session = None
            candidate = None
            try:
                if target:
                    session = Session(target, readonly=replay)
                candidate = Controller(
                    previous.board if previous else self.board_defaults,
                    previous.camera if previous else self.camera_defaults,
                    session=session, replay=replay, output=self.output,
                )
                if previous:
                    if not previous.stop.is_set():
                        previous.command("prepare_switch")
                    previous.close()
                self.controller = candidate
                self.revision = uuid.uuid4().hex
                if target:
                    self.roots.add(target.parent)
                if self.started:
                    candidate.start()
            except Exception:
                if candidate and self.controller is not candidate:
                    candidate.close()
                elif session and candidate is None:
                    session.close()
                raise
            return self.state()
