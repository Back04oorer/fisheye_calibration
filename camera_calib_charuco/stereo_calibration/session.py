"""Atomic session persistence, single-writer ownership, and candidate provenance."""

import fcntl
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, export, fit
from .config import BoardConfig, CameraConfig
from .observations import Observation
from .verification import evaluate_saved_checks


def now():
    return datetime.now(UTC).isoformat()


def atomic_bytes(path: Path, content: bytes):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, indent=2, allow_nan=False).encode())


class Session:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        self.readonly = readonly
        self._lock = None
        if not readonly:
            self._lock = (self.path / ".writer.lock").open("a")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self._lock.close()
                raise ValueError(
                    "This session is already open in another calibration process."
                ) from error
        try:
            self.data = json.loads((self.path / "session.json").read_text())
            if self.data["version"] != 1:
                raise ValueError("Unsupported session format.")
            self.board = BoardConfig(**self.data["board"])
            self.camera = CameraConfig(**self.data["camera"])
        except Exception:
            self.close()
            raise

    @classmethod
    def create(cls, path, board, camera):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        (path / "frames").mkdir()
        data = {
            "version": 1,
            "created_at": now(),
            "board": board.to_dict(),
            "camera": camera.to_dict(),
            "phase": "ready",
            "current_candidate": None,
            "captures": [],
            "candidates": [],
            "events": [],
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        }
        atomic_json(path / "session.json", data)
        return cls(path)

    def close(self):
        if self._lock is not None and not self._lock.closed:
            self._lock.close()

    def save(self):
        if self.readonly:
            raise ValueError("Replay sessions are read-only.")
        atomic_json(self.path / "session.json", self.data)

    def phase(self, name):
        old = self.data["phase"]
        self.data["phase"] = name
        try:
            self.save()
        except Exception:
            self.data["phase"] = old
            raise

    def captures(self, role):
        # A check keeps its original capture-candidate ID. Reuse images from this
        # candidate and its predecessors, never results from another model.
        candidates = set()
        for candidate in self.data["candidates"]:
            candidates.add(candidate["id"])
            if candidate["id"] == self.data["current_candidate"]:
                break
        if self.candidate is None:
            candidates.clear()
        return [
            c
            for c in self.data["captures"]
            if c["role"] == role
            and not c.get("undone")
            and (role == "training" or c["candidate"] in candidates)
        ]

    def observations(self, role):
        return [Observation.from_dict(c["observation"]) for c in self.captures(role)]

    def saved_views(self):
        """Small, durable gallery records, including checks from previous fits."""
        candidate = self.candidate
        if candidate is None and self.data.get("training_request"):
            candidate = next(reversed(self.data["candidates"]), None)
        used = set(candidate["training_captures"]) if candidate else set()
        results = {
            view["name"]: "passed" if view["passed"] else "failed"
            for fitted in self.data["candidates"]
            for view in (fitted.get("report") or {}).get("validation_views", [])
        }
        return [
            {
                "id": capture["id"],
                "role": capture["role"],
                "status": "excluded"
                if capture.get("undone")
                else "used"
                if capture["id"] in used
                else "training"
                if capture["role"] == "training"
                else results.get(capture["id"], "pending"),
            }
            for capture in self.data["captures"]
        ]

    def append(self, frame, observation, *, role, frame_id, goals, signature, quality):
        if self.readonly:
            raise ValueError("Cannot capture in replay mode.")
        if role not in ("training", "validation"):
            raise ValueError("Unknown capture role.")
        if role == "validation" and self.candidate is None:
            raise ValueError("Validation requires a frozen candidate first.")
        if role == "training" and self.candidate is not None:
            raise ValueError("Start another training round before changing a frozen candidate.")
        if observation.image_size != self.camera.eye_size or frame.shape[:2] != (
            self.camera.height,
            self.camera.width,
        ):
            raise ValueError(
                "Capture resolution changed. Start a new session for another camera mode."
            )
        key = uuid.uuid4().hex[:12]
        relative = f"frames/{key}.png"
        observation.name = key
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise OSError("Could not encode the capture.")
        record = {
            "id": key,
            "frame": relative,
            "frame_id": frame_id,
            "captured_at": now(),
            "role": role,
            "candidate": self.data["current_candidate"],
            "goals": goals,
            "signature": signature,
            "quality": quality,
            "observation": observation.to_dict(),
        }
        atomic_bytes(self.path / relative, encoded.tobytes())
        self.data["captures"].append(record)
        try:
            self.save()
        except Exception:
            self.data["captures"].pop()
            (self.path / relative).unlink(missing_ok=True)
            raise
        return record

    @property
    def candidate(self):
        return next(
            (c for c in self.data["candidates"] if c["id"] == self.data["current_candidate"]), None
        )

    def training_snapshot(self):
        return {
            "board": self.board.to_dict(),
            "capture_ids": [c["id"] for c in self.captures("training")],
            "observations": [c["observation"] for c in self.captures("training")],
        }

    def accept_fit(self, result):
        if self.readonly:
            raise ValueError("Replay sessions are read-only.")
        if result["capture_ids"] != self.training_snapshot()["capture_ids"]:
            raise ValueError("Training changed while fitting; the stale result was discarded.")
        model = Calibration.from_dict(result["model"])
        key = uuid.uuid4().hex[:12]
        candidate = {
            "id": key,
            "created_at": now(),
            "training_captures": result["capture_ids"],
            "model": result["model"],
            "diagnostics": result["diagnostics"],
            "report": None,
            "required_fresh_goals": [
                f"check-{key}"
                for key in (self.data.get("training_request") or {}).get("goals", [])
                if key != "near" and not key.startswith("tilt-")
            ],
        }
        export(model, self.path / "results" / key)
        atomic_json(self.path / "results" / key / "candidate.json", candidate)
        old = self.data["current_candidate"]
        request = self.data.pop("training_request", None)
        self.data["candidates"].append(candidate)
        self.data["current_candidate"] = key
        try:
            self.save()
        except Exception:
            self.data["candidates"].pop()
            self.data["current_candidate"] = old
            if request is not None:
                self.data["training_request"] = request
            raise

    def validate(self):
        if self.readonly:
            raise ValueError("Replay sessions are read-only.")
        candidate = self.candidate
        if candidate is None:
            raise ValueError("Fit a calibration before validation.")
        captures = self.captures("validation")
        if set(candidate["training_captures"]).intersection(c["id"] for c in captures):
            raise ValueError("A check image cannot also belong to the fit.")
        report = evaluate_saved_checks(
            captures,
            Calibration.from_dict(candidate["model"]),
            self.board,
            candidate["id"],
            candidate.get("required_fresh_goals", []),
        )
        report.update(
            candidate=candidate["id"], board=self.board.to_dict(), camera=self.camera.to_dict()
        )
        atomic_json(self.path / "results" / candidate["id"] / "report.json", report)
        old = candidate["report"]
        candidate["report"] = report
        try:
            self.save()
        except Exception:
            candidate["report"] = old
            raise
        return report

    def new_round(self, goal_keys=(), *, reason=None):
        if self.readonly:
            raise ValueError("Replay sessions are read-only.")
        old_candidate, old_phase = self.data["current_candidate"], self.data["phase"]
        old_request = self.data.get("training_request")
        self.data["training_request"] = {
            "goals": list(goal_keys),
            "existing_captures": [c["id"] for c in self.captures("training")],
            "reason": reason,
        }
        self.data["current_candidate"] = None
        self.data["phase"] = "collecting"
        self.data["events"].append(
            {
                "at": now(),
                "event": "additional_training",
                "candidate": old_candidate,
                "goals": list(goal_keys),
                "reason": reason,
            }
        )
        try:
            self.save()
        except Exception:
            self.data["current_candidate"], self.data["phase"] = old_candidate, old_phase
            if old_request is None:
                self.data.pop("training_request", None)
            else:
                self.data["training_request"] = old_request
            self.data["events"].pop()
            raise

    def undo(self):
        if self.readonly:
            raise ValueError("Replay sessions are read-only.")
        if self.candidate:
            raise ValueError("Start another training round before undoing training data.")
        captures = self.captures("training")
        if captures:
            captures[-1]["undone"] = True
            try:
                self.save()
            except Exception:
                captures[-1].pop("undone")
                raise


def fit_snapshot(snapshot):
    """Process-worker entry point: its input contains training observations only."""
    model, diagnostics = fit(
        [Observation.from_dict(x) for x in snapshot["observations"]],
        BoardConfig(**snapshot["board"]),
    )
    return {
        "capture_ids": snapshot["capture_ids"],
        "model": model.to_dict(),
        "diagnostics": diagnostics,
    }
