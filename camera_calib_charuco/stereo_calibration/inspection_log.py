"""Append-only diagnostics for every processed frame, separate from fit data."""

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2


class InspectionLog:
    def __init__(self, directory, metadata, *, save_images=False):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.save_images = save_images
        if save_images:
            (self.path / "frames").mkdir()
        self.stream = (self.path / "readings.jsonl").open("x", buffering=1)
        self.frames = 0
        self.error = None
        self.write(
            {
                "type": "metadata",
                **metadata,
                "images": (
                    "Lossless PNG of each processed camera frame, at original resolution and "
                    "side-by-side ordering; no overlays, rectification or additional JPEG encoding."
                    if save_images else None
                ),
            }
        )

    @property
    def active(self):
        return not self.stream.closed and self.error is None

    def write(self, record):
        if not self.active:
            return
        try:
            self.stream.write(
                json.dumps({"utc": datetime.now(UTC).isoformat(), **record}, allow_nan=False) + "\n"
            )
            self.stream.flush()
        except (OSError, ValueError) as error:
            self.error = str(error)

    def frame(self, packet, inspection, quality):
        if not self.active:
            return
        record = {
            "type": "frame",
            "frame_id": packet.sequence,
            "camera_monotonic_seconds": packet.timestamp,
            "inspection": inspection,
            "quality": quality,
        }
        if self.save_images:
            relative = Path("frames") / f"{packet.sequence:09d}.png"
            try:
                ok, encoded = cv2.imencode(".png", packet.image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise OSError("PNG encoder returned no image")
                with (self.path / relative).open("xb") as image:
                    image.write(encoded.tobytes())
                record["raw_image"] = relative.as_posix()
            except (OSError, cv2.error) as error:
                message = f"Could not save diagnostic frame {packet.sequence}: {error}"
                self.write({"type": "recording_error", "frame_id": packet.sequence, "error": message})
                self.error = message
                return
        self.write(record)
        if self.error is None:
            self.frames += 1

    def close(self):
        if not self.stream.closed:
            self.write({"type": "end", "frames": self.frames})
            try:
                self.stream.close()
            except OSError as error:
                self.error = str(error)

    def status(self):
        return {
            "path": str(self.path),
            "active": self.active,
            "frames": self.frames,
            "save_images": self.save_images,
            "error": self.error,
        }
