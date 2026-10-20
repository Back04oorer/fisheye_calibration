#!/usr/bin/env python3
"""Start the local dashboard once and open it without keeping a terminal open."""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def is_running(url):
    try:
        with urllib.request.urlopen(f"{url}/api/state", timeout=1) as response:
            state = json.load(response)
        return isinstance(state, dict) and "workspace_id" in state
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("Port must be between 1 and 65535.")

    url = f"http://127.0.0.1:{args.port}"
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    log_path = output / f"gui-{args.port}.log"
    # Serialise simultaneous desktop clicks so only one server starts.
    with (output / f"gui-{args.port}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not is_running(url):
            executable = ROOT / ".venv/bin/stereo-calibrate"
            if executable.is_file():
                command = [str(executable)]
            else:
                uv = shutil.which("uv")
                if not uv:
                    local_uv = Path.home() / ".local/bin/uv"
                    uv = str(local_uv) if local_uv.is_file() else None
                if not uv:
                    raise RuntimeError("Install uv, then run uv sync --locked in the project folder.")
                command = [uv, "run", "--locked", "stereo-calibrate"]
            command += ["gui", "--no-browser", "--port", str(args.port)]
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + 60
            while not is_running(url):
                if process.poll() is not None:
                    raise RuntimeError(f"The GUI could not start. See {log_path}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"The GUI is still starting. Try again shortly. Log: {log_path}")
                time.sleep(0.25)
            print(f"Started Camera Calibration (PID {process.pid}).", flush=True)

    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        message = str(error)
        print(message, flush=True)
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", "Camera Calibration", message], check=False)
        raise SystemExit(1)
