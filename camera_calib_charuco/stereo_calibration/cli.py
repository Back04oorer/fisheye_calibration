"""Command-line entry point for the shared calibration package."""

import argparse
import json

from . import __version__


def main():
    parser = argparse.ArgumentParser(description="Guided ChArUco fisheye stereo calibration")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "validate"):
        command = commands.add_parser(name)
        command.add_argument("session", help="Recorded session directory")
    gui = commands.add_parser("gui", help="Open the local guided calibration dashboard")
    source = gui.add_mutually_exclusive_group()
    source.add_argument("--session", help="Resume an existing session, initially paused")
    source.add_argument("--replay", help="Read-only replay of saved images and results")
    gui.add_argument("--camera", default="0", help="Video index or /dev/v4l/by-id device path")
    gui.add_argument("--width", type=int, default=2560)
    gui.add_argument("--height", type=int, default=720)
    gui.add_argument("--fps", type=int, default=30)
    gui.add_argument("--board", choices=("existing-20mm", "a4-30mm"), default="existing-20mm")
    gui.add_argument("--output", default="sessions", help="Directory for new sessions")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true")
    gui.add_argument("--stream-only", action="store_true",
                     help="Keep live camera video without running board detection; capture starts paused")
    gui.add_argument(
        "--inspection-log",
        nargs="?", const="auto",
        help="Record under SESSION/diagnostics/runs (or an explicit new directory)",
    )
    gui.add_argument(
        "--inspection-images",
        action="store_true",
        help="Also save full-resolution lossless PNGs for each reading; requires --inspection-log",
    )
    board_parser = commands.add_parser("board", help="Generate a true-size A4 ChArUco PDF")
    board_parser.add_argument(
        "--preset", choices=("existing-20mm", "a4-30mm"), default="existing-20mm"
    )
    board_parser.add_argument(
        "--output", help="Output PDF path (default: output/pdf/charuco-PRESET.pdf)"
    )
    args = parser.parse_args()
    from .session import Session, fit_snapshot

    if args.command == "board":
        from pathlib import Path

        from .boards import pdf
        from .config import PRESETS

        path = Path(args.output or f"output/pdf/charuco-{args.preset}.pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf(PRESETS[args.preset]))
        print(path.resolve())
        return

    if args.command == "gui":
        import threading
        import webbrowser

        import uvicorn

        from .config import PRESETS, CameraConfig
        from .controller import Controller
        from .web import create_app
        from .workspace import Workspace

        if not 0 < args.port < 65536:
            parser.error("Port must be between 1 and 65535.")
        if args.inspection_images and not args.inspection_log:
            parser.error("--inspection-images requires --inspection-log.")
        if args.stream_only and args.inspection_log:
            parser.error("--stream-only cannot record board inspection; omit --inspection-log.")
        if args.stream_only and not (args.session or args.replay):
            parser.error("--stream-only requires --session or --replay.")
        session = (
            Session(args.session or args.replay, readonly=bool(args.replay))
            if args.session or args.replay
            else None
        )
        device = int(args.camera) if args.camera.isdigit() else args.camera
        if args.inspection_log and session is None:
            parser.error("--inspection-log requires --session or --replay.")
        board = PRESETS[args.board]
        camera = CameraConfig(device, args.width, args.height, args.fps)
        controller = Controller(
            board, camera, session=session, replay=bool(args.replay), output=args.output,
            inspection_log=args.inspection_log, inspection_images=args.inspection_images,
            stream_only=args.stream_only,
        ) if session else None
        workspace = Workspace(
            board, camera, output=args.output, controller=controller,
        )
        url = f"http://127.0.0.1:{args.port}"
        print(f"Stereo calibration: {url}", flush=True)
        if not args.no_browser:
            opener = threading.Timer(1.0, webbrowser.open, args=(url,))
            opener.daemon = True
            opener.start()
        try:
            uvicorn.run(
                create_app(workspace, args.port),
                host="127.0.0.1",
                port=args.port,
                log_level="warning",
            )
        finally:
            workspace.close()
        return

    session = Session(args.session)
    try:
        if args.command == "calibrate":
            session.accept_fit(fit_snapshot(session.training_snapshot()))
            report = session.validate()
            session.phase("done" if report["passed"] else "validating")
            print(f"Calibration frozen: {session.path / 'results' / session.candidate['id']}")
        elif args.command == "validate":
            if session.candidate is None:
                parser.error("Fit this session before validating it.")
            print(json.dumps(session.validate(), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    main()
