"""Interactively collect side-by-side stereo ChArUco images."""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_calib_charuco.calib_charuco import (
    USB_CAM_INDEX,
    create_charuco_board,
    detect_markers_robust,
    setup_detector,
)


DEFAULT_OUTPUT_DIR = "stereo_images"
DEFAULT_STEREO_WIDTH = 2560
DEFAULT_STEREO_HEIGHT = 720
DEFAULT_FPS = 30
IMAGE_PREFIX = "stereo"
IMAGE_SUFFIX = ".jpg"
WINDOW_NAME = "Stereo ChArUco Capture"


def parse_args():
    """Parse camera and output settings."""
    parser = argparse.ArgumentParser(
        description="采集左右并排的双目 ChArUco 原始图像。"
    )
    parser.add_argument("--camera", type=int, default=USB_CAM_INDEX, help="USB 相机索引")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="双目原图保存目录")
    parser.add_argument(
        "--width", type=int, default=DEFAULT_STEREO_WIDTH, help="双目图像总宽度"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_STEREO_HEIGHT, help="双目图像高度"
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="采集帧率")
    parser.add_argument(
        "--target", type=int, default=None, help="可选的参考图像数量，不代表标定质量达标"
    )
    parser.add_argument(
        "--allow-undetected",
        action="store_true",
        help="允许保存未在左右两侧同时检测到标定板的图像",
    )
    return parser.parse_args()


def capture_pattern():
    """Return the regular expression used for managed stereo images."""
    return re.compile(
        rf"^{re.escape(IMAGE_PREFIX)}_(\d+){re.escape(IMAGE_SUFFIX)}$",
        re.IGNORECASE,
    )


def list_capture_files(output_dir):
    """Return managed stereo images sorted by numeric index."""
    pattern = capture_pattern()
    indexed_paths = []
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if path.is_file() and match:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort(key=lambda item: item[0])
    return indexed_paths


def next_capture_index(output_dir):
    """Return the next stereo image index."""
    captures = list_capture_files(output_dir)
    return captures[-1][0] + 1 if captures else 0


def save_frame(output_dir, frame):
    """Save one unmodified stereo frame and return its path."""
    index = next_capture_index(output_dir)
    path = output_dir / f"{IMAGE_PREFIX}_{index:03d}{IMAGE_SUFFIX}"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"无法写入图像: {path}")
    return path


def delete_last_capture(output_dir):
    """Delete and return the latest managed stereo image."""
    captures = list_capture_files(output_dir)
    if not captures:
        return None
    path = captures[-1][1]
    path.unlink()
    return path


def split_stereo_frame(frame):
    """Split an even-width side-by-side frame into left and right views."""
    width = frame.shape[1]
    if width < 2 or width % 2:
        raise ValueError(f"双目图像宽度必须是大于 1 的偶数，实际为 {width}")
    split_x = width // 2
    return frame[:, :split_x], frame[:, split_x:]


def detect_view(gray, detector, clahe, clahe_strong, marker_3d, board):
    """Detect the ChArUco markers in one camera view."""
    _, _, corners, ids = detect_markers_robust(
        gray, detector, clahe, clahe_strong, marker_3d, board
    )
    return corners, ids


def build_preview(left, right, left_detection, right_detection):
    """Build an annotated preview without modifying the saved frame."""
    left_preview = left.copy()
    right_preview = right.copy()
    left_corners, left_ids = left_detection
    right_corners, right_ids = right_detection
    if left_ids is not None:
        cv2.aruco.drawDetectedMarkers(left_preview, left_corners, left_ids)
    if right_ids is not None:
        cv2.aruco.drawDetectedMarkers(right_preview, right_corners, right_ids)
    preview = np.hstack((left_preview, right_preview))
    center_x = left_preview.shape[1]
    cv2.line(preview, (center_x, 0), (center_x, preview.shape[0]), (0, 255, 255), 2)
    return preview


def draw_status(preview, left_ids, right_ids, saved_count, target, message):
    """Draw detection state, image count, controls, and feedback."""
    left_ok = left_ids is not None
    right_ok = right_ids is not None
    both_ok = left_ok and right_ok
    status = f"LEFT: {'OK' if left_ok else 'FAILED'} | RIGHT: {'OK' if right_ok else 'FAILED'}"
    color = (0, 200, 0) if both_ok else (0, 0, 255)
    cv2.putText(
        preview,
        f"{status} | SAVED: {saved_count}" + (f" (reference: {target})" if target else ""),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    cv2.putText(
        preview,
        "[s] Save stereo frame  [d] Delete last  [ESC/q] Quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    if message:
        cv2.putText(
            preview,
            message,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
        )


def run_capture(args):
    """Run the interactive stereo capture loop."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    board, _, marker_3d = create_charuco_board()
    detector, clahe, clahe_strong = setup_detector()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        cap.release()
        print(f"错误：无法打开 USB 相机 {args.camera}")
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"相机: {args.camera} | 格式: {actual_fourcc} | "
        f"实际分辨率: {actual_width}x{actual_height} @ {actual_fps:g} fps"
    )
    if (actual_width, actual_height) != (args.width, args.height):
        cap.release()
        print(
            f"错误：相机未接受 {args.width}x{args.height} 请求，"
            f"实际为 {actual_width}x{actual_height}"
        )
        return 1
    print(f"双目原图目录: {output_dir}")
    print("按 s 保存双目原图，按 d 删除最近一张，按 ESC 或 q 退出。")
    print("图像数量不代表标定完成。自动采集及独立质量验证请使用 stereo-calibrate gui。")
    message = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("错误：相机读取失败")
                return 1
            try:
                left, right = split_stereo_frame(frame)
            except ValueError as error:
                print(f"错误：{error}")
                return 1

            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            left_detection = detect_view(
                left_gray, detector, clahe, clahe_strong, marker_3d, board
            )
            right_detection = detect_view(
                right_gray, detector, clahe, clahe_strong, marker_3d, board
            )
            left_ids = left_detection[1]
            right_ids = right_detection[1]
            both_detected = left_ids is not None and right_ids is not None

            preview = build_preview(left, right, left_detection, right_detection)
            saved_count = len(list_capture_files(output_dir))
            draw_status(
                preview, left_ids, right_ids, saved_count, args.target, message
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            message = ""

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                if not both_detected and not args.allow_undetected:
                    message = "NOT SAVED: board must be detected in LEFT and RIGHT"
                    continue
                try:
                    path = save_frame(output_dir, frame)
                except OSError as error:
                    message = f"SAVE FAILED: {error}"
                    print(f"错误：{error}")
                else:
                    message = f"SAVED: {path.name}"
                    print(f"已保存: {path}")
            elif key == ord("d"):
                try:
                    path = delete_last_capture(output_dir)
                except OSError as error:
                    message = f"DELETE FAILED: {error}"
                    print(f"错误：{error}")
                else:
                    if path is None:
                        message = "NOTHING TO DELETE"
                    else:
                        message = f"DELETED: {path.name}"
                        print(f"已删除: {path}")
    except KeyboardInterrupt:
        print("\n收到中断，结束采集。")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"采集结束，共保留 {len(list_capture_files(output_dir))} 张双目图像。")
    return 0


def main():
    """Run the command-line entry point."""
    return run_capture(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
