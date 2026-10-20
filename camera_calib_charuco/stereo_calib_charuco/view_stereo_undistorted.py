"""Display and validate corrected side-by-side stereo video."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_calib_charuco.calib_charuco import setup_detector
from stereo_calibration.config import PRESETS
from stereo_calibration.detection import setup_charuco_detector, detect_charuco_robust


DEFAULT_CAMERA_INDEX = 0
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"
DEFAULT_MODE = "rectify"
DEFAULT_GUIDE_SPACING = 40
DEFAULT_MIN_COMMON_MARKERS = 6
DEFAULT_OUTPUT_DIR = "undistorted_captures"
PASS_P95_PX = 2.0
WARNING_P95_PX = 3.0
WINDOW_NAME = "Stereo Corrected Viewer"
REQUIRED_PARAMETER_SHAPES = {
    "K1": (3, 3),
    "K2": (3, 3),
    "R1": (3, 3),
    "R2": (3, 3),
    "P1": (3, 4),
    "P2": (3, 4),
}


@dataclass(frozen=True)
class StereoParameters:
    """Camera and rectification matrices loaded from stereo calibration."""

    image_size: Tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    model: str = "pinhole"


@dataclass(frozen=True)
class RemapPair:
    """Cached OpenCV remap arrays for the left and right views."""

    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray


@dataclass(frozen=True)
class BoardDetection:
    """Detected marker quadrilaterals keyed by marker ID."""

    corners_by_id: Dict[int, np.ndarray]
    charuco_by_id: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationMetrics:
    """Rectified vertical-disparity statistics for one stereo frame."""

    common_marker_count: int
    matched_point_count: int
    mean: float
    median: float
    p95: float
    maximum: float
    status: str


def parse_args(argv=None):
    """Parse camera, calibration, validation, and capture settings."""
    parser = argparse.ArgumentParser(
        description="实时查看双目相机去畸变或极线校正后的左右图像。"
    )
    parser.add_argument("--board-preset", choices=PRESETS, default="existing-20mm")
    parser.add_argument("--params", default=None, help="双目标定 NPZ，默认自动选择最新文件")
    parser.add_argument(
        "--search-root", default=".", help="自动查找双目标定参数时使用的根目录"
    )
    parser.add_argument(
        "--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="USB 相机索引"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="双目总宽度，默认取标定宽度的两倍"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="双目高度，默认取标定图像高度"
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="请求帧率")
    parser.add_argument(
        "--fourcc", default=DEFAULT_FOURCC, help="四字符视频格式，默认 MJPG"
    )
    parser.add_argument(
        "--mode",
        choices=("undistort", "rectify"),
        default=DEFAULT_MODE,
        help="初始显示模式",
    )
    parser.add_argument(
        "--guide-spacing",
        type=int,
        default=DEFAULT_GUIDE_SPACING,
        help="水平参考线间距，单位为原始图像像素",
    )
    parser.add_argument("--no-guides", action="store_true", help="启动时隐藏水平参考线")
    parser.add_argument(
        "--no-validation", action="store_true", help="启动时关闭标定板识别和外参验证"
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=DEFAULT_MIN_COMMON_MARKERS,
        help="给出验证结论所需的最少同名可见 marker 数量",
    )
    parser.add_argument("--swap", action="store_true", help="交换输入帧中的左右顺序")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, help="按 s 保存校正图像的目录"
    )
    return parser.parse_args(argv)


def find_latest_stereo_params(search_root: Path = Path(".")) -> Path:
    """Find the newest timestamped stereo calibration NPZ."""
    pattern = "stereo_calib_*/results/stereo_params_*.npz"
    candidates = sorted(Path(search_root).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"未找到双目标定参数，搜索模式: {pattern}")
    return candidates[-1]


def _read_matrix(data, name, expected_shape=None):
    """Read one finite matrix from an NPZ archive."""
    if name not in data:
        raise ValueError(f"双目标定参数缺少 {name}")
    matrix = np.asarray(data[name], dtype=np.float64)
    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValueError(f"{name} 必须为 {expected_shape}，实际为 {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 包含非有限数值")
    return matrix


def _read_camera_model(data) -> str:
    """Return a canonical camera model, defaulting legacy files to pinhole."""
    if "model" not in data:
        return "pinhole"
    raw_model = np.asarray(data["model"])
    if raw_model.size != 1:
        raise ValueError(f"model 必须为单个字符串，实际形状为 {raw_model.shape}")
    value = raw_model.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    model = str(value).strip().lower()
    if model == "fisheye":
        return "fisheye"
    if model in ("pinhole", "rational", "brown", "brown5"):
        return "pinhole"
    raise ValueError(f"不支持的相机模型: {model}")


def load_stereo_parameters(path: Path) -> StereoParameters:
    """Load and validate matrices needed for both viewer modes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"双目标定参数不存在: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"查看器当前仅支持 NPZ 参数: {path}")

    with np.load(str(path), allow_pickle=False) as data:
        model = _read_camera_model(data)
        if "image_size" not in data:
            raise ValueError("双目标定参数缺少 image_size")
        raw_size = np.asarray(data["image_size"]).reshape(-1)
        if raw_size.size != 2:
            raise ValueError(f"image_size 必须包含宽和高，实际形状为 {raw_size.shape}")
        image_size = (int(raw_size[0]), int(raw_size[1]))
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size 必须为正数，实际为 {image_size}")

        matrices = {
            name: _read_matrix(data, name, shape)
            for name, shape in REQUIRED_PARAMETER_SHAPES.items()
        }
        D1 = _read_matrix(data, "D1")
        D2 = _read_matrix(data, "D2")
        if model == "fisheye":
            if D1.size != 4 or D2.size != 4:
                raise ValueError("fisheye 模型要求 D1 和 D2 各包含 4 个畸变系数")
            D1 = D1.reshape(4, 1)
            D2 = D2.reshape(4, 1)
        else:
            if D1.size < 4 or D2.size < 4:
                raise ValueError("pinhole 模型要求 D1 和 D2 至少包含 4 个畸变系数")
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)

    return StereoParameters(
        image_size=image_size,
        D1=D1,
        D2=D2,
        model=model,
        **matrices,
    )


def _build_view_maps(
    camera_matrix,
    distortion,
    rectification,
    projection,
    image_size,
    model,
):
    """Create remap arrays with the matching OpenCV camera model."""
    if model == "fisheye":
        return cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            rectification,
            projection,
            image_size,
            cv2.CV_32FC1,
        )
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        rectification,
        projection,
        image_size,
        cv2.CV_32FC1,
    )


def build_remap_sets(parameters: StereoParameters) -> Dict[str, RemapPair]:
    """Precompute pure-undistortion and stereo-rectification maps."""
    identity = np.eye(3, dtype=np.float64)
    undistort_left = _build_view_maps(
        parameters.K1,
        parameters.D1,
        identity,
        parameters.K1,
        parameters.image_size,
        parameters.model,
    )
    undistort_right = _build_view_maps(
        parameters.K2,
        parameters.D2,
        identity,
        parameters.K2,
        parameters.image_size,
        parameters.model,
    )
    rectify_left = _build_view_maps(
        parameters.K1,
        parameters.D1,
        parameters.R1,
        parameters.P1,
        parameters.image_size,
        parameters.model,
    )
    rectify_right = _build_view_maps(
        parameters.K2,
        parameters.D2,
        parameters.R2,
        parameters.P2,
        parameters.image_size,
        parameters.model,
    )
    return {
        "undistort": RemapPair(
            undistort_left[0],
            undistort_left[1],
            undistort_right[0],
            undistort_right[1],
        ),
        "rectify": RemapPair(
            rectify_left[0],
            rectify_left[1],
            rectify_right[0],
            rectify_right[1],
        ),
    }


def split_stereo_frame(frame: np.ndarray, image_size: Tuple[int, int]):
    """Split an exact side-by-side stereo frame into calibrated views."""
    if frame is None or frame.ndim not in (2, 3):
        raise ValueError("输入帧为空或维度无效")
    view_width, view_height = image_size
    expected_shape = (view_height, view_width * 2)
    actual_shape = frame.shape[:2]
    if actual_shape != expected_shape:
        raise ValueError(
            f"双目帧尺寸必须为 {expected_shape[1]}x{expected_shape[0]}，"
            f"实际为 {actual_shape[1]}x{actual_shape[0]}"
        )
    return frame[:, :view_width], frame[:, view_width:]


def remap_stereo_views(left, right, map_sets, mode):
    """Apply the selected cached correction maps to both camera views."""
    if mode not in map_sets:
        raise ValueError(f"未知校正模式: {mode}")
    maps = map_sets[mode]
    left_corrected = cv2.remap(
        left,
        maps.left_x,
        maps.left_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_corrected = cv2.remap(
        right,
        maps.right_x,
        maps.right_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_corrected, right_corrected


def detect_marker_corners(gray, detector, clahe, clahe_strong, charuco_detector=None) -> BoardDetection:
    """Detect board markers in one unscaled grayscale camera view."""
    if gray is None or gray.ndim != 2:
        raise ValueError("标定板识别要求非空灰度图像")

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        for enhancer in (clahe, clahe_strong):
            corners, ids, _ = detector.detectMarkers(enhancer.apply(gray))
            if ids is not None and len(ids) > 0:
                break
    if ids is None or len(ids) == 0:
        return BoardDetection({})

    charuco = {}
    if charuco_detector is not None:
        board, native_detector = charuco_detector
        detection, *_ = detect_charuco_robust(gray, board, native_detector, min_corners=4, scales=(1.,))
        if detection is not None:
            charuco = dict(zip(detection.ids.tolist(), detection.corners.reshape(-1, 2)))
    return BoardDetection(
        {
            int(marker_id): np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            for marker_id, marker_corners in zip(ids.reshape(-1), corners)
        }, charuco
    )


def _transform_points(
    points,
    camera_matrix,
    distortion,
    rectification,
    projection,
    model,
):
    """Transform raw points with the matching OpenCV camera model."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if model == "fisheye":
        return cv2.fisheye.undistortPoints(
            points,
            camera_matrix,
            distortion,
            R=rectification,
            P=projection,
        ).reshape(-1, 2)
    return cv2.undistortPoints(
        points,
        camera_matrix,
        distortion,
        R=rectification,
        P=projection,
    ).reshape(-1, 2)


def transform_board_detection(
    detection: BoardDetection,
    parameters: StereoParameters,
    side: str,
    mode: str,
) -> BoardDetection:
    """Map detected marker quadrilaterals into the active corrected view."""
    if side not in ("left", "right"):
        raise ValueError(f"未知相机目别: {side}")
    if mode not in ("undistort", "rectify"):
        raise ValueError(f"未知校正模式: {mode}")

    if side == "left":
        camera_matrix, distortion = parameters.K1, parameters.D1
        rectification, projection = parameters.R1, parameters.P1
    else:
        camera_matrix, distortion = parameters.K2, parameters.D2
        rectification, projection = parameters.R2, parameters.P2
    if mode == "undistort":
        rectification = np.eye(3, dtype=np.float64)
        projection = camera_matrix

    return BoardDetection(
        {
            marker_id: _transform_points(
                corners,
                camera_matrix,
                distortion,
                rectification,
                projection,
                parameters.model,
            )
            for marker_id, corners in detection.corners_by_id.items()
        }
    )


def classify_validation_status(p95, matched_point_count, min_markers):
    """Classify rectified vertical error using fixed pixel thresholds."""
    if matched_point_count < min_markers or not np.isfinite(p95):
        return "INSUFFICIENT"
    if p95 < PASS_P95_PX:
        return "PASS"
    if p95 <= WARNING_P95_PX:
        return "WARNING"
    return "FAIL"


def calculate_validation_metrics(
    left_detection: BoardDetection,
    right_detection: BoardDetection,
    parameters: StereoParameters,
    min_markers: int = DEFAULT_MIN_COMMON_MARKERS,
) -> ValidationMetrics:
    """Measure vertical disparity of corresponding corners without viewport cropping."""
    common_ids = sorted(
        set(left_detection.corners_by_id) & set(right_detection.corners_by_id)
    )
    if not common_ids:
        nan = float("nan")
        return ValidationMetrics(0, 0, nan, nan, nan, nan, "INSUFFICIENT")

    # Use true corner correspondences. The mean of distorted vertices is not
    # the projection of the physical marker centre.
    charuco_ids = sorted(set(left_detection.charuco_by_id) & set(right_detection.charuco_by_id))
    if charuco_ids:
        left_centers = np.asarray([left_detection.charuco_by_id[i] for i in charuco_ids])
        right_centers = np.asarray([right_detection.charuco_by_id[i] for i in charuco_ids])
    else:
        left_centers = np.concatenate([left_detection.corners_by_id[i] for i in common_ids])
        right_centers = np.concatenate([right_detection.corners_by_id[i] for i in common_ids])
    left_rectified = _transform_points(
        left_centers,
        parameters.K1,
        parameters.D1,
        parameters.R1,
        parameters.P1,
        parameters.model,
    )
    right_rectified = _transform_points(
        right_centers,
        parameters.K2,
        parameters.D2,
        parameters.R2,
        parameters.P2,
        parameters.model,
    )

    width, height = parameters.image_size
    finite = np.all(np.isfinite(left_rectified), axis=1) & np.all(
        np.isfinite(right_rectified), axis=1
    )
    if not np.all(finite):
        nan = float("nan")
        return ValidationMetrics(len(common_ids), 0, nan, nan, nan, nan, "INSUFFICIENT")
    errors = np.abs(left_rectified[:, 1] - right_rectified[:, 1])
    matched_count = int(errors.size)
    if matched_count == 0:
        nan = float("nan")
        return ValidationMetrics(
            len(common_ids), 0, nan, nan, nan, nan, "INSUFFICIENT"
        )

    mean = float(np.mean(errors))
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    status = classify_validation_status(p95, len(common_ids), min_markers)
    return ValidationMetrics(
        len(common_ids), matched_count, mean, median, p95, maximum, status
    )


def draw_board_detection(image, detection: BoardDetection, common_ids):
    """Draw corrected marker outlines and IDs in-place."""
    common_ids = set(common_ids)
    for marker_id, corners in detection.corners_by_id.items():
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            continue
        color = (0, 220, 0) if marker_id in common_ids else (0, 165, 255)
        polygon = np.rint(np.clip(corners, -100000, 100000)).astype(np.int32)
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True, color, 2)
        center = np.rint(corners.mean(axis=0)).astype(int)
        if 0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]:
            cv2.circle(image, tuple(center), 3, color, -1)
            cv2.putText(
                image,
                str(marker_id),
                (int(center[0]) + 5, int(center[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def annotate_board_detections(
    left,
    right,
    left_detection,
    right_detection,
    parameters,
    mode,
):
    """Return corrected-view copies annotated with detected board markers."""
    common_ids = set(left_detection.corners_by_id) & set(
        right_detection.corners_by_id
    )
    corrected_left_detection = transform_board_detection(
        left_detection, parameters, "left", mode
    )
    corrected_right_detection = transform_board_detection(
        right_detection, parameters, "right", mode
    )
    left_annotated = draw_board_detection(
        left.copy(), corrected_left_detection, common_ids
    )
    right_annotated = draw_board_detection(
        right.copy(), corrected_right_detection, common_ids
    )
    return left_annotated, right_annotated


def _format_metric(value):
    """Format a pixel metric or a missing-value placeholder."""
    return f"{value:.2f}" if np.isfinite(value) else "--"


def draw_validation_overlay(preview, metrics: ValidationMetrics, mode: str):
    """Draw a compact board-recognition and epipolar-error summary in-place."""
    status_colors = {
        "PASS": (0, 220, 0),
        "WARNING": (0, 210, 255),
        "FAIL": (0, 0, 255),
        "INSUFFICIENT": (190, 190, 190),
    }
    panel_width = min(preview.shape[1], 650)
    panel_height = min(preview.shape[0], 66)
    cv2.rectangle(preview, (0, 0), (panel_width - 1, panel_height - 1), (0, 0, 0), -1)
    status_color = status_colors[metrics.status]
    first_line = (
        f"VIEW {mode.upper()} | BOARD {metrics.common_marker_count} common / "
        f"{metrics.matched_point_count} usable | {metrics.status}"
    )
    second_line = (
        f"RECTIFIED |dy|: median {_format_metric(metrics.median)} px | "
        f"P95 {_format_metric(metrics.p95)} px"
    )
    cv2.putText(
        preview,
        first_line,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        second_line,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def compose_preview(left, right, guides=True, guide_spacing=40):
    """Combine full-resolution corrected views and add optional guides."""
    if guide_spacing <= 0:
        raise ValueError("参考线间距必须大于 0")
    if left.shape != right.shape:
        raise ValueError(f"校正后左右图像尺寸不一致: {left.shape} / {right.shape}")
    preview = np.hstack((left, right))
    if guides:
        preview = preview.copy()
        color = 255 if preview.ndim == 2 else (0, 255, 255)
        for y in range(guide_spacing, preview.shape[0], guide_spacing):
            cv2.line(preview, (0, y), (preview.shape[1] - 1, y), color, 1)
    return preview


def save_processed_pair(output_dir: Path, image: np.ndarray, mode: str) -> Path:
    """Save one full-resolution corrected stereo pair without annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{mode}_{timestamp}.jpg"
    if output_path.exists():
        raise FileExistsError(f"目标文件已存在，拒绝覆盖: {output_path}")
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"无法写入校正图像: {output_path}")
    return output_path


def validate_args(args):
    """Reject invalid camera and validation settings before device access."""
    if args.camera < 0:
        raise ValueError("--camera 不能为负数")
    if args.width is not None and args.width <= 0:
        raise ValueError("--width 必须大于 0")
    if args.height is not None and args.height <= 0:
        raise ValueError("--height 必须大于 0")
    if args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc 必须恰好包含 4 个字符")
    if args.guide_spacing <= 0:
        raise ValueError("--guide-spacing 必须大于 0")
    if args.min_markers <= 0:
        raise ValueError("--min-markers 必须大于 0")


def open_camera(args, image_size):
    """Open and validate the V4L2 stereo camera stream."""
    expected_width = image_size[0] * 2
    expected_height = image_size[1]
    requested_width = args.width if args.width is not None else expected_width
    requested_height = args.height if args.height is not None else expected_height
    if (requested_width, requested_height) != (expected_width, expected_height):
        raise ValueError(
            f"标定参数要求双目输入 {expected_width}x{expected_height}，"
            f"请求为 {requested_width}x{requested_height}"
        )

    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    if not camera.isOpened():
        camera.release()
        raise OSError(f"无法打开 USB 相机 {args.camera}")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (expected_width, expected_height):
        camera.release()
        raise RuntimeError(
            f"相机未接受 {expected_width}x{expected_height}，"
            f"实际为 {actual_width}x{actual_height}"
        )
    return camera


def _camera_format(camera):
    """Return a concise description of the negotiated camera stream."""
    fourcc_value = int(camera.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    return f"{fourcc} {width}x{height} @ {fps:g} fps"


def run_viewer(args):
    """Run the live correction and display loop."""
    validate_args(args)
    params_path = (
        Path(args.params)
        if args.params
        else find_latest_stereo_params(Path(args.search_root))
    )
    parameters = load_stereo_parameters(params_path)
    map_sets = build_remap_sets(parameters)
    detector, clahe, clahe_strong = setup_detector()
    board = PRESETS[args.board_preset].create()
    charuco_detector = (board, setup_charuco_detector(board))
    print(f"双目标定参数: {params_path}")
    print(f"相机模型: {parameters.model}")
    camera = open_camera(args, parameters.image_size)
    mode = args.mode
    guides = not args.no_guides
    validation_enabled = not args.no_validation
    output_dir = Path(args.output)

    print(f"相机格式: {_camera_format(camera)}")
    print(f"当前模式: {mode}")
    print(f"标定板验证: {'开启' if validation_enabled else '关闭'}")
    print("按 m 切换模式，v 切换标定板验证，g 切换参考线，s 保存，Esc 或 q 退出。")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("相机读取失败")
            left, right = split_stereo_frame(frame, parameters.image_size)
            if args.swap:
                left, right = right, left
            left_corrected, right_corrected = remap_stereo_views(
                left, right, map_sets, mode
            )
            if validation_enabled:
                left_detection = detect_marker_corners(
                    cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                    detector,
                    clahe,
                    clahe_strong,
                    charuco_detector,
                )
                right_detection = detect_marker_corners(
                    cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                    detector,
                    clahe,
                    clahe_strong,
                    charuco_detector,
                )
                metrics = calculate_validation_metrics(
                    left_detection,
                    right_detection,
                    parameters,
                    args.min_markers,
                )
                display_left, display_right = annotate_board_detections(
                    left_corrected,
                    right_corrected,
                    left_detection,
                    right_detection,
                    parameters,
                    mode,
                )
                preview = compose_preview(
                    display_left,
                    display_right,
                    guides,
                    args.guide_spacing,
                )
                draw_validation_overlay(preview, metrics, mode)
            else:
                preview = compose_preview(
                    left_corrected,
                    right_corrected,
                    guides,
                    args.guide_spacing,
                )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mode = "undistort" if mode == "rectify" else "rectify"
                print(f"当前模式: {mode}")
            elif key == ord("g"):
                guides = not guides
                print(f"水平参考线: {'显示' if guides else '隐藏'}")
            elif key == ord("v"):
                validation_enabled = not validation_enabled
                print(f"标定板验证: {'开启' if validation_enabled else '关闭'}")
            elif key == ord("s"):
                corrected_pair = np.hstack((left_corrected, right_corrected))
                saved_path = save_processed_pair(output_dir, corrected_pair, mode)
                print(f"已保存: {saved_path}")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\n收到中断，结束查看。")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None):
    """Run the command-line entry point."""
    try:
        return run_viewer(parse_args(argv))
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        print(f"错误: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
