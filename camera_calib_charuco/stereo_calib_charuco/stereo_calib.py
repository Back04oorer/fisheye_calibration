"""Calibrate fisheye stereo extrinsics from paired ChArUco images."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_calib_charuco.calib_charuco import (
    create_charuco_board,
    cv_imread,
    detect_charuco_robust,
    setup_charuco_detector,
)


from stereo_calibration.opencv import fisheye_flag


DEFAULT_LEFT_DIR = "charuco_left"
DEFAULT_RIGHT_DIR = "charuco_right"
DEFAULT_BASELINE_MM = 60.0
DEFAULT_MIN_CORNERS = 12
DEFAULT_POINTS_PER_PAIR = 12
DEFAULT_MIN_PAIRS = 12
DEFAULT_MAX_OUTLIER_ITERATIONS = 2
DEFAULT_OUTLIER_FLOOR_PX = 1.0
DEFAULT_PREVIEW_COUNT = 6
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STATUS_RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}


@dataclass
class StereoObservation:
    """Matched ChArUco points from one stereo pair."""

    name: str
    left_path: Path
    right_path: Path
    object_points: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    common_corner_count: int
    left_marker_count: int
    right_marker_count: int


@dataclass
class Rejection:
    """An image pair excluded from calibration and its reason."""

    name: str
    reason: str


@dataclass
class PairMetrics:
    """Rectified vertical disparity statistics for one stereo pair."""

    name: str
    point_count: int
    mean: float
    median: float
    p95: float
    maximum: float


@dataclass
class CalibrationResult:
    """Optimized fisheye stereo geometry and rectification quality."""

    rms: float
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: Tuple[int, int, int, int]
    roi2: Tuple[int, int, int, int]
    pair_metrics: List[PairMetrics]
    vertical_mean: float
    vertical_median: float
    vertical_p95: float
    vertical_maximum: float


def parse_args(argv=None):
    """Parse stereo input, filtering, and output settings."""
    parser = argparse.ArgumentParser(
        description="使用左右目 fisheye 内参计算双目外参和校正参数。"
    )
    parser.add_argument("--left-images", default=DEFAULT_LEFT_DIR, help="左目图像目录")
    parser.add_argument("--right-images", default=DEFAULT_RIGHT_DIR, help="右目图像目录")
    parser.add_argument(
        "--left-params", default=None, help="左目 fisheye 内参 NPZ 或 YAML"
    )
    parser.add_argument(
        "--right-params", default=None, help="右目 fisheye 内参 NPZ 或 YAML"
    )
    parser.add_argument(
        "--search-root", default=".", help="自动查找最新内参时使用的根目录"
    )
    parser.add_argument(
        "--baseline-mm",
        type=float,
        default=DEFAULT_BASELINE_MM,
        help="产品标称基线，仅用于结果校验",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=DEFAULT_MIN_CORNERS,
        help="每对图像至少需要的共同 ChArUco 角点数量",
    )
    parser.add_argument(
        "--points-per-pair",
        type=int,
        default=DEFAULT_POINTS_PER_PAIR,
        help="每对送入 OpenCV 鱼眼双目标定的固定角点数量",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=DEFAULT_MIN_PAIRS,
        help="离群筛选后必须保留的最少图像对数量",
    )
    parser.add_argument(
        "--max-outlier-iterations",
        type=int,
        default=DEFAULT_MAX_OUTLIER_ITERATIONS,
        help="按校正后垂直误差 P95 剔除离群图像对的最大迭代次数",
    )
    parser.add_argument(
        "--outlier-floor-px",
        type=float,
        default=DEFAULT_OUTLIER_FLOOR_PX,
        help="离群图像对垂直误差 P95 的最低阈值",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=DEFAULT_PREVIEW_COUNT,
        help="保存的代表性鱼眼校正预览图数量，0 表示不保存",
    )
    parser.add_argument("--output", default=None, help="输出目录，默认使用时间戳")
    return parser.parse_args(argv)


def list_images(directory: Path) -> Dict[str, Path]:
    """Return supported images keyed by exact filename."""
    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {directory}")
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def pair_image_paths(left_dir: Path, right_dir: Path):
    """Pair images by exact filename and return unpaired names."""
    left_images = list_images(Path(left_dir))
    right_images = list_images(Path(right_dir))
    paired_names = sorted(set(left_images) & set(right_images))
    pairs = [(left_images[name], right_images[name]) for name in paired_names]
    left_only = sorted(set(left_images) - set(right_images))
    right_only = sorted(set(right_images) - set(left_images))
    return pairs, left_only, right_only


def find_latest_params(side: str, search_root: Path = Path(".")) -> Path:
    """Find the latest timestamped fisheye intrinsic NPZ for one side."""
    if side not in ("left", "right"):
        raise ValueError(f"未知相机目别: {side}")
    pattern = f"charuco_{side}_calib_*/results/fisheye_params_*.npz"
    candidates = sorted(Path(search_root).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"未找到{side}目 fisheye 内参，搜索模式: {pattern}")
    return candidates[-1]


def _read_yaml_string(storage, key: str) -> str:
    """Read an optional string from an OpenCV YAML file."""
    node = storage.getNode(key)
    return "" if node.empty() else node.string()


def load_intrinsics(path: Path, expected_size=None):
    """Load and validate one four-coefficient OpenCV fisheye model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"内参文件不存在: {path}")

    model = ""
    image_size = None
    if path.suffix.lower() == ".npz":
        with np.load(str(path), allow_pickle=False) as data:
            if "mtx" not in data or "dist" not in data:
                raise ValueError(f"NPZ 缺少 mtx 或 dist: {path}")
            camera_matrix = np.asarray(data["mtx"], dtype=np.float64)
            distortion = np.asarray(data["dist"], dtype=np.float64)
            if "image_size" in data:
                image_size = tuple(int(x) for x in data["image_size"].reshape(-1))
            if "model" in data:
                model = str(np.asarray(data["model"]).item())
    elif path.suffix.lower() in (".yaml", ".yml"):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise OSError(f"无法打开内参 YAML: {path}")
        try:
            if not storage.getNode("image_width").empty():
                image_size = (int(storage.getNode("image_width").real()), int(storage.getNode("image_height").real()))
            camera_matrix = storage.getNode("camera_matrix").mat()
            distortion = storage.getNode("distortion_coefficients").mat()
            model = (_read_yaml_string(storage, "distortion_model") or _read_yaml_string(storage, "model"))
        finally:
            storage.release()
        if camera_matrix is None or distortion is None:
            raise ValueError(f"YAML 缺少相机矩阵或畸变系数: {path}")
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64)
    else:
        raise ValueError(f"不支持的内参文件类型: {path.suffix}")

    if expected_size is not None and image_size != tuple(expected_size):
        raise ValueError(f"Intrinsic image size {image_size} does not match images {expected_size}: {path}")
    if model and model.lower() != "fisheye":
        raise ValueError(f"内参模型必须为 fisheye，实际为 {model}: {path}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"相机矩阵必须为 3x3，实际为 {camera_matrix.shape}: {path}")
    if distortion.size != 4:
        raise ValueError(
            f"fisheye 畸变系数必须恰好为 4 个，实际为 {distortion.size}: {path}"
        )
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(distortion)):
        raise ValueError(f"内参包含非有限数值: {path}")
    return camera_matrix, distortion.reshape(4, 1)


def select_spatially_distributed_indices(
    left_points: np.ndarray,
    right_points: np.ndarray,
    count: int,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Select a deterministic farthest-point subset across both images."""
    left = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    if len(left) != len(right):
        raise ValueError("左右共同角点数量不一致")
    if count < 1 or count > len(left):
        raise ValueError(f"选择数量必须位于 1..{len(left)}，实际为 {count}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须大于 0")

    scale = np.asarray([width, height], dtype=np.float64)
    features = np.hstack((left / scale, right / scale))
    centroid = np.mean(features, axis=0)
    first = int(np.argmax(np.sum((features - centroid) ** 2, axis=1)))
    selected = [first]
    min_squared_distance = np.sum((features - features[first]) ** 2, axis=1)

    while len(selected) < count:
        min_squared_distance[np.asarray(selected, dtype=int)] = -1.0
        next_index = int(np.argmax(min_squared_distance))
        selected.append(next_index)
        candidate_distance = np.sum(
            (features - features[next_index]) ** 2, axis=1
        )
        min_squared_distance = np.minimum(
            min_squared_distance, candidate_distance
        )
    return np.asarray(selected, dtype=np.int32)


def match_common_charuco_points(
    left_ids: np.ndarray,
    left_points: np.ndarray,
    right_ids: np.ndarray,
    right_points: np.ndarray,
    chessboard_points: np.ndarray,
    points_per_pair: int,
    image_size: Tuple[int, int],
):
    """Intersect ChArUco IDs and return one fixed-size distributed point set."""
    left_map = {
        int(corner_id): point
        for corner_id, point in zip(
            np.asarray(left_ids).reshape(-1),
            np.asarray(left_points).reshape(-1, 2),
        )
    }
    right_map = {
        int(corner_id): point
        for corner_id, point in zip(
            np.asarray(right_ids).reshape(-1),
            np.asarray(right_points).reshape(-1, 2),
        )
    }
    common_ids = np.asarray(sorted(set(left_map) & set(right_map)), dtype=np.int32)
    if len(common_ids) < points_per_pair:
        raise ValueError(
            f"共同 ChArUco 角点仅 {len(common_ids)} 个，固定选点要求 {points_per_pair}"
        )

    common_left = np.asarray([left_map[int(i)] for i in common_ids], dtype=np.float64)
    common_right = np.asarray(
        [right_map[int(i)] for i in common_ids], dtype=np.float64
    )
    selected_indices = select_spatially_distributed_indices(
        common_left, common_right, points_per_pair, image_size
    )
    selected_ids = common_ids[selected_indices]
    objects = np.asarray(chessboard_points, dtype=np.float64)[selected_ids]
    return (
        len(common_ids),
        selected_ids,
        objects.reshape(-1, 1, 3),
        common_left[selected_indices].reshape(-1, 1, 2),
        common_right[selected_indices].reshape(-1, 1, 2),
    )


def collect_observations(
    pairs,
    min_corners: int,
    points_per_pair: int,
):
    """Detect and match direct ChArUco observations in all paired images."""
    board, chessboard_points, _ = create_charuco_board()
    detector = setup_charuco_detector(board)
    observations = []
    rejections = []
    image_size = None

    for left_path, right_path in pairs:
        name = left_path.name
        left_color = cv_imread(str(left_path))
        right_color = cv_imread(str(right_path))
        if left_color is None or right_color is None:
            rejections.append(Rejection(name, "无法读取左目或右目图像"))
            continue
        if left_color.shape[:2] != right_color.shape[:2]:
            rejections.append(
                Rejection(
                    name,
                    f"左右尺寸不一致: {left_color.shape[:2]} / {right_color.shape[:2]}",
                )
            )
            continue
        current_size = (left_color.shape[1], left_color.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejections.append(
                Rejection(name, f"图像尺寸 {current_size} 与首对 {image_size} 不一致")
            )
            continue

        left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
        left_detection = detect_charuco_robust(
            left_gray, board, detector=detector, min_corners=min_corners
        )
        right_detection = detect_charuco_robust(
            right_gray, board, detector=detector, min_corners=min_corners
        )
        left, left_count, left_markers, _, _ = left_detection
        right, right_count, right_markers, _, _ = right_detection
        if left is None or right is None:
            rejections.append(
                Rejection(
                    name,
                    "至少一侧可靠 ChArUco 检测不足: "
                    f"left={left_count}, right={right_count}, 要求 {min_corners}",
                )
            )
            continue

        common_ids = set(int(value) for value in left.ids.reshape(-1)) & set(
            int(value) for value in right.ids.reshape(-1)
        )
        if len(common_ids) < min_corners or len(common_ids) < points_per_pair:
            rejections.append(
                Rejection(
                    name,
                    f"共同 ChArUco 角点仅 {len(common_ids)} 个，要求至少 "
                    f"{max(min_corners, points_per_pair)}",
                )
            )
            continue

        common_count, _, objects, left_points, right_points = (
            match_common_charuco_points(
                left.ids,
                left.corners,
                right.ids,
                right.corners,
                chessboard_points,
                points_per_pair,
                image_size,
            )
        )
        observations.append(
            StereoObservation(
                name=name,
                left_path=left_path,
                right_path=right_path,
                object_points=objects,
                left_points=left_points,
                right_points=right_points,
                common_corner_count=common_count,
                left_marker_count=left_markers,
                right_marker_count=right_markers,
            )
        )

    return observations, rejections, image_size


def prepare_fisheye_inputs(observations):
    """Validate OpenCV's fixed-size Python binding requirement."""
    if not observations:
        raise ValueError("没有可用于鱼眼双目标定的观测")
    expected_count = len(observations[0].object_points)
    if expected_count < 4:
        raise ValueError("每对图像至少需要 4 个标定点")

    object_points = []
    left_points = []
    right_points = []
    for item in observations:
        counts = (
            len(item.object_points),
            len(item.left_points),
            len(item.right_points),
        )
        if counts != (expected_count, expected_count, expected_count):
            raise ValueError(
                f"{item.name} 点数 {counts} 与固定数量 {expected_count} 不一致"
            )
        object_points.append(
            np.asarray(item.object_points, dtype=np.float64).reshape(-1, 1, 3)
        )
        left_points.append(
            np.asarray(item.left_points, dtype=np.float64).reshape(-1, 1, 2)
        )
        right_points.append(
            np.asarray(item.right_points, dtype=np.float64).reshape(-1, 1, 2)
        )
    return object_points, left_points, right_points


def calculate_pair_metrics(
    observations,
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    R1,
    R2,
    P1,
    P2,
):
    """Calculate fisheye-rectified vertical disparities."""
    metrics = []
    all_errors = []
    for observation in observations:
        left_rectified = cv2.fisheye.undistortPoints(
            observation.left_points,
            left_camera_matrix,
            left_distortion,
            R=R1,
            P=P1,
        ).reshape(-1, 2)
        right_rectified = cv2.fisheye.undistortPoints(
            observation.right_points,
            right_camera_matrix,
            right_distortion,
            R=R2,
            P=P2,
        ).reshape(-1, 2)
        errors = np.abs(left_rectified[:, 1] - right_rectified[:, 1])
        all_errors.extend(errors.tolist())
        metrics.append(
            PairMetrics(
                name=observation.name,
                point_count=len(errors),
                mean=float(np.mean(errors)),
                median=float(np.median(errors)),
                p95=float(np.percentile(errors, 95)),
                maximum=float(np.max(errors)),
            )
        )

    combined = np.asarray(all_errors, dtype=np.float64)
    return (
        metrics,
        float(np.mean(combined)),
        float(np.median(combined)),
        float(np.percentile(combined, 95)),
        float(np.max(combined)),
    )


def _essential_and_approximate_fundamental(
    rotation: np.ndarray,
    translation: np.ndarray,
    left_camera_matrix: np.ndarray,
    right_camera_matrix: np.ndarray,
):
    """Build E and a compatibility-only pinhole-plane F approximation."""
    tx, ty, tz = np.asarray(translation, dtype=np.float64).reshape(3)
    cross = np.asarray(
        [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64
    )
    essential = cross @ rotation
    fundamental = (
        np.linalg.inv(right_camera_matrix).T
        @ essential
        @ np.linalg.inv(left_camera_matrix)
    )
    return essential, fundamental


def calibrate_once(
    observations,
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    image_size,
):
    """Run fixed-intrinsic fisheye stereo calibration and rectification once."""
    object_points, left_points, right_points = prepare_fisheye_inputs(observations)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-7,
    )
    calibration = cv2.fisheye.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_camera_matrix.copy(),
        left_distortion.copy(),
        right_camera_matrix.copy(),
        right_distortion.copy(),
        image_size,
        flags=fisheye_flag("CALIB_FIX_INTRINSIC"),
        criteria=criteria,
    )
    rms, _, _, _, _, rotation, translation = calibration[:7]
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        image_size,
        rotation,
        translation,
        flags=fisheye_flag("CALIB_ZERO_DISPARITY"),
        newImageSize=image_size,
        balance=0.0,
        fov_scale=1.0,
    )
    metrics = calculate_pair_metrics(
        observations,
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        R1,
        R2,
        P1,
        P2,
    )
    essential, fundamental = _essential_and_approximate_fundamental(
        rotation, translation, left_camera_matrix, right_camera_matrix
    )
    full_roi = (0, 0, int(image_size[0]), int(image_size[1]))
    return CalibrationResult(
        rms=float(rms),
        R=np.asarray(rotation, dtype=np.float64),
        T=np.asarray(translation, dtype=np.float64).reshape(3, 1),
        E=essential,
        F=fundamental,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=full_roi,
        roi2=full_roi,
        pair_metrics=metrics[0],
        vertical_mean=metrics[1],
        vertical_median=metrics[2],
        vertical_p95=metrics[3],
        vertical_maximum=metrics[4],
    )


def robust_outlier_threshold(
    scores: Sequence[float], floor_px: float = DEFAULT_OUTLIER_FLOOR_PX
) -> float:
    """Return a robust upper threshold for per-pair error scores."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("无法为空误差集合计算离群阈值")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_limit = median + 2.5 * 1.4826 * mad
    return max(float(floor_px), robust_limit)


def calibrate_with_filtering(
    observations,
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    image_size,
    min_pairs,
    max_iterations,
    outlier_floor_px,
):
    """Remove pairs with outlying vertical-error P95 and return the final fit."""
    active = list(observations)
    rejected = []
    history = []

    for iteration in range(max_iterations):
        result = calibrate_once(
            active,
            left_camera_matrix,
            left_distortion,
            right_camera_matrix,
            right_distortion,
            image_size,
        )
        threshold = robust_outlier_threshold(
            [metric.p95 for metric in result.pair_metrics], outlier_floor_px
        )
        candidates = sorted(
            [metric for metric in result.pair_metrics if metric.p95 > threshold],
            key=lambda metric: metric.p95,
            reverse=True,
        )
        removable_count = len(active) - min_pairs
        if not candidates or removable_count <= 0:
            history.append((iteration + 1, threshold, []))
            break
        removed_metrics = candidates[:removable_count]
        removed_names = {metric.name for metric in removed_metrics}
        history.append((iteration + 1, threshold, removed_metrics))
        for metric in removed_metrics:
            rejected.append(
                Rejection(
                    metric.name,
                    f"第 {iteration + 1} 次离群筛选: 垂直误差 P95 "
                    f"{metric.p95:.4f} px > {threshold:.4f} px",
                )
            )
        active = [item for item in active if item.name not in removed_names]

    final_result = calibrate_once(
        active,
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        image_size,
    )
    return active, rejected, history, final_result


def classify_lower_is_better(value, pass_limit, warning_limit):
    """Classify a lower-is-better quality metric."""
    if value < pass_limit:
        return "PASS"
    if value <= warning_limit:
        return "WARNING"
    return "FAIL"


def classify_baseline(estimated_mm: float, nominal_mm: float) -> str:
    """Classify baseline deviation from a positive nominal value."""
    if nominal_mm <= 0:
        raise ValueError("标称基线必须大于 0")
    relative_error = abs(estimated_mm - nominal_mm) / nominal_mm
    if relative_error <= 0.05:
        return "PASS"
    if relative_error <= 0.10:
        return "WARNING"
    return "FAIL"


def overall_status(statuses: Sequence[str]) -> str:
    """Return the most severe quality status."""
    return max(statuses, key=lambda value: STATUS_RANK[value])


def create_output_folders(output: Optional[str]):
    """Create a unique output tree without overwriting an earlier run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output is not None:
        root = Path(output)
        if root.exists():
            raise FileExistsError(f"输出目录已存在，拒绝覆盖: {root}")
    else:
        root = Path(f"stereo_calib_{timestamp}")
        suffix = 1
        while root.exists():
            root = Path(f"stereo_calib_{timestamp}_{suffix:02d}")
            suffix += 1
    folders = {
        "root": root,
        "results": root / "results",
        "logs": root / "logs",
        "previews": root / "rectified_previews",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    return folders, timestamp


def write_log(log_path: Path, message: str):
    """Append a timestamped message to the log and print it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{timestamp}] {message}\n")
    print(message)


def save_parameters(
    results_dir,
    timestamp,
    result,
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    image_size,
    nominal_baseline_mm,
    statuses,
    observations,
    rejections,
):
    """Write fisheye stereo parameters to NPZ and OpenCV YAML files."""
    estimated_baseline_mm = float(np.linalg.norm(result.T))
    used_names = np.asarray([item.name for item in observations], dtype=str)
    rejected_entries = np.asarray(
        [f"{item.name}: {item.reason}" for item in rejections], dtype=str
    )
    metric_names = np.asarray([item.name for item in result.pair_metrics], dtype=str)
    metric_values = np.asarray(
        [
            [item.point_count, item.mean, item.median, item.p95, item.maximum]
            for item in result.pair_metrics
        ],
        dtype=np.float64,
    )
    npz_path = results_dir / f"stereo_params_{timestamp}.npz"
    np.savez(
        str(npz_path),
        model=np.asarray("fisheye"),
        image_size=np.asarray(image_size, dtype=np.int32),
        K1=left_camera_matrix,
        D1=left_distortion,
        K2=right_camera_matrix,
        D2=right_distortion,
        R=result.R,
        T=result.T,
        E=result.E,
        F=result.F,
        fundamental_matrix_is_approximate=np.asarray(True),
        R1=result.R1,
        R2=result.R2,
        P1=result.P1,
        P2=result.P2,
        Q=result.Q,
        roi1=np.asarray(result.roi1, dtype=np.int32),
        roi2=np.asarray(result.roi2, dtype=np.int32),
        stereo_rms=result.rms,
        nominal_baseline_mm=nominal_baseline_mm,
        estimated_baseline_mm=estimated_baseline_mm,
        vertical_error_mean=result.vertical_mean,
        vertical_error_median=result.vertical_median,
        vertical_error_p95=result.vertical_p95,
        vertical_error_max=result.vertical_maximum,
        quality_status=np.asarray(statuses["overall"]),
        used_pairs=used_names,
        rejected_pairs=rejected_entries,
        pair_metric_names=metric_names,
        pair_metrics=metric_values,
    )

    yaml_path = results_dir / f"stereo_params_{timestamp}.yaml"
    storage = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"无法创建 YAML: {yaml_path}")
    try:
        storage.write("model", "fisheye")
        storage.write("image_width", int(image_size[0]))
        storage.write("image_height", int(image_size[1]))
        storage.write("left_camera_matrix", left_camera_matrix)
        storage.write("left_distortion_coefficients", left_distortion)
        storage.write("right_camera_matrix", right_camera_matrix)
        storage.write("right_distortion_coefficients", right_distortion)
        storage.write("R", result.R)
        storage.write("T", result.T)
        storage.write("E", result.E)
        storage.write("F", result.F)
        storage.write("fundamental_matrix_is_approximate", 1)
        storage.write("R1", result.R1)
        storage.write("R2", result.R2)
        storage.write("P1", result.P1)
        storage.write("P2", result.P2)
        storage.write("Q", result.Q)
        storage.write("roi1", np.asarray(result.roi1, dtype=np.int32))
        storage.write("roi2", np.asarray(result.roi2, dtype=np.int32))
        storage.write("stereo_rms", result.rms)
        storage.write("nominal_baseline_mm", float(nominal_baseline_mm))
        storage.write("estimated_baseline_mm", estimated_baseline_mm)
        storage.write("vertical_error_mean", result.vertical_mean)
        storage.write("vertical_error_median", result.vertical_median)
        storage.write("vertical_error_p95", result.vertical_p95)
        storage.write("vertical_error_max", result.vertical_maximum)
        storage.write("quality_status", statuses["overall"])
        storage.write("used_pair_count", len(observations))
        storage.write("rejected_pair_count", len(rejections))
    finally:
        storage.release()
    return npz_path, yaml_path


def format_matrix(matrix):
    """Format a matrix consistently for the text report."""
    return np.array2string(np.asarray(matrix), precision=8, suppress_small=False)


def save_report(
    report_path,
    left_params,
    right_params,
    image_size,
    nominal_baseline_mm,
    result,
    statuses,
    observations,
    rejections,
    history,
):
    """Write a human-readable calibration and quality report."""
    estimated_baseline_mm = float(np.linalg.norm(result.T))
    rotation_vector, _ = cv2.Rodrigues(result.R)
    rotation_degrees = np.degrees(rotation_vector.reshape(-1))
    with Path(report_path).open("w", encoding="utf-8") as stream:
        stream.write("=" * 64 + "\n")
        stream.write("鱼眼双目外参标定报告\n")
        stream.write("=" * 64 + "\n")
        stream.write("模型: OpenCV fisheye（4 参数）\n")
        stream.write(f"左目内参: {left_params}\n")
        stream.write(f"右目内参: {right_params}\n")
        stream.write(f"图像尺寸: {image_size[0]}x{image_size[1]}\n")
        stream.write(f"使用图像对: {len(observations)}\n")
        stream.write(f"剔除图像对: {len(rejections)}\n\n")
        stream.write("坐标约定: X_right = R * X_left + T\n")
        stream.write(f"双目 RMS: {result.rms:.6f} px [{statuses['rms']}]\n")
        stream.write(
            f"标称基线: {nominal_baseline_mm:.6f} mm\n"
            f"估计基线 ||T||: {estimated_baseline_mm:.6f} mm "
            f"[{statuses['baseline']}]\n"
            f"基线差值: {estimated_baseline_mm - nominal_baseline_mm:+.6f} mm\n"
        )
        stream.write(
            f"垂直误差均值: {result.vertical_mean:.6f} px\n"
            f"垂直误差中位数: {result.vertical_median:.6f} px "
            f"[{statuses['vertical_median']}]\n"
            f"垂直误差 P95: {result.vertical_p95:.6f} px "
            f"[{statuses['vertical_p95']}]\n"
            f"垂直误差最大值: {result.vertical_maximum:.6f} px\n"
            f"综合质量: {statuses['overall']}\n\n"
        )
        stream.write(f"旋转向量 (deg): {format_matrix(rotation_degrees)}\n")
        stream.write(f"R:\n{format_matrix(result.R)}\n\n")
        stream.write(f"T (mm):\n{format_matrix(result.T)}\n\n")
        stream.write(f"E:\n{format_matrix(result.E)}\n\n")
        stream.write(
            "F（兼容近似）:\n"
            f"{format_matrix(result.F)}\n"
            "说明: 鱼眼畸变像素不服从单个 3x3 基础矩阵；F 仅对应去畸变后的针孔归一化平面。\n\n"
        )
        stream.write(f"R1:\n{format_matrix(result.R1)}\n\n")
        stream.write(f"R2:\n{format_matrix(result.R2)}\n\n")
        stream.write(f"P1:\n{format_matrix(result.P1)}\n\n")
        stream.write(f"P2:\n{format_matrix(result.P2)}\n\n")
        stream.write(f"Q:\n{format_matrix(result.Q)}\n\n")

        stream.write("离群筛选记录（逐对垂直误差 P95）:\n")
        if not history:
            stream.write("  无筛选迭代\n")
        for iteration, threshold, removed in history:
            names = ", ".join(item.name for item in removed) or "无"
            stream.write(
                f"  迭代 {iteration}: 阈值 {threshold:.6f} px, 剔除 {names}\n"
            )

        stream.write("\n剔除图像对:\n")
        if not rejections:
            stream.write("  无\n")
        for item in rejections:
            stream.write(f"  {item.name}: {item.reason}\n")

        stream.write("\n保留图像对逐对误差:\n")
        metric_by_name = {item.name: item for item in result.pair_metrics}
        for observation in observations:
            metric = metric_by_name[observation.name]
            stream.write(
                f"  {observation.name}: common={observation.common_corner_count}, "
                f"selected={metric.point_count}, markers(L/R)="
                f"{observation.left_marker_count}/{observation.right_marker_count}, "
                f"mean={metric.mean:.6f}, median={metric.median:.6f}, "
                f"p95={metric.p95:.6f}, max={metric.maximum:.6f} px\n"
            )


def save_rectified_previews(
    preview_dir,
    observations,
    preview_count,
    result,
    left_camera_matrix,
    left_distortion,
    right_camera_matrix,
    right_distortion,
    image_size,
):
    """Save representative fisheye-rectified pairs with horizontal guides."""
    if preview_count <= 0 or not observations:
        return []
    count = min(preview_count, len(observations))
    indices = sorted(
        set(int(value) for value in np.linspace(0, len(observations) - 1, count))
    )
    left_map_x, left_map_y = cv2.fisheye.initUndistortRectifyMap(
        left_camera_matrix,
        left_distortion,
        result.R1,
        result.P1,
        image_size,
        cv2.CV_16SC2,
    )
    right_map_x, right_map_y = cv2.fisheye.initUndistortRectifyMap(
        right_camera_matrix,
        right_distortion,
        result.R2,
        result.P2,
        image_size,
        cv2.CV_16SC2,
    )
    saved = []
    for index in indices:
        observation = observations[index]
        left = cv_imread(str(observation.left_path))
        right = cv_imread(str(observation.right_path))
        if left is None or right is None:
            continue
        left_rectified = cv2.remap(left, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, right_map_x, right_map_y, cv2.INTER_LINEAR)
        preview = np.hstack((left_rectified, right_rectified))
        for y in range(40, preview.shape[0], 40):
            cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 255), 1)
        output_path = preview_dir / f"rectified_{Path(observation.name).stem}.jpg"
        if not cv2.imwrite(str(output_path), preview):
            raise OSError(f"无法写入校正预览图: {output_path}")
        saved.append(output_path)
    return saved


def validate_args(args):
    """Reject invalid numeric settings before processing data."""
    if args.baseline_mm <= 0:
        raise ValueError("--baseline-mm 必须大于 0")
    if args.min_corners < 4:
        raise ValueError("--min-corners 必须至少为 4")
    if args.points_per_pair < 4:
        raise ValueError("--points-per-pair 必须至少为 4")
    if args.min_corners < args.points_per_pair:
        raise ValueError("--min-corners 不能小于 --points-per-pair")
    if args.min_pairs < 3:
        raise ValueError("--min-pairs 必须至少为 3")
    if args.max_outlier_iterations < 0:
        raise ValueError("--max-outlier-iterations 不能为负数")
    if args.outlier_floor_px <= 0:
        raise ValueError("--outlier-floor-px 必须大于 0")
    if args.preview_count < 0:
        raise ValueError("--preview-count 不能为负数")


def run_calibration(args):
    """Execute the complete fisheye stereo calibration workflow."""
    validate_args(args)
    left_dir = Path(args.left_images)
    right_dir = Path(args.right_images)
    search_root = Path(args.search_root)
    left_params = (
        Path(args.left_params)
        if args.left_params
        else find_latest_params("left", search_root)
    )
    right_params = (
        Path(args.right_params)
        if args.right_params
        else find_latest_params("right", search_root)
    )
    left_camera_matrix, left_distortion = load_intrinsics(left_params)
    right_camera_matrix, right_distortion = load_intrinsics(right_params)
    pairs, left_only, right_only = pair_image_paths(left_dir, right_dir)
    if not pairs:
        raise ValueError("左右目录中没有同名图像对")

    folders, timestamp = create_output_folders(args.output)
    log_path = folders["logs"] / f"stereo_calib_{timestamp}.log"
    write_log(log_path, "===== 鱼眼双目外参标定启动 =====")
    write_log(log_path, f"左目 fisheye 内参: {left_params}")
    write_log(log_path, f"右目 fisheye 内参: {right_params}")
    write_log(log_path, f"找到同名图像对: {len(pairs)}")
    write_log(log_path, f"每对固定选取角点: {args.points_per_pair}")

    initial_rejections = [
        Rejection(name, "仅存在于左目目录") for name in left_only
    ] + [Rejection(name, "仅存在于右目目录") for name in right_only]
    observations, detection_rejections, image_size = collect_observations(
        pairs,
        args.min_corners,
        args.points_per_pair,
    )
    rejections = initial_rejections + detection_rejections
    write_log(
        log_path,
        f"ChArUco 筛选后保留 {len(observations)} 对，剔除 {len(rejections)} 对",
    )
    for item in rejections:
        write_log(log_path, f"剔除 {item.name}: {item.reason}")
    if image_size is None:
        raise RuntimeError("无法从图像对确定分辨率")
    load_intrinsics(left_params, image_size)
    load_intrinsics(right_params, image_size)
    if len(observations) < args.min_pairs:
        raise RuntimeError(
            f"有效图像对不足: {len(observations)}/{args.min_pairs}，无法标定"
        )

    active, outlier_rejections, history, result = calibrate_with_filtering(
        observations,
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        image_size,
        args.min_pairs,
        args.max_outlier_iterations,
        args.outlier_floor_px,
    )
    rejections.extend(outlier_rejections)
    for iteration, threshold, removed in history:
        write_log(
            log_path,
            f"离群筛选 {iteration}: P95 阈值 {threshold:.6f} px，"
            f"剔除 {len(removed)} 对",
        )

    estimated_baseline_mm = float(np.linalg.norm(result.T))
    statuses = {
        "rms": classify_lower_is_better(result.rms, 1.0, 1.5),
        "vertical_median": classify_lower_is_better(
            result.vertical_median, 0.5, 1.0
        ),
        "vertical_p95": classify_lower_is_better(result.vertical_p95, 2.0, 3.0),
        "baseline": classify_baseline(estimated_baseline_mm, args.baseline_mm),
    }
    statuses["overall"] = overall_status(list(statuses.values()))

    npz_path, yaml_path = save_parameters(
        folders["results"],
        timestamp,
        result,
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        image_size,
        args.baseline_mm,
        statuses,
        active,
        rejections,
    )
    report_path = folders["results"] / "stereo_calibration_report.txt"
    save_report(
        report_path,
        left_params,
        right_params,
        image_size,
        args.baseline_mm,
        result,
        statuses,
        active,
        rejections,
        history,
    )
    previews = save_rectified_previews(
        folders["previews"],
        active,
        args.preview_count,
        result,
        left_camera_matrix,
        left_distortion,
        right_camera_matrix,
        right_distortion,
        image_size,
    )

    write_log(log_path, f"最终使用图像对: {len(active)}")
    write_log(log_path, f"鱼眼双目 RMS: {result.rms:.6f} px [{statuses['rms']}]")
    write_log(log_path, f"T (mm): {result.T.reshape(-1)}")
    write_log(
        log_path,
        f"估计基线: {estimated_baseline_mm:.6f} mm "
        f"(标称 {args.baseline_mm:.6f} mm) [{statuses['baseline']}]",
    )
    write_log(
        log_path,
        f"垂直误差 median/P95: {result.vertical_median:.6f}/"
        f"{result.vertical_p95:.6f} px",
    )
    write_log(log_path, f"综合质量: {statuses['overall']}")
    write_log(log_path, f"NPZ: {npz_path}")
    write_log(log_path, f"YAML: {yaml_path}")
    write_log(log_path, f"报告: {report_path}")
    write_log(log_path, f"鱼眼校正预览图: {len(previews)} 张")
    write_log(log_path, f"所有结果: {folders['root']}")
    write_log(log_path, "===== 鱼眼双目外参标定结束 =====")
    return 0


def main(argv=None):
    """Run the command-line entry point."""
    try:
        return run_calibration(parse_args(argv))
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        print(f"错误: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
