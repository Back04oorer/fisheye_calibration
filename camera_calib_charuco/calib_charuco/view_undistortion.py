#!/usr/bin/env python3
"""实时并排显示相机原图和去畸变图。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAPTURE_FOLDER = Path(__file__).resolve().parent / "charuco_calib_20260728_220516"
DEFAULT_CALIBRATION_SIZE = (640, 480)
WINDOW_NAME = "Camera Undistortion Comparison"


@dataclass(frozen=True)
class CameraParameters:
    matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple
    reprojection_error: float
    source: Path


def resolve_parameter_file(path):
    """解析参数文件；目录输入会选择 results 中最新的参数文件。"""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise ValueError(f"参数路径不存在: {source}")

    results_folder = source / "results" if (source / "results").is_dir() else source
    candidates = list(results_folder.glob("params_*.yaml"))
    candidates.extend(results_folder.glob("params_*.yml"))
    candidates.extend(results_folder.glob("params_*.npz"))
    if not candidates:
        raise ValueError(f"目录中没有 params_*.yaml、params_*.yml 或 params_*.npz: {results_folder}")

    return max(
        candidates,
        key=lambda item: (item.stat().st_mtime, item.suffix.lower() in (".yaml", ".yml")),
    )


def load_camera_parameters(path, calibration_size=None):
    """从 OpenCV YAML 或 NumPy NPZ 文件读取相机参数。"""
    parameter_file = resolve_parameter_file(path)
    suffix = parameter_file.suffix.lower()
    stored_size = None
    error = float("nan")

    if suffix == ".npz":
        with np.load(str(parameter_file)) as data:
            if "mtx" not in data or "dist" not in data:
                raise ValueError(f"NPZ 缺少 mtx 或 dist: {parameter_file}")
            matrix = np.asarray(data["mtx"], dtype=np.float64)
            distortion = np.asarray(data["dist"], dtype=np.float64)
            if "image_size" in data:
                size_values = np.asarray(data["image_size"]).reshape(-1)
                if size_values.size == 2:
                    stored_size = tuple(int(value) for value in size_values)
            if "error" in data:
                error = float(np.asarray(data["error"]).reshape(-1)[0])
    elif suffix in (".yaml", ".yml"):
        storage = cv2.FileStorage(str(parameter_file), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"无法打开 YAML 参数文件: {parameter_file}")
        try:
            matrix = storage.getNode("camera_matrix").mat()
            distortion = storage.getNode("distortion_coefficients").mat()
            error_node = storage.getNode("reprojection_error")
            if not error_node.empty():
                error = float(error_node.real())
            width_node = storage.getNode("image_width")
            height_node = storage.getNode("image_height")
            if not width_node.empty() and not height_node.empty():
                stored_size = (int(width_node.real()), int(height_node.real()))
        finally:
            storage.release()
    else:
        raise ValueError(f"不支持的参数格式: {parameter_file.suffix}")

    if matrix is None or distortion is None:
        raise ValueError(f"参数文件缺少相机矩阵或畸变系数: {parameter_file}")

    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(1, -1)
    if matrix.shape != (3, 3):
        raise ValueError(f"相机矩阵必须为 3x3，当前为 {matrix.shape}")
    if distortion.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"畸变系数数量无效: {distortion.size}")
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("相机参数包含 NaN 或 Inf")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("相机焦距必须为正数")

    selected_size = tuple(calibration_size) if calibration_size else stored_size
    if selected_size is None:
        selected_size = DEFAULT_CALIBRATION_SIZE
    if len(selected_size) != 2 or min(selected_size) <= 0:
        raise ValueError(f"标定分辨率无效: {selected_size}")

    return CameraParameters(
        matrix=matrix,
        distortion=distortion,
        image_size=tuple(int(value) for value in selected_size),
        reprojection_error=error,
        source=parameter_file,
    )


def scale_camera_matrix(matrix, source_size, target_size):
    """按图像尺寸缩放焦距和主点。"""
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled = matrix.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def create_undistort_maps(parameters, frame_size):
    """为给定视频尺寸生成裁掉黑边的去畸变映射。"""
    scaled_matrix = scale_camera_matrix(
        parameters.matrix, parameters.image_size, frame_size
    )
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        scaled_matrix,
        parameters.distortion,
        frame_size,
        0.0,
        frame_size,
    )
    return cv2.initUndistortRectifyMap(
        scaled_matrix,
        parameters.distortion,
        None,
        new_matrix,
        frame_size,
        cv2.CV_32FC1,
    )


def add_label(image, text):
    """在画面左上角添加清晰标签。"""
    labeled = image.copy()
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    )
    background_width = min(labeled.shape[1], text_width + 24)
    cv2.rectangle(labeled, (0, 0), (background_width, 42), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def build_comparison(frame, map_x, map_y):
    """生成固定高度的左右对比画面。"""
    undistorted = cv2.remap(
        frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    original_view = add_label(frame, "Original")
    corrected_view = add_label(undistorted, "Undistorted")
    separator = np.full((frame.shape[0], 2, 3), 255, dtype=np.uint8)
    return np.hstack((original_view, separator, corrected_view))


def run_viewer(parameters, camera_index, requested_size):
    """打开相机并运行实时对比窗口。"""
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_size[1])
    if not capture.isOpened():
        capture.release()
        print(f"错误：无法打开相机索引 {camera_index}")
        return False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        map_x = None
        map_y = None
        mapped_size = None
        failed_reads = 0
        warned_aspect_ratio = False

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    print("错误：连续 30 帧读取失败，停止显示")
                    return False
                continue
            failed_reads = 0

            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size != mapped_size:
                source_ratio = parameters.image_size[0] / parameters.image_size[1]
                frame_ratio = frame_size[0] / frame_size[1]
                if not warned_aspect_ratio and abs(source_ratio - frame_ratio) > 0.01:
                    print(
                        "警告：视频与标定图像的宽高比不同，去畸变结果可能不准确: "
                        f"{frame_size} vs {parameters.image_size}"
                    )
                    warned_aspect_ratio = True
                map_x, map_y = create_undistort_maps(parameters, frame_size)
                mapped_size = frame_size

            comparison = build_comparison(frame, map_x, map_y)
            cv2.imshow(WINDOW_NAME, comparison)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return True
    except cv2.error as error:
        print(f"错误：无法创建或更新显示窗口: {error}")
        return False
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="实时比较相机原图和去畸变图")
    parser.add_argument(
        "--params",
        default=str(DEFAULT_CAPTURE_FOLDER),
        help="参数文件或包含 results 的采集目录",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="相机索引，默认 0")
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_CALIBRATION_SIZE,
        help="请求的视频分辨率，默认 640 480",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="覆盖参数文件的标定分辨率",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.frame_size) <= 0:
        print(f"错误：视频分辨率无效: {tuple(args.frame_size)}")
        return 1

    try:
        parameters = load_camera_parameters(args.params, args.calibration_size)
    except (OSError, ValueError, cv2.error) as error:
        print(f"错误：{error}")
        return 1

    print(f"参数文件: {parameters.source}")
    print(f"标定分辨率: {parameters.image_size[0]}x{parameters.image_size[1]}")
    if np.isfinite(parameters.reprojection_error):
        print(f"重投影误差: {parameters.reprojection_error:.6f} px")
    print("按 Q 或 ESC 退出")

    success = run_viewer(
        parameters,
        args.camera_index,
        tuple(args.frame_size),
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
