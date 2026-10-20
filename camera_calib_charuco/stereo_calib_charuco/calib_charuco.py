"""Calibrate one camera from ChArUco images or interactively captured frames."""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from cv2 import aruco

# Permit both direct legacy script execution and installed-package use.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_calibration.config import BoardConfig
from stereo_calibration.detection import (
    CharucoDetection, safe_corner_subpix, setup_charuco_detector, detect_charuco_robust,
)
from stereo_calibration.intrinsics import (
    CalibrationObservation, RemovedCorner, calibrate_fisheye, fisheye_residuals,
    filter_corner_outliers,
)


def create_charuco_board():
    dictionary = next(name for name in dir(aruco)
                      if name.startswith("DICT_") and getattr(aruco, name) == ARUCO_DICT)
    return BoardConfig(SQUARES_X, SQUARES_Y, SQUARE_SIZE, MARKER_SIZE, dictionary, True).geometry()



# ====================== Configuration defaults ======================
MODE = "image"  # image / usb / ip
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_SIZE = 20.0
MARKER_SIZE = 15.0
ARUCO_DICT = cv2.aruco.DICT_5X5_100

IMAGE_PATH = "charuco_left"
USB_CAM_INDEX = 0
IP_CAM_ADDRESS = "http://admin:admin@10.252.128.185:8081"
CAM_WIDTH = 1280
CAM_HEIGHT = 720

CAPTURE_COUNT = 50
SAVE_RAW_IMAGES = True
SAVE_CORNER_IMAGES = True
SAVE_PARAMS = True

DEFAULT_MIN_CHARUCO_CORNERS = 12
DEFAULT_POINT_OUTLIER_PX = 1.0
DEFAULT_MAX_OUTLIER_ITERATIONS = 5
DEFAULT_PREVIEW_COUNT = 4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    500,
    1e-11,
)






@dataclass
class RejectedImage:
    """An input image excluded from calibration and its reason."""

    name: str
    reason: str
    corner_count: int = 0
    marker_count: int = 0
    scale: float = 1.0
    variant: str = "raw"




@dataclass
class CalibrationBundle:
    """Recommended and compatibility calibration results."""

    fisheye_rms: float
    fisheye_matrix: np.ndarray
    fisheye_distortion: np.ndarray
    fisheye_rvecs: Sequence[np.ndarray]
    fisheye_tvecs: Sequence[np.ndarray]
    fisheye_view_errors: np.ndarray
    rational_result: tuple
    standard_result: tuple
    removed_corners: List[RemovedCorner]


def parse_args(argv=None):
    """Parse source, filtering, capture, and output settings."""
    parser = argparse.ArgumentParser(
        description="使用 ChArUco 图像标定单目内参，并输出 fisheye 与 rational 模型。"
    )
    parser.add_argument("--mode", choices=("image", "usb", "ip"), default=MODE)
    parser.add_argument("--images", default=IMAGE_PATH, help="image 模式的输入目录")
    parser.add_argument("--camera-index", type=int, default=USB_CAM_INDEX)
    parser.add_argument("--ip-address", default=IP_CAM_ADDRESS)
    parser.add_argument("--width", type=int, default=CAM_WIDTH)
    parser.add_argument("--height", type=int, default=CAM_HEIGHT)
    parser.add_argument("--capture-count", type=int, default=CAPTURE_COUNT)
    parser.add_argument("--output", default=None, help="显式输出目录；默认使用时间戳")
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=DEFAULT_MIN_CHARUCO_CORNERS,
        help="每张图至少保留的 ChArUco 角点数",
    )
    parser.add_argument(
        "--point-outlier-px",
        type=float,
        default=DEFAULT_POINT_OUTLIER_PX,
        help="删除孤立角点时使用的重投影残差阈值",
    )
    parser.add_argument(
        "--max-outlier-iterations",
        type=int,
        default=DEFAULT_MAX_OUTLIER_ITERATIONS,
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=DEFAULT_PREVIEW_COUNT,
        help="保存的 fisheye 去畸变预览数量，0 表示不保存",
    )
    parser.set_defaults(raw_copies=SAVE_RAW_IMAGES, corner_images=SAVE_CORNER_IMAGES)
    parser.add_argument("--raw-copies", dest="raw_copies", action="store_true")
    parser.add_argument("--no-raw-copies", dest="raw_copies", action="store_false")
    parser.add_argument("--corner-images", dest="corner_images", action="store_true")
    parser.add_argument(
        "--no-corner-images", dest="corner_images", action="store_false"
    )
    return parser.parse_args(argv)


def validate_args(args):
    """Reject numeric settings that cannot produce a valid calibration."""
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width 和 height 必须大于 0")
    if args.capture_count <= 0:
        raise ValueError("capture-count 必须大于 0")
    if args.min_charuco_corners < 4:
        raise ValueError("min-charuco-corners 必须至少为 4")
    if args.point_outlier_px <= 0:
        raise ValueError("point-outlier-px 必须大于 0")
    if args.max_outlier_iterations < 0:
        raise ValueError("max-outlier-iterations 不能小于 0")
    if args.preview_count < 0:
        raise ValueError("preview-count 不能小于 0")


def create_output_folders(image_path, output=None):
    """Create a new output tree and refuse an existing explicit directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output is None:
        input_name = os.path.basename(os.path.normpath(str(image_path)))
        root = Path(f"{input_name}_calib_{timestamp}")
    else:
        root = Path(output)
    if root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {root}")

    subfolders = {
        "raw": root / "raw_images",
        "corners": root / "corner_images",
        "results": root / "results",
        "logs": root / "logs",
        "previews": root / "previews",
    }
    for folder in subfolders.values():
        folder.mkdir(parents=True, exist_ok=False)
    return str(root), {key: str(value) for key, value in subfolders.items()}, timestamp


def write_log(log_file, content):
    """Append a timestamped log line and print the same message."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
    with open(log_file, "a", encoding="utf-8") as stream:
        stream.write(line)
    print(content)


def cv_imread(file_path):
    """Read an image through bytes so non-ASCII paths work on all platforms."""
    try:
        with open(file_path, "rb") as stream:
            data = np.frombuffer(stream.read(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)




def setup_detector():
    """Configure the marker detector used by capture and stereo modules."""
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 9
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.001
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    return detector, clahe, clahe_strong




def detect_markers_robust(
    gray, detector, clahe, clahe_strong, marker_3d, board
):
    """Detect markers while preserving the legacy helper return contract."""
    candidates = []
    for variant, image in (
        ("raw", gray),
        ("clahe", clahe.apply(gray)),
        ("clahe_strong", clahe_strong.apply(gray)),
    ):
        corners, ids, rejected = detector.detectMarkers(image)
        count = 0 if ids is None else len(ids)
        candidates.append((count, variant, corners, ids, rejected))

    count, _, corners, ids, rejected = max(candidates, key=lambda item: item[0])
    if count == 0 or ids is None:
        return None, None, None, None

    refined = detector.refineDetectedMarkers(gray, board, corners, ids, rejected)
    corners_refined, ids_refined = refined[0], refined[1]
    if ids_refined is None:
        return None, None, None, None

    object_points = []
    image_points = []
    kept_corners = []
    kept_ids = []
    height, width = gray.shape[:2]
    margin = 6.0
    # OpenCV bindings may return IDs as either (N,) or (N, 1).
    flat_ids = np.asarray(ids_refined, dtype=np.int32).reshape(-1)
    for marker_corners, marker_id_value in zip(corners_refined, flat_ids):
        marker_id = int(marker_id_value)
        points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        inside = (
            np.all(np.isfinite(points))
            and np.all(points[:, 0] >= margin)
            and np.all(points[:, 0] < width - margin)
            and np.all(points[:, 1] >= margin)
            and np.all(points[:, 1] < height - margin)
        )
        if marker_id not in marker_3d or not inside:
            continue
        refined_points, indices = safe_corner_subpix(
            gray, points, window=(5, 5), max_shift_px=5.0
        )
        if len(indices) != 4:
            continue
        object_points.append(marker_3d[marker_id])
        image_points.append(refined_points.reshape(4, 2))
        kept_corners.append(refined_points.reshape(1, 4, 2))
        kept_ids.append([marker_id])

    if len(object_points) < 6:
        return None, None, None, None

    return (
        np.asarray(object_points, dtype=np.float32).reshape(-1, 1, 3),
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
        kept_corners,
        np.asarray(kept_ids, dtype=np.int32),
    )






def refine_chessboard_corners(
    gray, marker_obj, marker_img, mtx, dist, chessboard_3d
):
    """Project and safely refine only chessboard corners inside the image."""
    ok, rvec, tvec = cv2.solvePnP(
        marker_obj.reshape(-1, 3), marker_img.reshape(-1, 2), mtx, dist
    )
    if not ok:
        return None, None
    projected, _ = cv2.projectPoints(chessboard_3d, rvec, tvec, mtx, dist)
    refined, indices = safe_corner_subpix(
        gray, projected, window=(3, 3), max_shift_px=10.0
    )
    if len(indices) == 0:
        return None, None
    objects = np.asarray(chessboard_3d, dtype=np.float32).reshape(-1, 1, 3)
    return objects[indices], refined


def compute_per_view_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Calculate root-mean-square reprojection error for each image."""
    errors = []
    for objects, images, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objects, rvec, tvec, mtx, dist)
        differences = images.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(differences**2, axis=1)))))
    return errors


def refine_calibration(objpoints, imgpoints, img_size, log_file=None):
    """Retain the legacy pinhole helper with named high-error view filtering."""
    result = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, criteria=CALIBRATION_CRITERIA
    )
    rms, matrix, distortion, rvecs, tvecs = result
    for iteration in range(2):
        errors = compute_per_view_errors(
            objpoints, imgpoints, rvecs, tvecs, matrix, distortion
        )
        if len(errors) <= 5:
            break
        threshold = float(np.mean(errors)) * 1.5
        keep = [index for index, error in enumerate(errors) if error <= threshold]
        if len(keep) == len(errors) or len(keep) < 5:
            break
        if log_file:
            write_log(
                log_file,
                f"迭代 {iteration + 1}: 剔除 {len(errors) - len(keep)} 张高误差图片",
            )
        objpoints = [objpoints[index] for index in keep]
        imgpoints = [imgpoints[index] for index in keep]
        rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            img_size,
            matrix,
            distortion,
            criteria=CALIBRATION_CRITERIA,
        )
    return rms, matrix, distortion, rvecs, tvecs, objpoints, imgpoints


def list_images(directory):
    """List supported input images in deterministic filename order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_observations(
    image_paths,
    board,
    min_corners,
    corner_directory=None,
    log_file=None,
):
    """Detect reliable observations and record every rejected image."""
    detector = setup_charuco_detector(board)
    chessboard_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    observations = []
    rejections = []
    image_size = None

    for path in image_paths:
        color = cv_imread(str(path))
        if color is None:
            rejections.append(RejectedImage(path.name, "无法读取图像"))
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejections.append(
                RejectedImage(
                    path.name,
                    f"图像尺寸不一致: {current_size}, 期望 {image_size}",
                )
            )
            continue

        detection, count, marker_count, scale, variant = detect_charuco_robust(
            gray, board, detector=detector, min_corners=min_corners
        )
        if detection is None:
            rejections.append(
                RejectedImage(
                    path.name,
                    f"可靠 ChArUco 角点少于 {min_corners}",
                    count,
                    marker_count,
                    scale,
                    variant,
                )
            )
            if log_file:
                write_log(
                    log_file,
                    f"拒绝 {path.name}: corners={count}, markers={marker_count}",
                )
            continue

        object_points = chessboard_points[detection.ids].reshape(-1, 1, 3)
        observation = CalibrationObservation(
            name=path.name,
            path=Path(path),
            object_points=object_points.astype(np.float64),
            image_points=detection.corners.astype(np.float64),
            corner_ids=detection.ids.copy(),
            initial_corner_count=len(detection.ids),
            marker_count=detection.marker_count,
            scale=detection.scale,
            variant=detection.variant,
            blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        )
        observations.append(observation)

        if corner_directory:
            annotated = color.copy()
            aruco.drawDetectedCornersCharuco(
                annotated,
                detection.corners.astype(np.float32),
                detection.ids.reshape(-1, 1),
            )
            cv2.putText(
                annotated,
                f"corners={len(detection.ids)} scale={detection.scale:g} {detection.variant}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(str(Path(corner_directory) / path.name), annotated)

        if log_file:
            write_log(
                log_file,
                f"接受 {path.name}: corners={len(detection.ids)}, "
                f"markers={detection.marker_count}, scale={detection.scale:g}, "
                f"variant={detection.variant}",
            )

    if image_size is None:
        raise ValueError("没有可读取的输入图像")
    return observations, rejections, image_size








def calibrate_pinhole(observations, image_size, flags):
    """Fit an OpenCV pinhole model and return extended diagnostics."""
    return cv2.calibrateCameraExtended(
        [item.object_points.astype(np.float32) for item in observations],
        [item.image_points.astype(np.float32) for item in observations],
        image_size,
        None,
        None,
        flags=flags,
        criteria=CALIBRATION_CRITERIA,
    )


def calibrate_models(
    observations,
    image_size,
    point_outlier_px,
    min_corners,
    max_outlier_iterations,
    log_file=None,
):
    """Fit fisheye, rational, and diagnostic Brown models."""
    fisheye_result, removed = filter_corner_outliers(
        observations,
        image_size,
        point_outlier_px,
        min_corners,
        max_outlier_iterations,
        log_file,
    )
    fisheye_rms, fisheye_matrix, fisheye_distortion, rvecs, tvecs = (
        fisheye_result
    )
    fisheye_view_errors, _ = fisheye_residuals(
        observations, fisheye_matrix, fisheye_distortion, rvecs, tvecs
    )
    rational_result = calibrate_pinhole(
        observations, image_size, cv2.CALIB_RATIONAL_MODEL
    )
    standard_result = calibrate_pinhole(observations, image_size, 0)
    return CalibrationBundle(
        fisheye_rms,
        fisheye_matrix,
        fisheye_distortion,
        rvecs,
        tvecs,
        fisheye_view_errors,
        rational_result,
        standard_result,
        removed,
    )


def _stack_vectors(vectors):
    """Convert OpenCV's vector list into a stable numeric array."""
    return np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])


def _write_yaml(path, camera_matrix, distortion, rms, model, image_size):
    """Write camera parameters with OpenCV FileStorage."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"无法创建 YAML: {path}")
    try:
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", distortion)
        storage.write("reprojection_error", float(rms))
        storage.write("distortion_model", model)
        storage.write("image_width", int(image_size[0]))
        storage.write("image_height", int(image_size[1]))
    finally:
        storage.release()


def save_previews(
    observations,
    camera_matrix,
    distortion,
    image_size,
    preview_directory,
    preview_count,
):
    """Save cropped rectilinear previews suitable for a near-180-degree lens."""
    if preview_count <= 0 or not observations:
        return None
    width, height = image_size
    focal = width * 0.3125
    output_matrix = np.asarray(
        [
            [focal, 0.0, (width - 1) / 2],
            [0.0, focal, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        np.eye(3),
        output_matrix,
        image_size,
        cv2.CV_16SC2,
    )
    indices = np.linspace(
        0, len(observations) - 1, min(preview_count, len(observations)), dtype=int
    )
    for index in np.unique(indices):
        item = observations[int(index)]
        image = cv_imread(str(item.path))
        if image is None:
            continue
        corrected = cv2.remap(
            image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imwrite(
            str(Path(preview_directory) / f"fisheye_undistorted_{item.name}"),
            corrected,
        )
    return output_matrix


def save_calibration_results(
    root_folder,
    subfolders,
    timestamp,
    observations,
    rejections,
    image_size,
    bundle,
    preview_count,
):
    """Persist both models, metrics, rejections, report, and previews."""
    results = Path(subfolders["results"])
    rational = bundle.rational_result
    standard = bundle.standard_result
    rat_rms, rat_matrix, rat_dist, rat_rvecs, rat_tvecs = rational[:5]
    rat_view_errors = np.asarray(rational[7]).reshape(-1)
    std_rms, std_matrix, std_dist = standard[:3]
    std_view_errors = np.asarray(standard[7]).reshape(-1)

    if SAVE_PARAMS:
        np.savez(
            results / f"params_{timestamp}.npz",
            mtx=rat_matrix,
            dist=rat_dist,
            rvecs=_stack_vectors(rat_rvecs),
            tvecs=_stack_vectors(rat_tvecs),
            error=float(rat_rms),
            image_size=np.asarray(image_size),
            model=np.asarray("rational"),
        )
        np.savez(
            results / f"fisheye_params_{timestamp}.npz",
            mtx=bundle.fisheye_matrix,
            dist=bundle.fisheye_distortion,
            rvecs=_stack_vectors(bundle.fisheye_rvecs),
            tvecs=_stack_vectors(bundle.fisheye_tvecs),
            error=float(bundle.fisheye_rms),
            image_size=np.asarray(image_size),
            model=np.asarray("fisheye"),
        )
        _write_yaml(
            results / f"params_{timestamp}.yaml",
            rat_matrix,
            rat_dist,
            rat_rms,
            "rational",
            image_size,
        )
        _write_yaml(
            results / f"fisheye_params_{timestamp}.yaml",
            bundle.fisheye_matrix,
            bundle.fisheye_distortion,
            bundle.fisheye_rms,
            "fisheye",
            image_size,
        )

    with open(results / "per_view_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "image",
                "corners_initial",
                "corners_used",
                "markers",
                "scale",
                "variant",
                "blur_laplacian",
                "fisheye_rms_px",
                "rational_rms_px",
            ]
        )
        for item, fish_error, rat_error in zip(
            observations, bundle.fisheye_view_errors, rat_view_errors
        ):
            writer.writerow(
                [
                    item.name,
                    item.initial_corner_count,
                    len(item.corner_ids),
                    item.marker_count,
                    item.scale,
                    item.variant,
                    f"{item.blur_score:.6f}",
                    f"{fish_error:.6f}",
                    f"{rat_error:.6f}",
                ]
            )

    with open(results / "rejected_images.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["image", "reason", "corners", "markers", "scale", "variant"]
        )
        for rejection in rejections:
            writer.writerow(
                [
                    rejection.name,
                    rejection.reason,
                    rejection.corner_count,
                    rejection.marker_count,
                    rejection.scale,
                    rejection.variant,
                ]
            )

    with open(
        results / "removed_corner_outliers.csv", "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "corner_id", "residual_px", "x", "y", "iteration"])
        for removed in bundle.removed_corners:
            writer.writerow(
                [
                    removed.image,
                    removed.corner_id,
                    f"{removed.residual_px:.6f}",
                    f"{removed.x:.3f}",
                    f"{removed.y:.3f}",
                    removed.iteration,
                ]
            )

    preview_matrix = save_previews(
        observations,
        bundle.fisheye_matrix,
        bundle.fisheye_distortion,
        image_size,
        subfolders["previews"],
        preview_count,
    )
    if preview_matrix is not None:
        storage = cv2.FileStorage(
            str(results / "preview_projection.yaml"), cv2.FILE_STORAGE_WRITE
        )
        storage.write("new_camera_matrix", preview_matrix)
        storage.write("projection", "rectilinear")
        storage.release()

    total_points = sum(len(item.corner_ids) for item in observations)
    report = f"""Charuco 单目内参标定报告
{'=' * 72}
输入图像: {len(observations) + len(rejections)}
成功使用: {len(observations)}
拒绝图像: {len(rejections)}
最终角点: {total_points}
删除孤立角点: {len(bundle.removed_corners)}
图像尺寸: {image_size[0]}x{image_size[1]}
标定板: {SQUARES_X}x{SQUARES_Y}, square={SQUARE_SIZE:.3f} mm, marker={MARKER_SIZE:.3f} mm
板型: legacy pattern=True

推荐模型: OpenCV fisheye
-----------------------
RMS: {bundle.fisheye_rms:.9f} px
逐帧 RMS median/P95/max: {np.median(bundle.fisheye_view_errors):.6f} / {np.percentile(bundle.fisheye_view_errors, 95):.6f} / {np.max(bundle.fisheye_view_errors):.6f} px
K:
{np.array2string(bundle.fisheye_matrix, precision=9)}
D:
{np.array2string(bundle.fisheye_distortion.ravel(), precision=12)}

兼容模型: OpenCV rational
-------------------------
RMS: {rat_rms:.9f} px
逐帧 RMS median/P95/max: {np.median(rat_view_errors):.6f} / {np.percentile(rat_view_errors, 95):.6f} / {np.max(rat_view_errors):.6f} px
K:
{np.array2string(rat_matrix, precision=9)}
D:
{np.array2string(rat_dist.ravel(), precision=12)}
说明: params_{timestamp}.npz/yaml 保存 rational 参数，供现有 pinhole 双目标定代码自动加载。

诊断模型: 普通 Brown 5 参数
---------------------------
RMS: {std_rms:.9f} px
逐帧 RMS median/P95/max: {np.median(std_view_errors):.6f} / {np.percentile(std_view_errors, 95):.6f} / {np.max(std_view_errors):.6f} px
K:
{np.array2string(std_matrix, precision=9)}
D:
{np.array2string(std_dist.ravel(), precision=12)}
"""
    (results / "calibration_report.txt").write_text(report, encoding="utf-8")
    return Path(root_folder)


def classify_intrinsic_accuracy(rms):
    """Classify subpixel intrinsic accuracy for the console summary."""
    if rms < 0.3:
        return "优秀"
    if rms < 0.5:
        return "良好"
    if rms < 1.0:
        return "可用"
    return "较差"


def format_calibration_summary(
    output,
    timestamp,
    observations,
    rejections,
    bundle,
):
    """Format complete matrices, distortion, and accuracy for the terminal."""
    rational = bundle.rational_result
    standard = bundle.standard_result
    rat_rms, rat_matrix, rat_dist = rational[:3]
    rat_view_errors = np.asarray(rational[7]).reshape(-1)
    std_rms = float(standard[0])
    total_points = sum(len(item.corner_ids) for item in observations)
    results = Path(output) / "results"

    return f"""
{'=' * 72}
ChArUco 单目内参标定结果
{'=' * 72}
有效图像: {len(observations)}/{len(observations) + len(rejections)}
拒绝图像: {len(rejections)}
最终角点: {total_points}
删除孤立角点: {len(bundle.removed_corners)}

[推荐] OpenCV fisheye 4 参数
精度等级: {classify_intrinsic_accuracy(bundle.fisheye_rms)}
全局 RMS: {bundle.fisheye_rms:.6f} px
逐帧 RMS median/P95/max: {np.median(bundle.fisheye_view_errors):.6f} / {np.percentile(bundle.fisheye_view_errors, 95):.6f} / {np.max(bundle.fisheye_view_errors):.6f} px
相机内参矩阵 K:
{np.array2string(bundle.fisheye_matrix, precision=9, suppress_small=False)}
畸变系数 D (k1, k2, k3, k4):
{np.array2string(bundle.fisheye_distortion.ravel(), precision=12, suppress_small=False)}
参数文件: {results / f'fisheye_params_{timestamp}.npz'}

[兼容现有双目代码] OpenCV rational
精度等级: {classify_intrinsic_accuracy(float(rat_rms))}
全局 RMS: {float(rat_rms):.6f} px
逐帧 RMS median/P95/max: {np.median(rat_view_errors):.6f} / {np.percentile(rat_view_errors, 95):.6f} / {np.max(rat_view_errors):.6f} px
相机内参矩阵 K:
{np.array2string(rat_matrix, precision=9, suppress_small=False)}
畸变系数 D (k1, k2, p1, p2, k3, k4, k5, k6 ...):
{np.array2string(rat_dist.ravel(), precision=12, suppress_small=False)}
参数文件: {results / f'params_{timestamp}.npz'}

[诊断] 普通 Brown 5 参数
全局 RMS: {std_rms:.6f} px
结论: {"该模型明显欠拟合，不建议使用。" if std_rms > float(rat_rms) * 2 else "该模型与扩展模型误差接近。"}

完整报告: {results / 'calibration_report.txt'}
逐帧指标: {results / 'per_view_metrics.csv'}
输出目录: {output}
{'=' * 72}"""


def analyze_and_save_results(result_file, error, mtx, dist):
    """Retain the original compact quality-report helper."""
    if error < 0.5:
        level, suggestion = "优秀", "精度极高"
    elif error < 1.0:
        level, suggestion = "良好", "精度合格"
    elif error < 3.0:
        level = "一般"
        suggestion = "增加更多角度图片、改善光照并确保标定板清晰"
    else:
        level = "较差"
        suggestion = "检查板参数、图像清晰度、反光和有效角点数量"
    with open(result_file, "w", encoding="utf-8") as stream:
        stream.write(f"重投影误差 (RMS): {error:.6f} 像素\n")
        stream.write(f"精度等级: {level}\n")
        stream.write(f"建议: {suggestion}\n\n")
        stream.write("相机内参矩阵:\n")
        stream.write(np.array2string(mtx, precision=4) + "\n\n")
        stream.write("畸变系数:\n")
        stream.write(np.array2string(dist.ravel(), precision=6) + "\n")
    return level, suggestion


def copy_raw_images(image_paths, raw_directory):
    """Copy decoded source images without changing their filenames."""
    for path in image_paths:
        image = cv_imread(str(path))
        if image is None:
            continue
        cv2.imwrite(str(Path(raw_directory) / path.name), image)


def capture_camera_images(args, subfolders, log_file):
    """Interactively capture valid marker frames for later ChArUco calibration."""
    source = args.camera_index if args.mode == "usb" else args.ip_address
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"无法打开相机源: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    board, _, marker_3d = create_charuco_board()
    detector, clahe, clahe_strong = setup_detector()
    paths = []
    try:
        while len(paths) < args.capture_count:
            ok, frame = capture.read()
            if not ok:
                raise OSError("相机读取失败")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = detect_markers_robust(
                gray, detector, clahe, clahe_strong, marker_3d, board
            )
            valid = detected[0] is not None
            display = frame.copy()
            if valid:
                aruco.drawDetectedMarkers(display, detected[2], detected[3])
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.putText(
                display,
                f"DETECTION: {'OK' if valid else 'FAILED'} | SAVED: {len(paths)}/{args.capture_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.putText(
                display,
                "[s] Save | [ESC/q] Finish",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Charuco Calibration Capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and valid:
                path = Path(subfolders["raw"]) / f"raw_{len(paths):03d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"无法保存采集图像: {path}")
                paths.append(path)
                write_log(log_file, f"保存采集图像 {path.name}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return paths


def run_calibration(args):
    """Run collection, robust observation detection, calibration, and saving."""
    validate_args(args)
    source_name = args.images if args.mode == "image" else f"charuco_{args.mode}"
    root_folder, subfolders, timestamp = create_output_folders(
        source_name, args.output
    )
    log_file = str(Path(subfolders["logs"]) / f"calib_charuco_{timestamp}.log")
    write_log(log_file, "===== ChArUco 内参标定开始 =====")
    write_log(
        log_file,
        f"模式={args.mode}, board={SQUARES_X}x{SQUARES_Y}, "
        f"square={SQUARE_SIZE} mm, marker={MARKER_SIZE} mm",
    )

    if args.mode == "image":
        image_paths = list_images(args.images)
        if not image_paths:
            raise ValueError(f"图片目录中没有支持的图像: {args.images}")
        write_log(log_file, f"找到 {len(image_paths)} 张图片")
        if args.raw_copies:
            copy_raw_images(image_paths, subfolders["raw"])
    else:
        image_paths = capture_camera_images(args, subfolders, log_file)
        if len(image_paths) < 5:
            raise ValueError(f"有效采集图像不足: {len(image_paths)}/5")

    board, _, _ = create_charuco_board()
    observations, rejections, image_size = collect_observations(
        image_paths,
        board,
        args.min_charuco_corners,
        subfolders["corners"] if args.corner_images else None,
        log_file,
    )
    if len(observations) < 5:
        raise ValueError(f"有效 ChArUco 图像不足: {len(observations)}/5")

    write_log(
        log_file,
        f"开始标定: accepted={len(observations)}, rejected={len(rejections)}",
    )
    bundle = calibrate_models(
        observations,
        image_size,
        args.point_outlier_px,
        args.min_charuco_corners,
        args.max_outlier_iterations,
        log_file,
    )
    output = save_calibration_results(
        root_folder,
        subfolders,
        timestamp,
        observations,
        rejections,
        image_size,
        bundle,
        args.preview_count,
    )

    rational_rms = float(bundle.rational_result[0])
    standard_rms = float(bundle.standard_result[0])
    write_log(log_file, f"fisheye RMS={bundle.fisheye_rms:.6f} px")
    write_log(log_file, f"rational RMS={rational_rms:.6f} px")
    write_log(log_file, f"Brown 5 参数 RMS={standard_rms:.6f} px")
    write_log(log_file, f"结果目录: {output}")
    write_log(log_file, "===== ChArUco 内参标定完成 =====")

    print(
        format_calibration_summary(
            output,
            timestamp,
            observations,
            rejections,
            bundle,
        )
    )
    return output


def main(argv=None):
    """Command-line entry point."""
    try:
        run_calibration(parse_args(argv))
    except (OSError, ValueError, FileNotFoundError, FileExistsError, cv2.error) as error:
        print(f"错误: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
