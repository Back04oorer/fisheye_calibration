import argparse
import cv2
import numpy as np
import os
from datetime import datetime
from cv2 import aruco

# ====================== 配置区 ======================
MODE = "usb"                     # image / usb / ip
SQUARES_X = 12                  # 棋盘格横向方格数
SQUARES_Y = 9                   # 棋盘格纵向方格数
SQUARE_SIZE = 30.0              # 方格边长 (mm)
MARKER_SIZE = 24.0              # ArUco 标记边长 (mm)
ARUCO_DICT = cv2.aruco.DICT_5X5_100

# 图片源配置
IMAGE_PATH = "charuco_400_300"    # MODE=image 时有效
USB_CAM_INDEX = 0
IP_CAM_ADDRESS = "http://admin:admin@10.252.128.185:8081"
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 采集配置
CAPTURE_COUNT = 30              # 需要采集的有效图像数量
SAVE_RAW_IMAGES = True          # 保存原始采集图像
SAVE_CORNER_IMAGES = True       # 保存带标记检测可视化的图像
SAVE_PARAMS = True              # 保存标定结果文件

# ====================================================

def create_output_folders(root_folder=None):
    """创建输出目录结构；未指定根目录时使用时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if root_folder is None:
        root_folder = f"charuco_calib_{timestamp}"
    subfolders = {
        "raw": os.path.join(root_folder, "raw_images"),
        "corners": os.path.join(root_folder, "corner_images"),
        "results": os.path.join(root_folder, "results"),
        "logs": os.path.join(root_folder, "logs")
    }
    for folder in subfolders.values():
        os.makedirs(folder, exist_ok=True)
    return root_folder, subfolders, timestamp

def write_log(log_file, content):
    """写入日志并打印到控制台"""
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{time_str}] {content}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(content)

def analyze_and_save_results(result_file, error, mtx, dist):
    """分析标定精度并生成报告"""
    if error < 0.5:
        level = "优秀"
        suggestion = "精度极高"
    elif error < 1.0:
        level = "良好"
        suggestion = "精度合格"
    elif error < 3.0:
        level = "一般"
        suggestion = "建议：增加更多角度图片、改善光照、确保标定板清晰"
    else:
        level = "较差"
        suggestion = "建议：检查标定板参数、重新采集（避免模糊和反光）、确保检测到足够角点"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Charuco 标定结果分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"重投影误差 (RMS): {error:.6f} 像素\n")
        f.write(f"精度等级: {level}\n")
        f.write(f"优化建议: {suggestion}\n\n")
        f.write("相机内参矩阵:\n")
        f.write(np.array2string(mtx, precision=4) + "\n\n")
        f.write("畸变系数:\n")
        f.write(np.array2string(dist.ravel(), precision=6) + "\n")
    return level, suggestion

def cv_imread(file_path):
    """支持中文路径的图像读取"""
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ---------- Charuco 标定相关函数 ----------
def create_charuco_board():
    """创建 Charuco 板对象及 3D 角点"""
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, aruco_dict)
    chessboard_corners_3d = board.getChessboardCorners()  # 所有棋盘格角点 (N, 3)
    # 构建每个 marker 的 3D 角点 (用于 PnP)
    marker_3d = {}
    tid = 0
    for row in range(SQUARES_Y):
        for col in range(SQUARES_X):
            if (row + col) % 2 == 0:
                cx = col * SQUARE_SIZE + SQUARE_SIZE / 2
                cy = row * SQUARE_SIZE + SQUARE_SIZE / 2
                hm = MARKER_SIZE / 2
                marker_3d[tid] = np.array([
                    [cx - hm, cy - hm, 0], [cx + hm, cy - hm, 0],
                    [cx + hm, cy + hm, 0], [cx - hm, cy + hm, 0]
                ], dtype=np.float32)
                tid += 1
    return board, chessboard_corners_3d, marker_3d

def setup_detector():
    """配置 ArUco 检测器（亚像素细化）"""
    dp = aruco.DetectorParameters()
    dp.adaptiveThreshWinSizeMin = 3
    dp.adaptiveThreshWinSizeMax = 23
    dp.adaptiveThreshWinSizeStep = 10
    dp.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = 9
    dp.cornerRefinementMaxIterations = 100
    dp.cornerRefinementMinAccuracy = 0.001
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = aruco.ArucoDetector(aruco_dict, dp)
    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    return detector, clahe, clahe_strong

def detect_markers_robust(gray, detector, clahe, clahe_strong, marker_3d):
    """鲁棒检测 ArUco 标记，返回 (obj_pts, img_pts, corners, ids)"""
    corners_mk, ids_mk, rejected = detector.detectMarkers(gray)
    if ids_mk is None or len(ids_mk) == 0:
        for cl in [clahe, clahe_strong]:
            gray_e = cl.apply(gray)
            corners_mk, ids_mk, rejected = detector.detectMarkers(gray_e)
            if ids_mk is not None and len(ids_mk) > 0:
                break
    if ids_mk is None or len(ids_mk) == 0:
        return None, None, None, None

    obj_pts = []
    img_pts = []
    for i in range(len(ids_mk)):
        mid = int(ids_mk[i][0])
        if mid in marker_3d and len(corners_mk[i]) > 0:
            obj_pts.append(marker_3d[mid])
            img_pts.append(corners_mk[i][0])

    if len(obj_pts) < 6:
        return None, None, None, None

    img_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    obj_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
    return obj_arr, img_arr, corners_mk, ids_mk

def refine_chessboard_corners(gray, marker_obj, marker_img, mtx, dist, chessboard_3d):
    """投影并细化位于图像有效区域内的棋盘格角点。"""
    obj_flat = marker_obj.reshape(-1, 3)
    img_flat = marker_img.reshape(-1, 2)
    if len(obj_flat) < 4:
        return None, None

    try:
        ret_pnp, rvec, tvec = cv2.solvePnP(obj_flat, img_flat, mtx, dist)
    except cv2.error:
        return None, None
    if not ret_pnp:
        return None, None

    projected, _ = cv2.projectPoints(chessboard_3d, rvec, tvec, mtx, dist)
    projected_flat = projected.reshape(-1, 2)
    height, width = gray.shape[:2]
    margin = 3
    valid = (
        np.isfinite(projected_flat).all(axis=1)
        & (projected_flat[:, 0] >= margin)
        & (projected_flat[:, 0] < width - margin)
        & (projected_flat[:, 1] >= margin)
        & (projected_flat[:, 1] < height - margin)
    )
    if np.count_nonzero(valid) < 6:
        return None, None

    ch_img = projected_flat[valid].reshape(-1, 1, 2).astype(np.float32)
    ch_obj = chessboard_3d[valid].reshape(-1, 1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    cv2.cornerSubPix(gray, ch_img, (margin, margin), (-1, -1), crit)

    refined_flat = ch_img.reshape(-1, 2)
    valid_refined = (
        np.isfinite(refined_flat).all(axis=1)
        & (refined_flat[:, 0] >= margin)
        & (refined_flat[:, 0] < width - margin)
        & (refined_flat[:, 1] >= margin)
        & (refined_flat[:, 1] < height - margin)
    )
    if np.count_nonzero(valid_refined) < 6:
        return None, None

    ch_img = ch_img[valid_refined]
    ch_obj = ch_obj[valid_refined]
    return ch_obj, ch_img

def compute_per_view_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """计算每张图片的重投影误差"""
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = np.sqrt(np.mean((imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)) ** 2))
        errors.append(err)
    return errors

def refine_calibration(objpoints, imgpoints, img_size, log_file=None):
    """标定并通过重投影误差筛选剔除异常图片，迭代优化精度"""
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags, criteria=criteria)

    for it in range(2):
        errors = compute_per_view_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
        mean_err = float(np.mean(errors))
        if len(objpoints) <= 5:
            break
        # 剔除误差超过均值 1.5 倍的图片
        keep = [i for i, e in enumerate(errors) if e <= mean_err * 1.5]
        if len(keep) == len(objpoints) or len(keep) < 5:
            break
        if log_file is not None:
            write_log(log_file, f"迭代{it + 1}: 剔除{len(objpoints) - len(keep)}张高误差图片，剩余{len(keep)}张")
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, mtx, dist, flags=flags, criteria=criteria)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints


def run_calibration(captured_raw_paths, mk_objpoints, mk_imgpoints, img_size,
                    chessboard_corners_3d, root_folder, subfolders, timestamp,
                    log_file):
    """使用已检测的 marker 数据完成标定并保存结果。"""
    write_log(log_file, f"\n第一阶段：利用 {len(mk_objpoints)} 张图像的标记角点进行初始标定")
    ret_mk, mtx_init, dist_init, _, _ = cv2.calibrateCamera(
        mk_objpoints, mk_imgpoints, img_size, None, None
    )
    write_log(log_file, f"初始标定 RMS: {ret_mk:.6f} px")
    write_log(log_file, f"初始内参矩阵:\n{mtx_init}")

    write_log(log_file, "\n第二阶段：投影精调棋盘格角点...")
    ch_objpoints = []
    ch_imgpoints = []
    refine_data = []

    for i, (raw_path, mk_obj, mk_img) in enumerate(zip(
            captured_raw_paths, mk_objpoints, mk_imgpoints)):
        try:
            img = cv_imread(raw_path)
        except OSError:
            img = None
        if img is None:
            write_log(log_file, f"  警告：无法读取图像 {raw_path}，跳过")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ch_obj, ch_img = refine_chessboard_corners(
            gray, mk_obj, mk_img, mtx_init, dist_init, chessboard_corners_3d
        )
        refine_data.append((gray, mk_obj, mk_img))
        image_name = os.path.basename(raw_path)
        if ch_obj is not None:
            ch_objpoints.append(ch_obj)
            ch_imgpoints.append(ch_img)
            write_log(
                log_file,
                f"  图像 {i + 1} ({image_name}): 成功提取 {len(ch_obj)} 个棋盘格角点",
            )
        else:
            ch_objpoints.append(mk_obj)
            ch_imgpoints.append(mk_img)
            write_log(
                log_file,
                f"  图像 {i + 1} ({image_name}): 棋盘格投影失败，回退使用 marker 角点",
            )

    if len(refine_data) < 5:
        write_log(log_file, f"错误：可用于精调的图像不足（{len(refine_data)}/5）")
        return None

    write_log(log_file, "\n迭代精调：用标定结果重新投影精调角点...")
    mtx_cur, dist_cur = mtx_init, dist_init
    for it in range(2):
        ch_objpoints_it = []
        ch_imgpoints_it = []
        for gray, mk_obj, mk_img in refine_data:
            ch_obj, ch_img = refine_chessboard_corners(
                gray, mk_obj, mk_img, mtx_cur, dist_cur, chessboard_corners_3d
            )
            if ch_obj is not None:
                ch_objpoints_it.append(ch_obj)
                ch_imgpoints_it.append(ch_img)
            else:
                ch_objpoints_it.append(mk_obj)
                ch_imgpoints_it.append(mk_img)

        ret_it, mtx_cur, dist_cur, _, _ = cv2.calibrateCamera(
            ch_objpoints_it, ch_imgpoints_it, img_size, mtx_cur, dist_cur
        )
        write_log(log_file, f"  迭代 {it + 1}: RMS = {ret_it:.6f} px")
        ch_objpoints = ch_objpoints_it
        ch_imgpoints = ch_imgpoints_it

    write_log(log_file, f"\n最终标定使用 {len(ch_objpoints)} 张图像（含重投影误差筛选优化）")
    ret, mtx, dist, rvecs, tvecs, ch_objpoints, ch_imgpoints = refine_calibration(
        ch_objpoints, ch_imgpoints, img_size, log_file
    )
    write_log(log_file, f"筛选后用于标定的图像数量: {len(ch_objpoints)}")

    write_log(log_file, "\n========== 最终标定结果 ==========")
    write_log(log_file, f"重投影误差 (RMS): {ret:.6f} px")
    write_log(log_file, f"内参矩阵:\n{mtx}")
    write_log(log_file, f"畸变系数:\n{dist.ravel()}")

    write_log(log_file, "\n逐图重投影误差:")
    for i in range(len(ch_objpoints)):
        proj, _ = cv2.projectPoints(ch_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = np.sqrt(np.mean(
            (ch_imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)) ** 2
        ))
        write_log(log_file, f"  筛选后图像 {i + 1}: {err:.6f} px")

    param_npz = None
    param_yaml = None
    if SAVE_PARAMS:
        param_npz = os.path.join(subfolders["results"], f"params_{timestamp}.npz")
        np.savez(param_npz, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, error=ret)
        param_yaml = os.path.join(subfolders["results"], f"params_{timestamp}.yaml")
        fs = cv2.FileStorage(param_yaml, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", mtx)
        fs.write("distortion_coefficients", dist)
        fs.write("reprojection_error", ret)
        fs.release()
        write_log(log_file, f"参数已保存至: {param_npz} 和 {param_yaml}")

    report_file = os.path.join(
        subfolders["results"], f"calibration_report_{timestamp}.txt"
    )
    level, suggestion = analyze_and_save_results(report_file, ret, mtx, dist)
    write_log(log_file, f"分析报告已保存至: {report_file}")
    write_log(log_file, f"\n精度等级: {level}\n建议: {suggestion}")

    print("\n" + "=" * 60)
    print("标定完成")
    print("=" * 60)
    print(f"重投影误差: {ret:.6f} 像素")
    print(f"精度等级: {level}")
    print(f"建议: {suggestion}")
    print("\n相机内参矩阵:")
    print(mtx)
    print("\n畸变系数:")
    print(dist.ravel())
    print(f"\n所有结果已保存至: {root_folder}")
    print("=" * 60)

    write_log(log_file, "===== 标定程序正常结束 =====")
    return {
        "error": ret,
        "mtx": mtx,
        "dist": dist,
        "view_count": len(ch_objpoints),
        "npz": param_npz,
        "yaml": param_yaml,
        "report": report_file,
    }


# ====================== 主程序 ======================
def run_interactive_calibration():
    # 创建输出目录
    root_folder, subfolders, timestamp = create_output_folders()
    log_file = os.path.join(subfolders["logs"], f"calib_charuco_{timestamp}.log")

    # 打印欢迎信息
    print("\n" + "=" * 60)
    print("CHARUCO 相机标定工具（交互式采集）")
    print("=" * 60)
    print(f"输出文件夹: {root_folder}")
    print(f"工作模式: {MODE}")
    print(f"标定板: {SQUARES_X}x{SQUARES_Y} 方格, 方格尺寸 {SQUARE_SIZE}mm, 标记尺寸 {MARKER_SIZE}mm")
    print(f"计划采集数量: {CAPTURE_COUNT} 张有效图像")
    print("\n按键说明：")
    print("  [s] - 保存当前帧（必须检测到足够标记）")
    print("  [ESC] - 退出采集并开始标定")
    print("=" * 60 + "\n")

    write_log(log_file, "===== Charuco 标定程序启动 =====")
    write_log(log_file, f"模式: {MODE} | 标定板: {SQUARES_X}x{SQUARES_Y} | 方格: {SQUARE_SIZE}mm | 标记: {MARKER_SIZE}mm")

    # 创建 Charuco 板对象和 3D 点集
    _, chessboard_corners_3d, marker_3d = create_charuco_board()
    detector, clahe, clahe_strong = setup_detector()

    # 数据存储
    mk_objpoints = []      # marker 物体点 (用于第一阶段)
    mk_imgpoints = []      # marker 图像点
    captured_raw_paths = []  # 保存的原始图像路径（用于第二阶段精调）

    # 图像源初始化
    cap = None
    image_list = []
    img_size = None

    if MODE == "usb":
        cap = cv2.VideoCapture(USB_CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        write_log(log_file, f"USB 相机已打开 (索引 {USB_CAM_INDEX})")
    elif MODE == "ip":
        cap = cv2.VideoCapture(IP_CAM_ADDRESS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        write_log(log_file, f"IP 相机已连接: {IP_CAM_ADDRESS}")
    elif MODE == "image":
        if not os.path.exists(IMAGE_PATH):
            write_log(log_file, f"错误: 图片文件夹 {IMAGE_PATH} 不存在")
            return
        image_list = sorted(
            os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        )
        write_log(log_file, f"找到 {len(image_list)} 张本地图片")
    else:
        write_log(log_file, f"错误: 未知模式 {MODE}")
        return

    # 采集循环
    count_saved = 0
    frame_idx = 0

    while True:
        # 读取图像
        if MODE in ["usb", "ip"]:
            ret, frame = cap.read()
            if not ret:
                write_log(log_file, "相机读取失败，退出")
                break
            img_raw = frame.copy()
            is_auto_mode = False
        else:  # image 模式：自动遍历，无需按键
            if frame_idx >= len(image_list):
                break
            img_raw = cv_imread(image_list[frame_idx])
            if img_raw is None:
                frame_idx += 1
                continue
            is_auto_mode = True

        img_display = img_raw.copy()
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        # 检测标记
        mk_obj, mk_img, mk_corners, mk_ids = detect_markers_robust(
            gray, detector, clahe, clahe_strong, marker_3d
        )
        detection_ok = (mk_obj is not None)

        # 更新显示状态
        if detection_ok:
            status_text = f"DETECTION: OK | SAVED: {count_saved}/{CAPTURE_COUNT}"
            color = (0, 255, 0)
            # 绘制检测到的标记
            if mk_corners is not None and mk_ids is not None:
                cv2.aruco.drawDetectedMarkers(img_display, mk_corners, mk_ids)
        else:
            status_text = "DETECTION: FAILED"
            color = (0, 0, 255)

        cv2.putText(img_display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img_display, "[s] Save | [ESC] Exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if is_auto_mode:
            cv2.putText(img_display, "AUTO MODE - processing...", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Charuco Calibration", img_display)
        key = cv2.waitKey(1) & 0xFF

        # 处理交互
        if key == 27:  # ESC
            write_log(log_file, "用户按下 ESC，结束采集")
            break

        # 保存条件：手动模式下按 's' 且检测有效；自动模式下自动保存有效帧
        should_save = False
        if is_auto_mode:
            should_save = detection_ok
        else:
            if key == ord('s') and detection_ok:
                should_save = True

        if should_save and detection_ok:
            # 保存数据
            mk_objpoints.append(mk_obj)
            mk_imgpoints.append(mk_img)

            # 保存原始图像（强制保存，确保第二阶段有数据）
            raw_path = os.path.join(subfolders["raw"], f"raw_{count_saved:03d}.jpg")
            cv2.imwrite(raw_path, img_raw)
            captured_raw_paths.append(raw_path)

            # 保存角点可视化图
            if SAVE_CORNER_IMAGES:
                corner_path = os.path.join(subfolders["corners"], f"corner_{count_saved:03d}.jpg")
                cv2.imwrite(corner_path, img_display)

            marker_count = len(mk_obj) // 4
            write_log(
                log_file,
                f"已保存有效图像 {count_saved + 1}/{CAPTURE_COUNT} (markers: {marker_count})",
            )
            count_saved += 1

            if count_saved >= CAPTURE_COUNT and not is_auto_mode:
                write_log(log_file, f"已达到设定采集数量 {CAPTURE_COUNT}，自动结束采集")
                break

        if is_auto_mode:
            frame_idx += 1
            # 自动模式结束后跳出循环
            if frame_idx >= len(image_list):
                write_log(log_file, "本地图片处理完毕")
                break

    # 释放资源
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    # 检查有效数据数量
    if len(mk_objpoints) < 5:
        write_log(log_file, f"错误：有效标记图像不足（{len(mk_objpoints)}/5），无法标定")
        return False

    result = run_calibration(
        captured_raw_paths,
        mk_objpoints,
        mk_imgpoints,
        img_size,
        chessboard_corners_3d,
        root_folder,
        subfolders,
        timestamp,
        log_file,
    )
    return result is not None


def recalibrate_saved_folder(capture_folder):
    """读取现有采集目录中的原图并重新标定，不修改原图。"""
    capture_folder = os.path.abspath(capture_folder)
    raw_folder = os.path.join(capture_folder, "raw_images")
    if not os.path.isdir(raw_folder):
        print(f"错误：采集目录缺少 raw_images 文件夹: {capture_folder}")
        return False

    image_paths = sorted(
        os.path.join(raw_folder, name) for name in os.listdir(raw_folder)
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    if len(image_paths) < 5:
        print(f"错误：raw_images 中至少需要 5 张图像，当前只有 {len(image_paths)} 张")
        return False

    root_folder, subfolders, timestamp = create_output_folders(capture_folder)
    log_file = os.path.join(subfolders["logs"], f"calib_charuco_{timestamp}.log")
    write_log(log_file, "===== Charuco 离线重标定启动 =====")
    write_log(log_file, f"采集目录: {capture_folder}")
    write_log(log_file, f"找到 {len(image_paths)} 张原始图像")
    write_log(
        log_file,
        f"标定板: {SQUARES_X}x{SQUARES_Y} | 方格: {SQUARE_SIZE}mm | 标记: {MARKER_SIZE}mm",
    )

    _, chessboard_corners_3d, marker_3d = create_charuco_board()
    detector, clahe, clahe_strong = setup_detector()
    captured_raw_paths = []
    mk_objpoints = []
    mk_imgpoints = []
    img_size = None

    for i, image_path in enumerate(image_paths):
        try:
            img = cv_imread(image_path)
        except OSError:
            img = None
        image_name = os.path.basename(image_path)
        if img is None:
            write_log(log_file, f"  图像 {i + 1} ({image_name}): 读取失败，跳过")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if img_size is None:
            img_size = current_size
        elif current_size != img_size:
            write_log(
                log_file,
                f"错误：图像 {image_name} 尺寸为 {current_size}，预期为 {img_size}",
            )
            return False

        mk_obj, mk_img, _, _ = detect_markers_robust(
            gray, detector, clahe, clahe_strong, marker_3d
        )
        if mk_obj is None:
            write_log(log_file, f"  图像 {i + 1} ({image_name}): 有效 marker 不足，跳过")
            continue

        captured_raw_paths.append(image_path)
        mk_objpoints.append(mk_obj)
        mk_imgpoints.append(mk_img)
        marker_count = len(mk_obj) // 4
        write_log(
            log_file,
            f"  图像 {i + 1} ({image_name}): 检测到 {marker_count} 个 marker",
        )

    if len(mk_objpoints) < 5:
        write_log(log_file, f"错误：有效标记图像不足（{len(mk_objpoints)}/5），无法标定")
        return False

    write_log(log_file, f"共有 {len(mk_objpoints)} 张图像进入标定")
    result = run_calibration(
        captured_raw_paths,
        mk_objpoints,
        mk_imgpoints,
        img_size,
        chessboard_corners_3d,
        root_folder,
        subfolders,
        timestamp,
        log_file,
    )
    return result is not None


def parse_args(argv=None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="ChArUco 相机标定工具")
    parser.add_argument(
        "--recalibrate-folder",
        metavar="CAPTURE_DIR",
        help="重新标定包含 raw_images 的现有采集目录",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.recalibrate_folder:
        success = recalibrate_saved_folder(args.recalibrate_folder)
    else:
        success = run_interactive_calibration()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
