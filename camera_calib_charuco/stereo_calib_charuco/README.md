# 双目 ChArUco 标定使用说明

## 准备

自动采集与移动指引请先看[新版浏览器 GUI](../README.md)。以下保留手动 CLI 流程。

在项目根目录执行本文命令。标定板和脚本配置必须一致：`11 x 8` 方格、方格边长 `20 mm`、ArUco marker 边长 `15 mm`、字典 `DICT_5X5_100`、legacy 布局。旧版 `90 / 67 mm` 板的数据必须使用旧板的真实尺寸，不能直接套用新默认值。

标定期间不要改变两台相机的相对位置。默认输入是 `2560 x 720` 左右并排画面，左目在前、右目在后，每个单目画面为 `1280 x 720`。

## 1. 采集双目图像

```bash
python3 capture_stereo.py
```

窗口中按 `s` 保存，按 `d` 删除最近一张，按 `q` 或 `Esc` 退出。脚本默认只在左右画面都检测到标定板时保存，图像写入 `stereo_images/`。

图像数量本身不代表标定质量。让标定板覆盖画面中央、四角和边缘，并改变距离、俯仰角和偏航角；每次移动后保持标定板静止再保存。新版 GUI 会根据覆盖与姿态引导采集，并使用新的图像验证已冻结的标定结果。

相机索引不是 0 时，执行：

```bash
python3 capture_stereo.py --camera 1
```

## 2. 裁剪左右图像

```bash
python3 crop_stereo.py
```

脚本从 `stereo_images/` 读取并排原图，按宽度中点裁剪，输出到：

- `charuco_left/`：左目图像。
- `charuco_right/`：右目图像。

如果设备输出顺序是右目在前、左目在后，执行：

```bash
python3 crop_stereo.py --swap
```

输出文件已存在时，脚本默认跳过。需要重新裁剪时增加 `--overwrite`。

## 3. 标定左右目内参

分别执行：

```bash
python3 calib_charuco.py --mode image --images charuco_left
python3 calib_charuco.py --mode image --images charuco_right
```

输出目录格式为：

```text
charuco_left_calib_<时间戳>/
charuco_right_calib_<时间戳>/
```

控制台会输出 fisheye、rational 和 Brown 5 参数模型的 RMS。当前镜头应使用 fisheye 结果，重点检查：

- 有效图像数量是否充足。
- fisheye RMS 是否稳定在亚像素范围。
- `previews/` 中的去畸变图像是否自然、连续。
- `results/calibration_report.txt` 是否有异常拒绝或离群记录。

双目标定会自动查找左右目最新的 `fisheye_params_*.npz`。`params_*.npz` 保存的是 rational 兼容模型，不要把它传给当前鱼眼双目标定脚本。

## 4. 标定双目外参

左右目内参完成后执行：

```bash
python3 stereo_calib.py
```

脚本默认读取 `charuco_left/`、`charuco_right/` 和最新的左右 fisheye 内参。每对图像必须同名，例如两侧都存在 `stereo_012.jpg`。

输出目录格式为：

```text
stereo_calib_<时间戳>/
```

建议按以下标准检查控制台和报告：

| 指标 | PASS 标准 |
| --- | --- |
| 双目 RMS | `< 1.0 px` |
| 估计基线 | 与标称 `60 mm` 的偏差不超过 5% |
| 垂直误差 median | `< 0.5 px` |
| 垂直误差 P95 | `< 2.0 px` |

综合质量为 `FAIL` 时不要使用该结果。先检查左右图像是否成对、相机是否移动、标定板是否模糊或只出现在画面很小的区域，再重新采集。

需要指定内参时执行：

```bash
python3 stereo_calib.py \
  --left-params charuco_left_calib_20260729_164020/results/fisheye_params_20260729_164020.npz \
  --right-params charuco_right_calib_20260729_163950/results/fisheye_params_20260729_163950.npz
```

## 5. 实时验证

连接双目相机后执行：

```bash
python3 view_stereo_undistorted.py
```

查看器自动选择最新双目标定 NPZ，并根据其中的 `model` 使用 fisheye 或 pinhole 校正接口。默认显示极线校正结果。

| 按键 | 功能 |
| --- | --- |
| `m` | 切换纯去畸变和双目极线校正。 |
| `v` | 开关标定板识别与垂直误差验证。 |
| `g` | 开关水平参考线。 |
| `s` | 保存当前无标注的校正图像。 |
| `q` / `Esc` | 退出。 |

验证时让标定板同时出现在左右画面。对应特征应落在同一水平线上，状态应为 `PASS`。如果原始视频是右目在前，执行：

```bash
python3 view_stereo_undistorted.py --swap
```

指定参数或相机索引的示例：

```bash
python3 view_stereo_undistorted.py \
  --params stereo_calib_20260729_172611/results/stereo_params_20260729_172611.npz \
  --camera 1
```

## 文件说明

### 输入与中间图像

| 路径 | 内容 |
| --- | --- |
| `stereo_images/stereo_NNN.jpg` | 相机采集的左右并排原图。 |
| `charuco_left/stereo_NNN.jpg` | 裁剪后的左目图像。 |
| `charuco_right/stereo_NNN.jpg` | 裁剪后的右目图像。 |

### 单目标定输出

`charuco_<left|right>_calib_<时间戳>/` 包含：

| 路径 | 内容 |
| --- | --- |
| `results/fisheye_params_*.npz` | 推荐使用的 OpenCV fisheye 内参，供双目标定读取。 |
| `results/fisheye_params_*.yaml` | fisheye 内参的 YAML 版本。 |
| `results/params_*.npz` | rational 兼容内参，不用于当前鱼眼双目标定。 |
| `results/calibration_report.txt` | RMS、相机矩阵、畸变系数和质量摘要。 |
| `results/per_view_metrics.csv` | 每张图像的角点数量和重投影误差。 |
| `results/rejected_images.csv` | 未参与标定的图像及原因。 |
| `results/removed_corner_outliers.csv` | 被删除的角点离群值。 |
| `results/preview_projection.yaml` | 单目预览使用的新相机矩阵。 |
| `raw_images/` | 本次单目标定使用的原图副本。 |
| `corner_images/` | ChArUco 角点检测标注图。 |
| `previews/` | fisheye 去畸变预览。 |
| `logs/` | 运行日志。 |

### 双目标定输出

`stereo_calib_<时间戳>/` 包含：

| 路径 | 内容 |
| --- | --- |
| `results/stereo_params_*.npz` | 程序读取的完整鱼眼双目标定参数。 |
| `results/stereo_params_*.yaml` | 便于查看或供 OpenCV 读取的 YAML 参数。 |
| `results/stereo_calibration_report.txt` | RMS、基线、垂直误差、使用和剔除图像对。 |
| `rectified_previews/` | 带水平参考线的双目校正预览。 |
| `logs/` | 运行日志。 |

NPZ 中常用字段：

| 字段 | 含义 |
| --- | --- |
| `model` | 相机模型，当前结果为 `fisheye`。 |
| `K1`, `D1` | 左目相机矩阵和 4 参数鱼眼畸变。 |
| `K2`, `D2` | 右目相机矩阵和 4 参数鱼眼畸变。 |
| `R`, `T` | 左目坐标到右目坐标的变换，`X_right = R * X_left + T`；`T` 单位为 mm。 |
| `R1`, `R2` | 左右目的极线校正旋转矩阵。 |
| `P1`, `P2` | 校正后的左右投影矩阵。 |
| `Q` | 视差到三维坐标的重投影矩阵。 |
| `E` | 本质矩阵。 |
| `F` | 兼容近似值，不适用于原始畸变鱼眼像素。 |
| `used_pairs`, `rejected_pairs` | 参与标定和被剔除的图像对。 |

查看器按 `s` 保存的图像位于 `undistorted_captures/`。
