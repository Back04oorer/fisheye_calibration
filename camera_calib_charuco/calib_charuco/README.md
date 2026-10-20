# ChArUco 相机标定工具

本项目使用 ChArUco 标定板计算单目相机内参，并实时比较原图与去畸变图。当前标定模型为 OpenCV 针孔模型，畸变参数为 `k1、k2、p1、p2、k3`。

## 前置条件

- Python 3。
- NumPy。
- 带 `aruco` 模块的 OpenCV，例如 `opencv-contrib-python`。
- 标定板尺寸、Marker 尺寸、字典和实际标定板一致。

## 标定

先检查 `calib_charuco.py` 顶部的配置：

```python
MODE = "usb"          # image / usb / ip
USB_CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAPTURE_COUNT = 30
```

标定板默认参数为：

- `12 x 9` 个方格。
- 方格边长 `30 mm`。
- Marker 边长 `24 mm`。
- ArUco 字典 `DICT_5X5_100`。

启动交互式标定：

```bash
python3 calib_charuco.py
```

画面检测成功后按 `S` 保存当前帧，按 `Esc` 结束采集并开始标定。至少需要 5 张有效图像；建议让标定板覆盖画面中央、四角和边缘，并改变距离与倾斜角度。

标定完成后，程序创建带时间戳的目录：

```text
charuco_calib_YYYYMMDD_HHMMSS/
```

已有采集图像也可以离线重新标定：

```bash
python3 calib_charuco.py \
  --recalibrate-folder charuco_calib_YYYYMMDD_HHMMSS
```

指定目录必须包含 `raw_images/`，且所有图像尺寸必须一致。

## 验证

使用标定结果打开原图与去畸变图对比窗口：

```bash
python3 view_undistortion.py \
  --params charuco_calib_YYYYMMDD_HHMMSS \
  --camera-index 0 \
  --frame-size 640 480 \
  --calibration-size 640 480
```

参数说明：

- `--params`：参数文件，或包含 `results/` 的标定目录。建议始终显式指定，避免使用失效的默认目录。
- `--camera-index`：待验证的相机索引。
- `--frame-size`：向相机请求的视频分辨率。
- `--calibration-size`：生成标定参数时使用的图像分辨率，必须与标定图像一致。

当前参数文件没有记录图像尺寸，因此验证 640 x 480 标定结果时应显式传入 `--calibration-size 640 480`。程序收到视频帧后，会按相机实际返回的帧尺寸生成去畸变映射；如果宽高比与标定图像不同，会输出警告。

对比时重点观察画面边缘的直线。去畸变图中的直线应更接近直线，画面不应出现明显拉伸或局部变形。按 `Q` 或 `Esc` 退出。

## 文件说明

```text
charuco_calib_YYYYMMDD_HHMMSS/
|-- raw_images/       原始标定图像
|-- corner_images/    Marker 检测结果图
|-- logs/             标定过程日志
`-- results/          标定参数和结果报告
```

`results/` 中的文件：

| 文件 | 内容 |
| --- | --- |
| `params_*.npz` | NumPy 格式；包含内参 `mtx`、畸变系数 `dist`、每幅图的 `rvecs/tvecs` 和总 RMS `error`。 |
| `params_*.yaml` | OpenCV 格式；包含 `camera_matrix`、`distortion_coefficients` 和 `reprojection_error`。 |
| `calibration_report_*.txt` | 内参、畸变系数、RMS 和精度等级摘要。 |

日志记录每个阶段的 RMS、有效角点数量、被剔除的高误差图像数量和最终参数。若查看器提示“参数路径不存在”，先查找当前已有参数：

```bash
find . -path '*/results/params_*.yaml' -print
```

然后将对应的标定目录传给 `--params`。
