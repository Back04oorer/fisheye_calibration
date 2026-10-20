"""Crop side-by-side stereo images into paired monocular images."""

import argparse
from pathlib import Path

import cv2


DEFAULT_INPUT_DIR = "stereo_images"
DEFAULT_LEFT_DIR = "charuco_left"
DEFAULT_RIGHT_DIR = "charuco_right"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    """Parse crop directories and layout options."""
    parser = argparse.ArgumentParser(
        description="把左右并排的双目图像批量裁剪为左、右单目图像。"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="双目原图目录")
    parser.add_argument("--left-output", default=DEFAULT_LEFT_DIR, help="左目输出目录")
    parser.add_argument("--right-output", default=DEFAULT_RIGHT_DIR, help="右目输出目录")
    parser.add_argument(
        "--split-x",
        type=int,
        default=None,
        help="水平裁剪位置，默认使用图像宽度中点",
    )
    parser.add_argument("--swap", action="store_true", help="交换左右输出")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出图像")
    return parser.parse_args()


def list_input_images(input_dir):
    """Return supported input images in deterministic name order."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_stereo_image(image, split_x=None):
    """Split one side-by-side image into left and right views."""
    width = image.shape[1]
    if split_x is None:
        if width < 2 or width % 2:
            raise ValueError(f"图像宽度必须是大于 1 的偶数，实际为 {width}")
        split_x = width // 2
    if split_x <= 0 or split_x >= width:
        raise ValueError(f"裁剪位置 {split_x} 超出有效范围 1..{width - 1}")
    return image[:, :split_x], image[:, split_x:]


def write_pair(left_path, right_path, left, right):
    """Write one left/right image pair."""
    if not cv2.imwrite(str(left_path), left):
        raise OSError(f"无法写入左目图像: {left_path}")
    if not cv2.imwrite(str(right_path), right):
        left_path.unlink(missing_ok=True)
        raise OSError(f"无法写入右目图像: {right_path}")


def crop_images(args):
    """Crop every input image and return a process exit code."""
    input_dir = Path(args.input)
    left_dir = Path(args.left_output)
    right_dir = Path(args.right_output)
    if not input_dir.is_dir():
        print(f"错误：双目原图目录不存在: {input_dir}")
        return 1

    input_paths = list_input_images(input_dir)
    if not input_paths:
        print(f"错误：目录中没有可处理的图像: {input_dir}")
        return 1

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    failed = 0

    for input_path in input_paths:
        left_path = left_dir / input_path.name
        right_path = right_dir / input_path.name
        if not args.overwrite and (left_path.exists() or right_path.exists()):
            print(f"跳过已存在的输出: {input_path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"失败：无法读取 {input_path}")
            failed += 1
            continue
        try:
            left, right = split_stereo_image(image, args.split_x)
            if args.swap:
                left, right = right, left
            write_pair(left_path, right_path, left, right)
        except (OSError, ValueError) as error:
            print(f"失败：{input_path.name}: {error}")
            failed += 1
            continue

        print(
            f"已裁剪: {input_path.name} -> "
            f"left {left.shape[1]}x{left.shape[0]}, "
            f"right {right.shape[1]}x{right.shape[0]}"
        )
        processed += 1

    print(f"完成：成功 {processed}，跳过 {skipped}，失败 {failed}")
    return 1 if failed else 0


def main():
    """Run the command-line entry point."""
    return crop_images(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
