#!/usr/bin/env python3
"""
十字形 Skybox 分割工具
支持两种十字布局：
  - 横向十字（4宽×3高）: 最常见格式
      □ Top □
  Left Front Right Back
      □ Bottom □

  - 竖向十字（3宽×4高）:
      □ Top □
      □ Front □
      □ Bottom □
      □ Back □ (旋转180°)
  Left/Right 在第二行两侧
"""

from PIL import Image
import os
import sys


def detect_layout(img_w, img_h):
    """根据宽高比判断十字类型"""
    ratio = img_w / img_h
    if abs(ratio - 4 / 3) < 0.05:
        return "horizontal"   # 4:3 横向十字
    elif abs(ratio - 3 / 4) < 0.05:
        return "vertical"     # 3:4 竖向十字
    else:
        return None


def crop_face(img, col, row, face_w, face_h):
    """裁剪指定格子的图像"""
    x = col * face_w
    y = row * face_h
    return img.crop((x, y, x + face_w, y + face_h))


def split_horizontal_cross(img, out_dir, fmt):
    """
    横向十字布局 (4列 × 3行)：
    列/行   0       1       2       3
      0   [空]    Top    [空]    [空]
      1   Left   Front  Right   Back
      2   [空]   Bottom [空]    [空]
    """
    w, h = img.size
    fw, fh = w // 4, h // 3

    faces = {
        "top":    (1, 0),
        "left":   (0, 1),
        "front":  (1, 1),
        "right":  (2, 1),
        "back":   (3, 1),
        "bottom": (1, 2),
    }

    results = {}
    for name, (col, row) in faces.items():
        face = crop_face(img, col, row, fw, fh)
        path = os.path.join(out_dir, f"{name}.{fmt}")
        face.save(path)
        results[name] = path
        print(f"  保存 {name:8s} → {path}  ({fw}×{fh})")

    return results


def split_vertical_cross(img, out_dir, fmt):
    """
    竖向十字布局 (3列 × 4行)：
    列/行   0       1       2
      0   [空]    Top    [空]
      1   Left   Front  Right
      2   [空]   Bottom [空]
      3   [空]   Back   [空]  ← 旋转180°
    """
    w, h = img.size
    fw, fh = w // 3, h // 4

    faces_raw = {
        "top":    (1, 0, 0),
        "left":   (0, 1, 0),
        "front":  (1, 1, 0),
        "right":  (2, 1, 0),
        "bottom": (1, 2, 0),
        "back":   (1, 3, 180),   # back 在底部，需旋转
    }

    results = {}
    for name, (col, row, rotate) in faces_raw.items():
        face = crop_face(img, col, row, fw, fh)
        if rotate:
            face = face.rotate(rotate)
        path = os.path.join(out_dir, f"{name}.{fmt}")
        face.save(path)
        results[name] = path
        rot_str = f" (旋转{rotate}°)" if rotate else ""
        print(f"  保存 {name:8s} → {path}  ({fw}×{fh}){rot_str}")

    return results


def split_skybox(input_path, out_dir=None, fmt="png", layout=None):
    """
    主函数：分割十字形 skybox 图

    Args:
        input_path: 输入图片路径
        out_dir:    输出目录（默认与输入文件同目录下的 skybox_faces/）
        fmt:        输出格式，png / jpg / tga 等
        layout:     强制指定布局 "horizontal" / "vertical"，None 则自动检测
    """
    if not os.path.isfile(input_path):
        print(f"[错误] 文件不存在：{input_path}")
        sys.exit(1)

    img = Image.open(input_path).convert("RGBA" if fmt == "png" else "RGB")
    w, h = img.size
    print(f"图片尺寸：{w} × {h}")

    # 自动检测布局
    if layout is None:
        layout = detect_layout(w, h)
        if layout is None:
            print(f"[警告] 无法自动识别十字类型（宽高比 {w/h:.3f}）")
            print("  请手动指定 layout='horizontal' 或 layout='vertical'")
            sys.exit(1)
        print(f"自动识别布局：{layout} cross")
    else:
        print(f"使用指定布局：{layout} cross")

    # 准备输出目录
    if out_dir is None:
        base = os.path.splitext(input_path)[0]
        out_dir = base + "_faces"
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录：{out_dir}\n")

    # 分割
    if layout == "horizontal":
        results = split_horizontal_cross(img, out_dir, fmt)
    else:
        results = split_vertical_cross(img, out_dir, fmt)

    print(f"\n完成！共输出 {len(results)} 张面图。")
    return results


# ── 直接运行示例 ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="十字形 Skybox 分割工具")
    parser.add_argument("input",  help="输入图片路径（十字形 skybox）")
    parser.add_argument("-o", "--output", default=None, help="输出目录（默认：输入文件名_faces/）")
    parser.add_argument("-f", "--format", default="png",
                        choices=["png", "jpg", "tga", "bmp"], help="输出格式（默认 png）")
    parser.add_argument("-l", "--layout", default=None,
                        choices=["horizontal", "vertical"], help="强制指定布局（默认自动检测）")
    args = parser.parse_args()

    split_skybox(args.input, args.output, args.format, args.layout)