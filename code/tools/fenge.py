#!/usr/bin/env python3
"""
Cross-shaped skybox strip splitter.

Supports two layouts:
  - Horizontal cross (4 wide × 3 tall), common format:
      [ ] Top [ ]
  Left Front Right Back
      [ ] Bottom [ ]

  - Vertical cross (3 wide × 4 tall):
      [ ] Top [ ]
      [ ] Front [ ]
      [ ] Bottom [ ]
      [ ] Back [ ]  (rotated 180°)
  Left / Right sit on the sides of the second row.
"""

from PIL import Image
import os
import sys


def detect_layout(img_w, img_h):
    """Infer cross layout from aspect ratio."""
    ratio = img_w / img_h
    if abs(ratio - 4 / 3) < 0.05:
        return "horizontal"  # 4:3 horizontal cross
    elif abs(ratio - 3 / 4) < 0.05:
        return "vertical"  # 3:4 vertical cross
    else:
        return None


def crop_face(img, col, row, face_w, face_h):
    """Crop one cell from the strip."""
    x = col * face_w
    y = row * face_h
    return img.crop((x, y, x + face_w, y + face_h))


def split_horizontal_cross(img, out_dir, fmt):
    """
    Horizontal cross (4 columns × 3 rows):
    col/row   0       1       2       3
      0   [ ]     Top     [ ]     [ ]
      1   Left   Front  Right   Back
      2   [ ]    Bottom  [ ]     [ ]
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
        print(f"  saved {name:8s} -> {path}  ({fw}x{fh})")

    return results


def split_vertical_cross(img, out_dir, fmt):
    """
    Vertical cross (3 columns × 4 rows):
    col/row   0       1       2
      0   [ ]     Top     [ ]
      1   Left   Front  Right
      2   [ ]    Bottom  [ ]
      3   [ ]     Back   [ ]  <- rotate 180°
    """
    w, h = img.size
    fw, fh = w // 3, h // 4

    faces_raw = {
        "top":    (1, 0, 0),
        "left":   (0, 1, 0),
        "front":  (1, 1, 0),
        "right":  (2, 1, 0),
        "bottom": (1, 2, 0),
        "back":   (1, 3, 180),  # back row: rotate 180°
    }

    results = {}
    for name, (col, row, rotate) in faces_raw.items():
        face = crop_face(img, col, row, fw, fh)
        if rotate:
            face = face.rotate(rotate)
        path = os.path.join(out_dir, f"{name}.{fmt}")
        face.save(path)
        results[name] = path
        rot_str = f" (rotated {rotate}°)" if rotate else ""
        print(f"  saved {name:8s} -> {path}  ({fw}x{fh}){rot_str}")

    return results


def split_skybox(input_path, out_dir=None, fmt="png", layout=None):
    """
    Split a cross-shaped skybox image into six faces.

    Args:
        input_path: Path to the input image.
        out_dir: Output directory (default: <input_basename>_faces next to the file).
        fmt: Output format: png / jpg / tga / bmp.
        layout: Force "horizontal" or "vertical"; None = auto-detect from aspect ratio.
    """
    if not os.path.isfile(input_path):
        print(f"[error] file not found: {input_path}")
        sys.exit(1)

    img = Image.open(input_path).convert("RGBA" if fmt == "png" else "RGB")
    w, h = img.size
    print(f"image size: {w} x {h}")

    if layout is None:
        layout = detect_layout(w, h)
        if layout is None:
            print(f"[warn] cannot detect cross layout (aspect {w/h:.3f})")
            print("  pass -l horizontal or -l vertical")
            sys.exit(1)
        print(f"auto layout: {layout} cross")
    else:
        print(f"layout: {layout} cross")

    if out_dir is None:
        base = os.path.splitext(input_path)[0]
        out_dir = base + "_faces"
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {out_dir}\n")

    if layout == "horizontal":
        results = split_horizontal_cross(img, out_dir, fmt)
    else:
        results = split_vertical_cross(img, out_dir, fmt)

    print(f"\ndone: wrote {len(results)} face images.")
    return results


# --- CLI ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a cross-shaped skybox strip into six cube faces")
    parser.add_argument("input", help="input image path (cross skybox layout)")
    parser.add_argument("-o", "--output", default=None, help="output directory (default: <input_stem>_faces/)")
    parser.add_argument("-f", "--format", default="png",
                        choices=["png", "jpg", "tga", "bmp"], help="output format (default png)")
    parser.add_argument("-l", "--layout", default=None,
                        choices=["horizontal", "vertical"], help="force layout (default: auto)")
    args = parser.parse_args()

    split_skybox(args.input, args.output, args.format, args.layout)
