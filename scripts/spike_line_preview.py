#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "pillow>=11,<14", "scipy>=1.13,<2"]
# ///

"""Draw line bounds on one-column notes; no OCR or model downloads.

    uv run --no-project --script spike_line_preview.py notes.jpg -o preview.png

Teal = line proposal; orange = suspect proposal (heuristics, not confidence).
Increase --gap when one line splits; decrease it when adjacent lines merge.
Lower --offset to detect fainter ink, at the cost of more background noise.
This desktop evaluator uses SciPy; it does not establish mobile suitability.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi


def _deskew(mask):
    thumb = Image.fromarray(mask.astype("uint8") * 255)
    thumb.thumbnail((600, 600), Image.Resampling.NEAREST)

    def score(angle):
        rotated = thumb.rotate(float(angle), expand=True)
        rows = (np.asarray(rotated) > 0).sum(axis=1).astype(float)
        return float(np.dot(rows, rows) / max(rows.sum(), 1))

    baseline = score(0)
    coarse = list(range(-5, 6))
    best = max(coarse, key=score)
    fine = np.arange(max(-5, best - 0.75), min(5, best + 0.75) + 0.01, 0.25)
    angle = float(max(fine, key=score))
    return angle if abs(angle) >= 0.25 and score(angle) > baseline * 1.05 else 0.0


def segment(image, *, max_side=1600, offset=10.0, gap=0.35, deskew=True):
    """Return source-coordinate line quadrilaterals and the deskew correction."""
    if max_side < 64 or offset < 0 or gap < 0:
        raise ValueError("max_side must be >=64; offset and gap must be nonnegative")
    gray = image.convert("L")
    gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width, height = gray.size
    pixels = np.asarray(gray, dtype=np.float32)
    window = max(15, int(min(width, height) / 15) | 1)
    mask = pixels < ndi.uniform_filter(pixels, size=window) - offset
    angle = _deskew(mask) if deskew and mask.any() else 0.0
    mask_image = Image.fromarray(mask.astype("uint8") * 255)
    mask = np.asarray(mask_image.rotate(angle, expand=True)) > 0
    rotated_height, rotated_width = mask.shape

    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())
    components = []
    for label, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is not None and areas[label] >= 3:
            ys, xs = slices
            components.append((xs.start, ys.start, xs.stop, ys.stop))
    if not components:
        return [], angle
    heights = [b - t for l, t, r, b in components if b - t >= 4 and r - l < width * 0.8]
    writing_height = float(np.median(heights)) if heights else 12.0

    keep = areas >= 3
    keep[0] = False
    profile = keep[labels].sum(axis=1).astype(float)
    profile = ndi.uniform_filter1d(profile, size=max(1, round(writing_height * 0.2)))
    active = profile > max(1.0, profile.max() * 0.025)
    edges = np.diff(np.r_[False, active, False].astype("int8"))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    bands = []
    for top, bottom in zip(starts, stops):
        if bands and top - bands[-1][1] <= max(1, writing_height * gap):
            bands[-1][1] = int(bottom)
        else:
            bands.append([int(top), int(bottom)])
    if not bands:
        return [], angle

    band_array = np.asarray(bands)
    centers = band_array.mean(axis=1)
    groups = [[] for _ in bands]
    suspect = [False for _ in bands]
    for left, top, right, bottom in components:
        overlaps = np.maximum(0, np.minimum(bottom, band_array[:, 1]) - np.maximum(top, band_array[:, 0]))
        index = int(np.argmax(overlaps))
        if not overlaps.any():
            index = int(np.argmin(abs(centers - (top + bottom) / 2)))
            if abs(centers[index] - (top + bottom) / 2) > writing_height:
                continue
        if (overlaps > 0).sum() > 1:
            for touched in np.flatnonzero(overlaps):
                suspect[int(touched)] = True
        groups[index].append((left, top, right, bottom))

    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    inverse = np.array([[cosine, sine], [-sine, cosine]])
    regions = []
    for index, group in enumerate(groups):
        if not group:
            continue
        bounds = np.asarray(group)
        left, top = bounds[:, :2].min(axis=0)
        right, bottom = bounds[:, 2:].max(axis=0)
        padding = max(2, round((bottom - top) * 0.15))
        box = (left - padding, top - padding, right + padding, bottom + padding)
        l, t, r, b = box
        corners = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=float)
        corners -= [rotated_width / 2, rotated_height / 2]
        corners = corners @ inverse + [width / 2, height / 2]
        corners *= [image.width / width, image.height / height]
        suspect[index] |= bottom - top > writing_height * 2.3
        if index and top - padding < bands[index - 1][1]:
            suspect[index] = True
        if index + 1 < len(bands) and bottom + padding > bands[index + 1][0]:
            suspect[index] = True
        regions.append({"points": corners.tolist(), "suspect": bool(suspect[index])})
    return regions, angle


def preview(input_path, output_path, **settings):
    """Write an annotated PNG while keeping the input file unchanged."""
    source_path, target_path = Path(input_path), Path(output_path)
    if source_path.resolve() == target_path.resolve():
        raise ValueError("output must differ from the input file")
    with Image.open(source_path) as decoded:
        if decoded.format not in {"JPEG", "PNG"}:
            raise ValueError("input must be JPEG or PNG")
        oriented = ImageOps.exif_transpose(decoded).convert("RGBA")
        source = Image.new("RGBA", oriented.size, "white")
        source.alpha_composite(oriented)
        source = source.convert("RGB")
    regions, angle = segment(source, **settings)
    annotated = source.copy()
    draw = ImageDraw.Draw(annotated)
    stroke = max(2, round(max(source.size) / 600))
    font = ImageFont.load_default(size=max(14, stroke * 5))
    flagged = []
    for index, region in enumerate(regions, start=1):
        points = [tuple(point) for point in region["points"]]
        color = "#db6800" if region["suspect"] else "#008f83"
        draw.line(points + [points[0]], fill=color, width=stroke)
        label = str(index) + ("?" if region["suspect"] else "")
        x = max(0, min(point[0] for point in points))
        y = max(0, min(point[1] for point in points) - font.size - 5)
        bounds = draw.textbbox((x, y), label, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((x, y), label, font=font, fill="white")
        if region["suspect"]:
            flagged.append(index)
    annotated.save(target_path, format="PNG")
    print(f"{target_path}: {len(regions)} lines; correction {angle:+.2f} degrees; suspect {flagged}")
    return len(regions), angle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="PNG path; default: <input-stem>.lines.png")
    parser.add_argument("--max-side", type=int, default=1600, help="analysis image's longest side (default: 1600)")
    parser.add_argument("--offset", type=float, default=10.0, help="adaptive threshold offset; lower finds fainter ink")
    parser.add_argument("--gap", type=float, default=0.35, help="merge gaps up to this fraction of writing height")
    parser.add_argument("--no-deskew", action="store_true", help="disable small-angle correction")
    args = parser.parse_args()
    output = args.output or args.image.with_name(args.image.stem + ".lines.png")
    preview(args.image, output, max_side=args.max_side, offset=args.offset, gap=args.gap, deskew=not args.no_deskew)


if __name__ == "__main__":
    main()
