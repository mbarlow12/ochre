"""Prepare independent RGB crops while retaining their page locations."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from .model import Detection

type Bounds = tuple[int, int, int, int]
type RelativeBounds = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PreparedRegion:
    """Recognition input and its source location; bounds are left, top, right, bottom.

    ``image`` is an independent RGB Pillow image, with no recognizer-specific
    resizing, padding, normalization or tensor conversion.
    """

    detection: Detection
    bounds: Bounds
    relative_bounds: RelativeBounds
    image: Image.Image


@dataclass(frozen=True, slots=True)
class SegmentedPage:
    """Prepared regions in detector order, in a single source coordinate frame."""

    width: int
    height: int
    regions: tuple[PreparedRegion, ...]


def segment_image(image: Image.Image, detections: Sequence[Detection]) -> SegmentedPage:
    """Straighten each detection from the same RGB image used for detection.

    Use ``load_image`` before detection to apply EXIF orientation and flatten
    transparency. Pixel bounds enclose the quadrilateral, are clipped to the
    page, and use exclusive right/bottom edges. Normalized bounds divide x by
    page width and y by page height. Original corners remain unmodified.
    """
    if image.mode != "RGB":
        raise ValueError("segment_image requires an RGB image; use load_image before detection")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("source image must have positive dimensions")
    indices = [detection.index for detection in detections]
    if len(set(indices)) != len(indices):
        raise ValueError("detection indices must be unique within a page")

    source = np.asarray(image)
    regions = []
    for detection in detections:
        points = np.asarray(detection.corners, dtype=np.float64)
        top_left, top_right, bottom_right, bottom_left = points
        crop_width = max(1, math.ceil(max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )))
        crop_height = max(1, math.ceil(max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )))
        destination = np.float32([
            [0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height],
        ])
        transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
        crop = cv2.warpPerspective(
            source,
            transform,
            (crop_width, crop_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        # Round outward, retaining fractional detector corners separately.
        left = max(0, min(width, math.floor(float(points[:, 0].min()))))
        top = max(0, min(height, math.floor(float(points[:, 1].min()))))
        right = max(0, min(width, math.ceil(float(points[:, 0].max()))))
        bottom = max(0, min(height, math.ceil(float(points[:, 1].max()))))
        regions.append(PreparedRegion(
            detection=detection,
            bounds=(left, top, right, bottom),
            relative_bounds=(left / width, top / height, right / width, bottom / height),
            image=Image.fromarray(crop),
        ))
    return SegmentedPage(width=width, height=height, regions=tuple(regions))
