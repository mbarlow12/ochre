"""Prepare detected text regions for interchangeable recognition models."""

from .file import load_image, save_segments
from .image import PreparedRegion, SegmentedPage, segment_image
from .model import Detection, detections_from_paddle

__all__ = [
    "Detection",
    "PreparedRegion",
    "SegmentedPage",
    "detections_from_paddle",
    "load_image",
    "save_segments",
    "segment_image",
]


def main() -> None:
    print("Hello from ochre!")
