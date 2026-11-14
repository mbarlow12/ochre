"""Model-neutral detections and the adapter for PaddleOCR text detection."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

type Point = tuple[float, float]
type Quadrilateral = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class Detection:
    """A page-local detection with source-pixel corners in TL, TR, BR, BL order.

    ``index`` is the original detector index, not a reading-order position.
    Coordinates refer to the same oriented image that was sent to the detector.
    """

    index: int
    confidence: float
    corners: Quadrilateral

    def __post_init__(self) -> None:
        prefix = f"detection {self.index}"
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"{prefix}: index must be a nonnegative integer")
        try:
            points = np.asarray(self.corners, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}: corners must be four numeric (x, y) pairs") from exc
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"{prefix}: corners must be four finite (x, y) pairs")
        edges = np.roll(points, -1, axis=0) - points
        following = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not (turns > 0).all():
            raise ValueError(f"{prefix}: corners must form a convex TL, TR, BR, BL quadrilateral")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}: confidence must be a number between 0 and 1") from exc
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{prefix}: confidence must be a finite number between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "corners", tuple(tuple(float(v) for v in point) for point in points))


def detections_from_paddle(result: Mapping[str, Any]) -> tuple[Detection, ...]:
    """Extract paired ``dt_polys`` / ``dt_scores`` without loading Paddle.

    Paddle already returns source-image coordinates. No resizing or sorting is
    applied here. Both plain mappings and live Paddle result objects work.
    """
    try:
        polygons, scores = result["dt_polys"], result["dt_scores"]
    except KeyError as exc:
        raise ValueError(f"Paddle detection result is missing {exc.args[0]}") from exc
    if len(polygons) != len(scores):
        raise ValueError("Paddle dt_polys and dt_scores must have matching counts")
    return tuple(
        Detection(index=index, confidence=score, corners=polygon)
        for index, (polygon, score) in enumerate(zip(polygons, scores, strict=True))
    )
