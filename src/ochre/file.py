"""Image loading and optional filesystem export for prepared regions."""

import json
from pathlib import Path

from PIL import Image, ImageOps

from .image import SegmentedPage


def load_image(path: str | Path) -> Image.Image:
    """Decode an oriented RGB source once, compositing transparency onto white."""
    with Image.open(path) as decoded:
        oriented = ImageOps.exif_transpose(decoded).convert("RGBA")
        background = Image.new("RGBA", oriented.size, "white")
        return Image.alpha_composite(background, oriented).convert("RGB")


def save_segments(
    page: SegmentedPage,
    output_dir: str | Path,
    *,
    source_path: str | Path | None = None,
) -> Path:
    """Save PNG crops and a manifest; return the manifest path.

    Filenames use original detector indices. Reusing a directory replaces
    matching exports; the manifest lists only regions from this invocation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = []
    for region in page.regions:
        detection = region.detection
        filename = f"region-{detection.index:04d}.png"
        region.image.save(output_dir / filename, format="PNG")
        regions.append({
            "id": detection.index,
            "confidence": detection.confidence,
            "corners": detection.corners,
            "bounds": region.bounds,
            "relative_bounds": region.relative_bounds,
            "crop_size": region.image.size,
            "file": filename,
        })
    manifest = {
        "source_path": str(Path(source_path).resolve()) if source_path is not None else None,
        "source_size": [page.width, page.height],
        "regions": regions,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path
