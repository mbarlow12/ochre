# Ochre

Prepare detected text regions for recognition. Segmentation uses the current
PaddleOCR text detector; no recognition model is loaded or called.

## Try the detector and crops

```sh
uv sync
uv run --script scripts/spike_paddle_ocr.py
uv run --script scripts/spike_paddle_ocr.py scripts/ochre-handwriting-sample-01-2-of-2.jpg --output-dir output-det/page-2
```

The first invocation uses the bundled first handwriting sample and writes to
`output-det`. The demo runs `PP-OCRv6_tiny_det` on CPU with the existing dilation,
box threshold (`0.6`), and unclip ratio (`1.4`) settings. Paddle downloads detector
weights on first use.

Each run saves straightened RGB PNGs (`region-0000.png`, etc.), `manifest.json`,
the Paddle detection JSON (`res.json`), and an annotated detection preview.
Use a separate output directory for each page. Reusing a directory replaces
matching files; use the manifest to identify the current run's crops.

## Library API

```python
from ochre import detections_from_paddle, load_image, save_segments, segment_image

# Use this same oriented source for detection and cropping. Convert RGB to BGR
# when supplying its numpy array to Paddle (see the demo).
source = load_image("notes.jpg")
# detection_result is one result from TextDetection.predict, not full PaddleOCR.
detections = detections_from_paddle(detection_result)
page = segment_image(source, detections)

for region in page.regions:
    crop = region.image  # independent RGB PIL.Image, ready for a future adapter
    print(region.detection.index, region.bounds, region.relative_bounds)

save_segments(page, "output-det/notes", source_path="notes.jpg")
```

You can also construct `Detection(index, confidence, corners)` directly. The
package does not import Paddle or any recognition framework. Its output contract
is `SegmentedPage(width, height, regions)` containing `PreparedRegion` records,
so future recognizer adapters can own resizing, normalization, batching, tensor
conversion, and inference without changing segmentation.

### Coordinates and geometry

- `load_image` applies EXIF orientation and composites transparency onto white.
  Every detection and page coordinate refers to that oriented source, whose
  dimensions appear as `[width, height]` in the manifest's `source_size`.
- Detection `corners` are four source-pixel `(x, y)` pairs in top-left,
  top-right, bottom-right, bottom-left order. Paddle already returns coordinates
  in its input image's frame; do not rescale them again.
- `bounds` are `(left, top, right, bottom)`, rounded outward and clipped to the
  page, with exclusive right/bottom edges. `relative_bounds` divide horizontal
  coordinates by page width and vertical coordinates by page height, giving
  values in `[0, 1]`. Original corners are retained even at image boundaries.
- Each crop is perspective-rectified with dimensions derived from the longer
  of each pair of opposing edges, rounded up. Areas outside the source are
  white. There is no extra padding or automatic 90-degree rotation.
- Regions retain detector order. Their IDs are original detector indices,
  local to the page; this stage does not determine reading order.
- Empty detections return an empty page and export an empty `regions` list.
  Invalid quadrilaterals, confidence values, or mismatched polygon/score counts
  raise `ValueError` instead of silently dropping regions.

The manifest links each region's `id`, `confidence`, `corners`, `bounds`, and
`relative_bounds` to its PNG `file` and `[width, height]` `crop_size`.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Geometry, coordinate, image loading, and export tests use synthetic fixtures and
need no model weights or network access.
