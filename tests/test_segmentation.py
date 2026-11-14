import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from ochre import Detection, detections_from_paddle, load_image, save_segments, segment_image


def rectangle(left, top, right, bottom):
    return ((left, top), (right, top), (right, bottom), (left, bottom))


class DetectionTests(unittest.TestCase):
    def test_package_import_does_not_load_model_frameworks(self):
        for module in ("paddle", "paddleocr", "paddlex", "torch", "transformers"):
            with self.subTest(module=module):
                self.assertNotIn(module, sys.modules)

    def test_paddle_arrays_preserve_coordinates_order_and_scores(self):
        polygons = np.array([rectangle(50, 40, 90, 60), rectangle(5, 2, 30, 12)], dtype=np.int16)
        scores = np.array([0.91, 0.72], dtype=np.float32)
        detections = detections_from_paddle({"dt_polys": polygons, "dt_scores": scores})
        for index, detection in enumerate(detections):
            self.assertEqual(detection.index, index)
            self.assertEqual(detection.corners, tuple(map(tuple, polygons[index])))
            self.assertAlmostEqual(detection.confidence, scores[index], places=6)
        # Metadata owns its coordinates rather than retaining Paddle's buffer.
        polygons[:] = 0
        self.assertEqual(detections[0].corners, rectangle(50, 40, 90, 60))

    def test_empty_paddle_arrays(self):
        for polygons in ([], np.array([]), np.empty((0, 4, 2))):
            with self.subTest(shape=np.shape(polygons)):
                self.assertEqual(detections_from_paddle({"dt_polys": polygons, "dt_scores": []}), ())

    def test_mismatched_counts_and_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "matching counts"):
            detections_from_paddle({"dt_polys": [rectangle(0, 0, 10, 5)], "dt_scores": []})
        with self.assertRaisesRegex(ValueError, "dt_scores"):
            detections_from_paddle({"dt_polys": []})

    def test_invalid_geometry_identifies_detection(self):
        invalid = [
            [(0, 0), (1, 1)],
            rectangle(0, 0, 0, 10),
            ((0, 0), (10, 10), (0, 10), (10, 0)),
            ((0, 0), (0, 10), (10, 10), (10, 0)),
            ((0, 0), (10, 0), (5, 2), (0, 10)),
            rectangle(0, 0, float("nan"), 10),
            rectangle(0, 0, float("inf"), 10),
        ]
        for corners in invalid:
            with self.subTest(corners=corners), self.assertRaisesRegex(ValueError, "detection 7"):
                Detection(7, 0.9, corners)

    def test_invalid_confidence(self):
        for confidence in (float("nan"), float("inf"), -0.1, 1.1, None):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(ValueError, "detection 3"):
                Detection(3, confidence, rectangle(0, 0, 10, 5))


class GeometryTests(unittest.TestCase):
    def test_rectangle_matches_source_pixels_and_location(self):
        pixels = np.arange(80 * 60 * 3, dtype=np.uint8).reshape(60, 80, 3)
        source = Image.fromarray(pixels)
        detection = Detection(4, 0.9, rectangle(10, 20, 70, 50))
        page = segment_image(source, [detection])
        self.assertEqual((page.width, page.height), (80, 60))
        region, = page.regions
        self.assertIs(region.detection, detection)
        self.assertEqual(region.bounds, (10, 20, 70, 50))
        self.assertEqual(region.relative_bounds, (10 / 80, 20 / 60, 70 / 80, 50 / 60))
        self.assertEqual(region.image.mode, "RGB")
        self.assertEqual(region.image.size, (60, 30))
        np.testing.assert_array_equal(np.asarray(region.image), pixels[20:50, 10:70])
        np.testing.assert_array_equal(np.asarray(source), pixels)
        region.image.paste("white", (0, 0, 60, 30))
        np.testing.assert_array_equal(np.asarray(source), pixels)

    def test_tilted_and_perspective_regions_preserve_corner_colors(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        locations = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        quads = [
            ((35, 30), (175, 60), (165, 100), (25, 70)),
            ((25, 25), (180, 45), (155, 120), (45, 100)),
        ]
        for corners in quads:
            with self.subTest(corners=corners):
                source = Image.new("RGB", (220, 160), "white")
                draw = ImageDraw.Draw(source)
                points = np.array(corners, dtype=float)

                def position(u, v):
                    top = points[0] * (1 - u) + points[1] * u
                    bottom = points[3] * (1 - u) + points[2] * u
                    return tuple(top * (1 - v) + bottom * v)

                # Paint corner blocks in source space, independently of the warp.
                for (u, v), color in zip(locations, colors, strict=True):
                    draw.polygon([
                        position(u - 0.18, v - 0.18), position(u + 0.18, v - 0.18),
                        position(u + 0.18, v + 0.18), position(u - 0.18, v + 0.18),
                    ], fill=color)
                region, = segment_image(source, [Detection(0, 1, corners)]).regions
                for (u, v), color in zip(locations, colors, strict=True):
                    actual = region.image.getpixel((int(u * region.image.width), int(v * region.image.height)))
                    np.testing.assert_allclose(actual, color, atol=5)

    def test_page_edges_fill_white_and_keep_original_corners(self):
        corners = rectangle(-5, -3, 15, 12)
        region, = segment_image(Image.new("RGB", (10, 10), (20, 40, 60)), [Detection(0, 1, corners)]).regions
        self.assertEqual(region.image.size, (20, 15))
        self.assertEqual(region.image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(region.image.getpixel((19, 14)), (255, 255, 255))
        self.assertEqual(region.image.getpixel((8, 6)), (20, 40, 60))
        self.assertEqual(region.detection.corners, corners)
        self.assertEqual(region.bounds, (0, 0, 10, 10))
        self.assertEqual(region.relative_bounds, (0, 0, 1, 1))

    def test_fractional_bounds_round_outward(self):
        region, = segment_image(
            Image.new("RGB", (100, 200)), [Detection(0, 1, rectangle(1.2, 3.5, 10.1, 20.7))],
        ).regions
        self.assertEqual(region.bounds, (1, 3, 11, 21))
        self.assertEqual(region.relative_bounds, (0.01, 0.015, 0.11, 0.105))

    def test_bounds_keep_precision_near_integer_edges(self):
        region, = segment_image(
            Image.new("RGB", (100, 100)),
            [Detection(0, 1, rectangle(1.99999999, 3.99999999, 10.00000001, 20.00000001))],
        ).regions
        self.assertEqual(region.bounds, (1, 3, 11, 21))

    def test_order_is_not_reading_order_and_tall_crops_are_not_rotated(self):
        detections = [Detection(8, 0.8, rectangle(30, 20, 40, 70)), Detection(2, 0.9, rectangle(1, 1, 20, 10))]
        page = segment_image(Image.new("RGB", (80, 80)), detections)
        self.assertEqual([region.detection.index for region in page.regions], [8, 2])
        self.assertEqual(page.regions[0].image.size, (10, 50))

    def test_empty_page(self):
        page = segment_image(Image.new("RGB", (30, 40)), [])
        self.assertEqual((page.width, page.height, page.regions), (30, 40, ()))

    def test_reject_duplicate_ids_and_unnormalized_images(self):
        detection = Detection(0, 1, rectangle(0, 0, 5, 5))
        with self.assertRaisesRegex(ValueError, "unique"):
            segment_image(Image.new("RGB", (10, 10)), [detection, detection])
        with self.assertRaisesRegex(ValueError, "RGB"):
            segment_image(Image.new("RGBA", (10, 10)), [])
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            segment_image(Image.new("RGB", (0, 10)), [])


class FileTests(unittest.TestCase):
    def test_load_orients_exif_and_does_not_change_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.png"
            pixels = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(2, 3, 3)
            original = Image.fromarray(pixels)
            exif = Image.Exif()
            exif[274] = 6
            original.save(path, exif=exif)
            file_bytes = path.read_bytes()
            loaded = load_image(path)
            self.assertEqual(loaded.size, (2, 3))
            self.assertEqual(loaded.mode, "RGB")
            np.testing.assert_array_equal(loaded, original.transpose(Image.Transpose.ROTATE_270))
            self.assertEqual(path.read_bytes(), file_bytes)

    def test_load_flattens_transparency_and_accepts_grayscale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            rgba = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
            rgba.putpixel((1, 0), (255, 0, 0, 128))
            rgba.putpixel((2, 0), (0, 0, 255, 255))
            rgba.save(path)
            loaded = load_image(path)
            self.assertEqual(loaded.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(loaded.getpixel((1, 0)), (255, 127, 127))
            self.assertEqual(loaded.getpixel((2, 0)), (0, 0, 255))
            Image.new("L", (2, 2), 37).save(path)
            self.assertEqual(load_image(path).getpixel((0, 0)), (37, 37, 37))

    def test_export_round_trip_manifest_and_crops(self):
        source = Image.new("RGB", (80, 60), (10, 20, 200))
        page = segment_image(source, [
            Detection(9, 0.8, rectangle(40, 30, 70, 50)),
            Detection(1, 0.9, rectangle(10, 5, 30, 15)),
        ])
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = save_segments(page, Path(directory) / "nested", source_path="notes.jpg")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["source_path"], str(Path("notes.jpg").resolve()))
            self.assertEqual(manifest["source_size"], [80, 60])
            self.assertEqual([entry["id"] for entry in manifest["regions"]], [9, 1])
            for region, entry in zip(page.regions, manifest["regions"], strict=True):
                self.assertEqual(entry["file"], f"region-{region.detection.index:04d}.png")
                self.assertEqual(entry["confidence"], region.detection.confidence)
                self.assertEqual(entry["corners"], list(map(list, region.detection.corners)))
                self.assertEqual(entry["bounds"], list(region.bounds))
                self.assertEqual(entry["relative_bounds"], list(region.relative_bounds))
                with Image.open(manifest_path.parent / entry["file"]) as crop:
                    self.assertEqual(crop.format, "PNG")
                    self.assertEqual(crop.mode, "RGB")
                    self.assertEqual(list(crop.size), entry["crop_size"])
                    np.testing.assert_array_equal(crop, region.image)

    def test_export_empty_page_and_replace_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Image.new("RGB", (10, 20))
            save_segments(segment_image(source, [Detection(0, 1, rectangle(0, 0, 5, 5))]), directory)
            manifest_path = save_segments(segment_image(source, []), directory)
            self.assertEqual(json.loads(manifest_path.read_text()), {
                "source_path": None, "source_size": [10, 20], "regions": [],
            })


class DemoTests(unittest.TestCase):
    def test_demo_uses_bgr_input_and_exports_before_annotating(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "spike_paddle_ocr.py"
        namespace = runpy.run_path(str(script))  # Importing the demo must not run a model.
        source = Image.new("RGB", (20, 10), (255, 0, 0))

        class FakeResult(dict):
            def save_to_json(self, save_path):
                self.json_saved_before_preview = not self.get("annotated", False)

            def save_to_img(self, save_path):
                self["input_img"][:] = 0  # Paddle's preview writer mutates its buffer.
                self["annotated"] = True

        class FakeDetector:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.paddlex_predictor = SimpleNamespace(post_op=SimpleNamespace(use_dilation=False))
                self.closed = False

            def predict(self, *, input, **kwargs):
                np.testing.assert_array_equal(input[0, 0], [0, 0, 255])
                self.predict_options = kwargs
                self.result = FakeResult(
                    input_img=input, dt_polys=[rectangle(2, 2, 18, 8)], dt_scores=[0.9],
                )
                return [self.result]

            def close(self):
                self.closed = True

        detector = FakeDetector()
        detector_options = []

        def make_detector(**kwargs):
            detector_options.append(kwargs)
            return detector

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            output_dir = Path(directory) / "output"
            source.save(source_path)
            with (
                patch.dict(sys.modules, {"paddleocr": SimpleNamespace(TextDetection=make_detector)}),
                patch.object(sys, "argv", [str(script), str(source_path), "--output-dir", str(output_dir)]),
                patch("builtins.print"),
            ):
                namespace["main"]()
            with Image.open(output_dir / "region-0000.png") as crop:
                self.assertEqual(crop.getpixel((4, 2)), (255, 0, 0))
            self.assertTrue(detector.closed)
            self.assertEqual(detector_options, [{"model_name": "PP-OCRv6_tiny_det", "engine": "paddle", "device": "cpu"}])
            self.assertTrue(detector.paddlex_predictor.post_op.use_dilation)
            self.assertTrue(detector.result.json_saved_before_preview)
            self.assertEqual(detector.predict_options, {"batch_size": 1, "box_thresh": 0.6, "unclip_ratio": 1.4})
            self.assertEqual(load_image(source_path).getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
