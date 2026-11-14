"""Detect text and export straightened crops without running recognition.

    uv run --script scripts/spike_paddle_ocr.py [IMAGE] --output-dir output-det
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ochre import detections_from_paddle, load_image, save_segments, segment_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("ochre-handwriting-sample-01-1-of-2.jpg"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output-det"))
    args = parser.parse_args()
    source = load_image(args.image)

    # Keep Paddle out of the package API and the --help path.
    from paddleocr import TextDetection

    model = TextDetection(model_name="PP-OCRv6_tiny_det", engine="paddle", device="cpu")
    try:
        post = model.paddlex_predictor.post_op
        if hasattr(post, "use_dilation"):
            post.use_dilation = True
        # Paddle expects BGR arrays. Give it a separate buffer so previews cannot
        # draw annotations onto the original RGB source used for crops.
        output = model.predict(
            input=cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR),
            batch_size=1,
            box_thresh=0.6,
            unclip_ratio=1.4,
        )
        for result in output:
            page = segment_image(source, detections_from_paddle(result))
            manifest = save_segments(page, args.output_dir, source_path=args.image)
            result["input_path"] = str(args.image)
            result.save_to_json(save_path=str(args.output_dir / "res.json"))
            result.save_to_img(save_path=str(args.output_dir))
            print(f"{len(page.regions)} regions saved to {manifest}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
