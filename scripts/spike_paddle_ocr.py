from pathlib import Path
import os.path
from paddleocr import PaddleOCR, TextDetection

# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv6_tiny_det",
#     text_recognition_model_name="PP-OCRv6_tiny_rec",
#     engine="transformers",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )
# result = ocr.predict("./ochre-handwriting-sample-01-1-of-2.jpg")
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

scripts_dir = Path(os.path.dirname(__file__)).resolve()
print(scripts_dir)

model = TextDetection(model_name="PP-OCRv6_tiny_det", engine="paddle", device="cpu")
post = model.paddlex_predictor.post_op
if hasattr(post, "use_dilation"):
    print("using dilation")
    post.use_dilation = True
output = model.predict(
    input=str(scripts_dir / "ochre-handwriting-sample-01-1-of-2.jpg"),
    batch_size=1,
    box_thresh=0.6,
    unclip_ratio=1.4
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output-det/")
    res.save_to_json(save_path="./output-det/res.json")
