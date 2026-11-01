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


model = TextDetection(model_name="PP-OCRv6_tiny_det", engine="transformers")
output = model.predict(input="./ochre-handwriting-sample-01-1-of-2.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output-det/")
    res.save_to_json(save_path="./output-det/res.json")
