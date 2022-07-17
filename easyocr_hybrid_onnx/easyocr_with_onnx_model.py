import easyocr

reader = easyocr.Reader(["ja"], gpu=True)
r = reader.readtext('easyocr_hybrid_onnx/dummyImg.jpg', onnx=True)

print(r)