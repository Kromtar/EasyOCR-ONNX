import easyocr

reader = easyocr.Reader(["en"], gpu=True)
r = reader.readtext('dummyImg.jpg', onnx=True)

print(r)