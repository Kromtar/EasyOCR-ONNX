import easyocr

reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext('easyocr_only_onnx/dummyImg.jpg')
print(result)