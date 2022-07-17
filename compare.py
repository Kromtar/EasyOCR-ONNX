import easyocr_hybrid_onnx.easyocr as easyocr_hybrid_onnx
import easyocr_only_onnx.easyocr as easyocr_only_onnx

reader1 = easyocr_hybrid_onnx.Reader(["ja"], gpu=True, customEasyOcrModulePath="easyocr_hybrid_onnx")
r1 = reader1.readtext('img1.png', onnx=False)

reader2 = easyocr_only_onnx.Reader(["ja"], gpu=True)
r2 = reader2.readtext('img1.png')

print(r1)
print(r2)