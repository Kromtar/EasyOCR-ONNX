import easyocr
import time

start_time = time.time()
reader = easyocr.Reader(['ja'], gpu=True)
result = reader.readtext('easyocr_only_onnx/dummyImg.jpg')
print(result)
print("--- %s seconds ---" % (time.time() - start_time))