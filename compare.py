import numpy as np
import easyocr_hybrid_onnx.easyocr as easyocr_hybrid_onnx
import easyocr_only_onnx.easyocr as easyocr_only_onnx

leng = "en"
images = ["dummyImg.jpg"]

reader1 = easyocr_hybrid_onnx.Reader([leng], gpu=True, customEasyOcrModulePath="easyocr_hybrid_onnx")
reader2 = easyocr_only_onnx.Reader([leng], gpu=True)

for img in images:

    r1 = reader1.readtext(img, onnx=False)
    r2 = reader2.readtext(img)

    #Validates same number of character groups
    if len(r1) == len(r2):
        for g1, g2 in zip(r1, r2):
            #Compare character cut zone (detector model)
            if np.array_equal(g1[0],g2[0]):
                print("Same character cut zones")
            else:
                print("Different character cut zones")
            #Compare character
            print("Standard EasyOCR:", g1[1])
            print("Only ONNX EasyOCR:",g2[1])
            if g1[1] == g2[1]:
                print("Same characters")
            else:
                print("Different characters")
            #Compares accuracy
            print("Standard EasyOCR:",g1[2])
            print("Only ONNX EasyOCR:",g2[2])
            diff = abs(g1[2] - g2[2])
            print("The difference in accuracy is:", diff)
    else:
        print("Different character cut zones count")