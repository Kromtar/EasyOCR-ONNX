import numpy as np
import easyocr_hybrid_onnx.easyocr as easyocr_hybrid_onnx
import easyocr_only_onnx.easyocr as easyocr_only_onnx

reader1 = easyocr_hybrid_onnx.Reader(["ja"], gpu=True, customEasyOcrModulePath="easyocr_hybrid_onnx")
reader2 = easyocr_only_onnx.Reader(["ja"], gpu=True)

images = ["img9.jpg"]

for img in images:

    r1 = reader1.readtext(img, onnx=False)
    r2 = reader2.readtext(img)

    #Valida mismo numero de conjunto de caracteres
    if len(r1) == len(r2):
        for g1, g2 in zip(r1, r2):
            #Compara zona de caracteres
            if np.array_equal(g1[0],g2[0]):
                print("Mismas zonas de caracteres")
            else:
                print("Distintas zonas de caracters")
            #Compara caracteres
            if g1[1] == g2[1]:
                print("Mismo conjunto de caractres")
            else:
                print("El conjunto de caracteres tiene diferencia")
            print(g1[1])
            print(g2[1])
            #Compara presiciones
            diff = abs(g1[2] - g2[2])
            print("La diferencia de presicion es:", diff)
    else:
        print("No se ha detectado el mismo numero de conjuntos de caracteres")