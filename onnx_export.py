#leng_list = ["abq", "ady", "af", "ang", "ar", "as", "ava", "az", "be", "bg", "bh", "bho", "bn", "bs", "ch_sim", "ch_tra", "che", "cs", "cy", "da", "dar", "de", "en", "es", "et", "fa", "fr", "ga", "gom", "hi", "hr", "hu", "id", "inh", "is", "it", "ja", "kbd", "kn", "ko", "ku", "la", "lbe", "lez", "it", "lv", "mah", "mai", "mi", "mn", "mr", "ms", "mt", "ne", "new", "nl", "no", "oc", "pi", "pl", "pt", "ro", "ru", "rs_cyrillic", "rs_latin", "sck", "sk", "sl", "sq", "sv", "sw", "ta", "tab", "te", "th", "tjk", "tl", "tr", "ug", "uk", "ur", "uz", "vi"]
import os
import easyocr

print("Select the language to which you want to export the ONNX models, use the acronym indicated in this list: https://www.jaided.ai/easyocr/ (Example: 'en' for English)")
lang = input("Language:")

reader = easyocr.Reader([lang], gpu=True)
reader.readtext('dummyImg.jpg')

os.rename("detection_model.onnx","onnx_models/detection_model.onnx")
recModelTempName = "recognition_model.onnx"
recModelDest= "onnx_models/" + lang + "_" + recModelTempName
os.rename(recModelTempName,recModelDest)

print("Ready")