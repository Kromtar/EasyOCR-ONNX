import easyocr
reader = easyocr.Reader(['ja'], gpu=True)
result = reader.readtext('img.png')
print(result)