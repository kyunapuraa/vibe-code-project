import sys
import time
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

path = sys.argv[1]
page = int(sys.argv[2]) if len(sys.argv) > 2 else 1

reader = PdfReader(path)
print("Страниц в PDF:", len(reader.pages))

direct = reader.pages[page - 1].extract_text() or ""
print("\n1) Обычное чтение PDF (pypdf): %d симв." % len(direct))
print(direct[:300].replace("\n", " | "))

img = convert_from_path(path, dpi=200, first_page=page, last_page=page)[0]
img.save("data/ocr_page.png")
print("\nСтраница превращена в картинку: data/ocr_page.png", img.size)

start = time.time()
text = pytesseract.image_to_string(img, lang="rus+eng")
data = pytesseract.image_to_data(img, lang="rus+eng", output_type=pytesseract.Output.DICT)
confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
print("\n2) Tesseract (распознавание с картинки): %d симв. за %.1f с" % (len(text), time.time() - start))
print("   слов: %d, средняя уверенность: %.0f%%" % (len(confs), sum(confs) / max(len(confs), 1)))
print(text[:600].replace("\n", " | "))
