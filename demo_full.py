import sys
import time
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

path = sys.argv[1]
page = int(sys.argv[2]) if len(sys.argv) > 2 else 1

direct = PdfReader(path).pages[page - 1].extract_text() or ""

img = convert_from_path(path, dpi=200, first_page=page, last_page=page)[0]
img.save("data/ocr_page.png")

start = time.time()
text = pytesseract.image_to_string(img, lang="rus+eng")
data = pytesseract.image_to_data(img, lang="rus+eng", output_type=pytesseract.Output.DICT)
confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
elapsed = time.time() - start

print("=" * 70)
print("1) ОБЫЧНОЕ ЧТЕНИЕ PDF (pypdf): %d симв." % len(direct))
print("=" * 70)
print(direct)
print()
print("=" * 70)
print("2) TESSERACT с картинки: %d симв. за %.1f с, слов %d, уверенность %.0f%%" % (len(text), elapsed, len(confs), sum(confs) / max(len(confs), 1)))
print("=" * 70)
print(text)
