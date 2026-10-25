import os
import re
from pypdf import PdfReader
from docx import Document

MAX_LEN = 700
MIN_LEN = 100
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def ocr_image(path):
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path), lang="rus+eng")


def ocr_pdf(path):
    import pytesseract
    from pdf2image import convert_from_path
    pages = convert_from_path(path, dpi=200)
    return [(i + 1, pytesseract.image_to_string(img, lang="rus+eng")) for i, img in enumerate(pages)]


def read_pages(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        if sum(len(t.strip()) for _, t in pages) < 50:
            return ocr_pdf(path)
        return pages
    if ext in IMAGE_EXT:
        return [(1, ocr_image(path))]
    if ext == ".docx":
        doc = Document(path)
        return [(1, "\n\n".join(p.text for p in doc.paragraphs))]
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8") as f:
            return [(1, f.read())]
    raise ValueError("Неподдерживаемый формат: " + ext)


def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hard_cut(text, max_len):
    parts = []
    while len(text) > max_len:
        cut = text.rfind(" ", 0, max_len)
        if cut <= 0:
            cut = max_len
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def split_text(text, max_len=MAX_LEN):
    units = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_len:
                units.extend(hard_cut(sent, max_len))
            else:
                units.append(sent)
    chunks = []
    current = ""
    for u in units:
        if current and len(current) + len(u) + 1 > max_len:
            chunks.append(current)
            current = u
        else:
            current = (current + " " + u).strip()
    if current:
        chunks.append(current)
    return chunks


def merge_small(chunks, min_len=MIN_LEN):
    merged = []
    for c in chunks:
        if merged and len(c) < min_len:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged


def extract(path, with_vision=False):
    source = os.path.basename(path)
    result = []
    for page, raw in read_pages(path):
        chunks = merge_small(split_text(clean(raw)))
        for chunk in chunks:
            if len(chunk) > 40:
                result.append({"text": chunk, "source": source, "page": page})
    if with_vision and os.path.splitext(path)[1].lower() in IMAGE_EXT:
        from vision import describe
        labels = describe(path)
        if labels:
            result.append({"text": "На изображении: " + ", ".join(labels), "source": source, "page": 1})
    return result


if __name__ == "__main__":
    import sys
    items = extract(sys.argv[1])
    print("кусков:", len(items))
    lens = [len(it["text"]) for it in items]
    if lens:
        print("длины: мин", min(lens), "макс", max(lens), "средняя", sum(lens) // len(lens))
    for it in items[:10]:
        print(it["page"], len(it["text"]), it["text"][:80].replace("\n", " "))
