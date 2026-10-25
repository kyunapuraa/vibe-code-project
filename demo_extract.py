import sys
from extract import read_pages, clean, split_text, merge_small, extract


def short(text, n=150):
    return text[:n].replace("\n", " | ")


def demo(path):
    print("=" * 70)
    print("ФАЙЛ:", path)
    pages = read_pages(path)
    print("1) Прочитано страниц:", len(pages))
    page, raw = pages[0]
    print("   Сырой текст стр.", page, "(%d симв.):" % len(raw))
    print("   ", short(raw))
    cleaned = clean(raw)
    print("2) После очистки (%d симв.):" % len(cleaned))
    print("   ", short(cleaned))
    parts = split_text(cleaned)
    print("3) Нарезано по предложениям на кусков:", len(parts), "-> после склейки мелких:", len(merge_small(parts)))
    items = extract(path)
    lens = [len(i["text"]) for i in items]
    print("4) Итог extract(): кусков", len(items), "| длины мин", min(lens), "макс", max(lens))
    for it in items[:3]:
        print("   -", it["source"], "стр.", it["page"], "(%d симв.)" % len(it["text"]))
        print("    ", short(it["text"], 200))


for p in sys.argv[1:]:
    demo(p)
