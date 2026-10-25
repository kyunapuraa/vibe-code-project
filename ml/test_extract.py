import os
import pytest
from extract import extract


def test_txt_format(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Матрицы. Матрицей называется таблица чисел. " * 20, encoding="utf-8")
    items = extract(str(p))
    assert len(items) >= 1
    for it in items:
        assert set(it) == {"text", "source", "page"}
        assert it["source"] == "a.txt"
        assert it["page"] == 1


def test_chunk_length(tmp_path):
    text = ". ".join("Предложение номер %d про матрицы и определители" % i for i in range(100))
    p = tmp_path / "b.txt"
    p.write_text(text, encoding="utf-8")
    items = extract(str(p))
    assert len(items) > 1
    assert all(len(it["text"]) <= 800 for it in items)


def test_unknown_format(tmp_path):
    p = tmp_path / "c.xyz"
    p.write_text("текст", encoding="utf-8")
    with pytest.raises(ValueError):
        extract(str(p))


@pytest.mark.skipif(not os.path.exists("data/page1.png"), reason="нет тестовой картинки")
def test_image_ocr():
    items = extract("data/page1.png")
    assert len(items) >= 1
    assert "множеств" in " ".join(it["text"] for it in items)
