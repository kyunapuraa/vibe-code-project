import re
import zlib

import numpy as np
import pytest

import embeddings
import knowledge


class FakeModel:
    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), embeddings.EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            text = re.sub(r"^(query|passage): ", "", text.lower())
            for word in re.findall(r"\w+", text):
                out[row, zlib.crc32(word[:5].encode()) % embeddings.EMBED_DIM] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", FakeModel())


LECTURE = (
    "Матрицы. Матрицей называется прямоугольная таблица чисел. "
    "Матрицы можно складывать и умножать, у квадратной матрицы есть определитель.\n\n"
    "Производная. Производная функции показывает скорость её изменения. "
    "Таблица производных и правила дифференцирования помогают считать производные.\n\n"
    "Интегралы. Неопределённый интеграл обратен производной. "
    "Интегрирование по частям и замена переменной — основные приёмы."
)


def test_chunk_text_splits_and_keeps_source():
    chunks = knowledge.chunk_text(LECTURE * 3, "Матан")
    assert len(chunks) > 1
    assert all(c["source"] == "Матан" and c["page"] == 1 for c in chunks)


def test_chunk_text_short_note_is_kept():
    chunks = knowledge.chunk_text("Купить мел", "Заметка")
    assert chunks == [{"text": "Купить мел", "source": "Заметка", "page": 1}]


def test_chunk_text_empty():
    assert knowledge.chunk_text("   ") == []


def test_index_file_txt(tmp_path):
    path = tmp_path / "tmp123.txt"
    path.write_text(LECTURE, encoding="utf-8")
    records = knowledge.index_file(str(path), "Лекция 1.txt")
    assert records
    assert all(r["source"] == "Лекция 1.txt" for r in records)
    assert all(len(r["embedding"]) == embeddings.EMBED_DIM for r in records)


def test_index_file_unsupported(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        knowledge.index_file(str(path))


def test_find_returns_meta():
    records = knowledge.index_text(LECTURE, "Матан")
    rows = [
        {"id": i, "document_id": 7, "page": 1, "source": r["source"], "text": r["text"], "embedding": r["embedding"]}
        for i, r in enumerate(records)
    ]
    top = knowledge.find("определитель матрицы", rows, k=1)
    assert "определитель" in top[0]["text"]
    assert top[0]["source"] == "Матан"
    assert top[0]["document_id"] == 7
