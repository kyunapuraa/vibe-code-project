import json
import os
import re
import zlib

import numpy as np
import pytest

import embeddings as emb

REAL = os.getenv("RUN_MODEL_TESTS") == "1"


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), emb.EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            text = re.sub(r"^(query|passage): ", "", text.lower())
            for word in re.findall(r"\w+", text):
                out[row, zlib.crc32(word[:5].encode()) % emb.EMBED_DIM] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    if REAL:
        yield None
        return
    model = FakeModel()
    monkeypatch.setattr(emb, "_model", model)
    monkeypatch.setattr(emb, "MODEL_NAME", "intfloat/multilingual-e5-small")
    yield model


CHUNKS_TEXT = [
    "Матрицы: сложение, умножение матриц и определитель квадратной матрицы.",
    "Производная функции, правила дифференцирования и таблица производных.",
    "Интегралы: неопределённый интеграл, замена переменной, интегрирование по частям.",
    "Теория вероятностей: случайные события, формула Байеса.",
]


def make_db():
    return [{"id": i + 1, "text": t, "embedding": emb.embed(t)} for i, t in enumerate(CHUNKS_TEXT)]


def test_embed_returns_list_of_floats_with_right_dim():
    vector = emb.embed("Определитель матрицы")
    assert isinstance(vector, list)
    assert len(vector) == emb.EMBED_DIM
    assert all(isinstance(x, float) for x in vector)


def test_embed_is_normalized():
    vector = np.array(emb.embed("Определитель матрицы"))
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-4


def test_embed_is_json_serializable():
    json.dumps(emb.embed("Матрицы"))


def test_embed_empty_text_gives_zero_vector():
    assert emb.embed("") == [0.0] * emb.EMBED_DIM
    assert emb.embed(None) == [0.0] * emb.EMBED_DIM


@pytest.mark.skipif(REAL, reason="проверка префиксов только на заглушке")
def test_e5_prefixes(fake_model):
    emb.embed("текст")
    emb.embed_query("вопрос")
    assert fake_model.calls[0] == ["passage: текст"]
    assert fake_model.calls[1] == ["query: вопрос"]


def test_embed_chunks_from_strings():
    records = emb.embed_chunks(["Первый кусок", "Второй кусок"])
    assert [r["text"] for r in records] == ["Первый кусок", "Второй кусок"]
    assert all(len(r["embedding"]) == emb.EMBED_DIM for r in records)


def test_embed_chunks_keeps_ml1_metadata():
    chunks = [{"text": "Матрицы", "title": "Линал", "date": "2026-10-01", "position": 0}]
    record = emb.embed_chunks(chunks)[0]
    assert record["title"] == "Линал"
    assert record["date"] == "2026-10-01"
    assert record["position"] == 0
    assert len(record["embedding"]) == emb.EMBED_DIM


def test_embed_chunks_empty():
    assert emb.embed_chunks([]) == []
    assert emb.embed_chunks(None) == []


@pytest.mark.parametrize(
    "query, expected_id",
    [
        ("где я изучал матрицы", 1),
        ("определитель матрицы", 1),
        ("производная функции", 2),
        ("интегрирование по частям", 3),
        ("формула Байеса", 4),
    ],
)
def test_search_top_finds_right_chunk(query, expected_id):
    top = emb.search_top(query, make_db(), k=1)
    assert top[0]["id"] == expected_id


def test_search_returns_text():
    assert emb.search("где я изучал матрицы", make_db()) == CHUNKS_TEXT[0]


def test_search_top_sorted_and_limited():
    top = emb.search_top("матрицы", make_db(), k=3)
    assert len(top) == 3
    scores = [t["score"] for t in top]
    assert scores == sorted(scores, reverse=True)


def test_search_accepts_tuples_json_and_bytes():
    db = make_db()
    rows = [
        (db[0]["id"], db[0]["text"], json.dumps(db[0]["embedding"])),
        (db[1]["id"], db[1]["text"], np.array(db[1]["embedding"], dtype=np.float32).tobytes()),
        (db[2]["id"], db[2]["text"], np.array(db[2]["embedding"])),
    ]
    assert emb.search("определитель матрицы", rows) == CHUNKS_TEXT[0]


def test_search_skips_broken_embeddings():
    db = make_db()
    db.append({"id": 99, "text": "битый", "embedding": [1.0, 2.0]})
    db.append({"id": 100, "text": "пустой", "embedding": None})
    db.append({"id": 101, "text": "нули", "embedding": [0.0] * emb.EMBED_DIM})
    top = emb.search_top("матрицы", db, k=10)
    assert {t["id"] for t in top} == {1, 2, 3, 4}


def test_search_empty_inputs():
    assert emb.search("", make_db()) == ""
    assert emb.search("матрицы", []) == ""
    assert emb.search_top("матрицы", None) == []


def test_search_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(emb, "_encode", boom)
    db = [{"id": 1, "text": "x", "embedding": [0.1] * emb.EMBED_DIM}]
    assert emb.search("матрицы", db) == ""
    assert emb.embed("матрицы") == [0.0] * emb.EMBED_DIM