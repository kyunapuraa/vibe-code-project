import asyncio
import os
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

os.environ["ML_DISABLE_WHISPER"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import bot
import db
import embeddings


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


class FakeMessage:
    def __init__(self, text, user_id=1):
        self.text = text
        self.from_user = SimpleNamespace(id=user_id)
        self.answers = []

    async def answer(self, text, **kwargs):
        self.answers.append(text)
        return SimpleNamespace(delete=self._noop)

    async def _noop(self):
        return None


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "_model", FakeModel())
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    asyncio.run(db.init_db())


MATRIX = "Матрицей называется прямоугольная таблица чисел. У квадратной матрицы есть определитель."
DERIV = "Производная функции показывает скорость изменения. Есть таблица производных."


def run(coro):
    return asyncio.run(coro)


def test_note_saves_chunks_bound_to_user():
    msg = FakeMessage("/note " + MATRIX, user_id=1)
    run(bot.save_note(msg))
    assert "Сохранил" in msg.answers[-1]
    rows = run(db.get_user_chunks(1))
    assert len(rows) == 1
    assert rows[0]["text"] == MATRIX
    assert len(rows[0]["embedding"]) == embeddings.EMBED_DIM * 4
    assert run(db.get_user_chunks(2)) == []


def test_ask_finds_nearest_chunk():
    run(bot.save_note(FakeMessage("/note " + MATRIX)))
    run(bot.save_note(FakeMessage("/note " + DERIV)))
    msg = FakeMessage("/ask где определитель матрицы")
    run(bot.ask(msg))
    answer = msg.answers[-1]
    assert answer.startswith("Нашёл в")
    assert MATRIX in answer.split("\n\n")[1]


def test_ask_searches_only_own_materials():
    run(bot.save_note(FakeMessage("/note " + MATRIX, user_id=1)))
    msg = FakeMessage("/ask матрицы", user_id=2)
    run(bot.ask(msg))
    assert "нет сохранённых материалов" in msg.answers[-1]


def test_ask_without_query():
    msg = FakeMessage("/ask")
    run(bot.ask(msg))
    assert "Напишите вопрос" in msg.answers[-1]


def test_docs_and_forget():
    run(bot.save_note(FakeMessage("/note " + MATRIX)))
    docs = run(db.get_documents(1))
    assert len(docs) == 1 and docs[0]["chunks"] == 1
    msg = FakeMessage(f"/forget {docs[0]['id']}")
    run(bot.forget_document(msg))
    assert "удалён" in msg.answers[-1]
    assert run(db.get_user_chunks(1)) == []
    assert run(db.get_documents(1)) == []


def test_tasks_table_still_works():
    task_id = run(db.add_task(1, "Сделай отчёт", "Сделать отчёт", "2030-01-01T18:00:00+03:00", "normal"))
    assert run(db.get_task(task_id, 1))["title"] == "Сделать отчёт"
