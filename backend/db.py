from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from array import array

import aiosqlite


DB_PATH = Path(os.getenv("DB_PATH", "tasks.db"))
MOSCOW_TZ = ZoneInfo("Europe/Moscow")

CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    title TEXT NOT NULL,
    deadline TEXT,
    priority TEXT NOT NULL DEFAULT 'обычный',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'done')),
    created_at TEXT NOT NULL
)
"""

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'file',
    created_at TEXT NOT NULL
)
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
)
"""


async def init_db() -> None:
    """Создаёт базу и таблицу задач, если их ещё нет."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute(CREATE_TASKS_TABLE)
        await db.execute(CREATE_DOCUMENTS_TABLE)
        await db.execute(CREATE_CHUNKS_TABLE)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks (user_id)"
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_user_status
            ON tasks (user_id, status)
            """
        )
        await db.commit()


async def add_task(
    user_id: int,
    original_text: str,
    title: str,
    deadline: str | None,
    priority: str,
) -> int:
    """Сохраняет задачу и возвращает её ID."""
    if not original_text.strip():
        raise ValueError("Исходное сообщение не должно быть пустым.")
    if not title.strip():
        raise ValueError("Название задачи не должно быть пустым.")

    created_at = datetime.now(MOSCOW_TZ).isoformat(timespec="seconds")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO tasks (
                user_id,
                original_text,
                title,
                deadline,
                priority,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                user_id,
                original_text.strip(),
                title.strip(),
                deadline,
                priority.strip() or "обычный",
                created_at,
            ),
        )
        await db.commit()

        if cursor.lastrowid is None:
            raise RuntimeError("Не удалось получить ID созданной задачи.")

        return cursor.lastrowid


async def get_active_tasks(
    user_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Возвращает активные задачи.

    Если user_id указан — только задачи этого пользователя.
    Если не указан — все активные задачи, например, для восстановления
    напоминаний после перезапуска бота.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if user_id is None:
            cursor = await db.execute(
                """
                SELECT id, user_id, original_text, title, deadline,
                       priority, status, created_at
                FROM tasks
                WHERE status = 'active'
                ORDER BY deadline IS NULL, deadline, id
                """
            )
        else:
            cursor = await db.execute(
                """
                SELECT id, user_id, original_text, title, deadline,
                       priority, status, created_at
                FROM tasks
                WHERE status = 'active' AND user_id = ?
                ORDER BY deadline IS NULL, deadline, id
                """,
                (user_id,),
            )

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_task(task_id: int, user_id: int) -> dict[str, Any] | None:
    """Возвращает задачу пользователя по ID или None, если её нет."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, user_id, original_text, title, deadline,
                   priority, status, created_at
            FROM tasks
            WHERE id = ? AND user_id = ?
            """,
            (task_id, user_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None


async def mark_done(task_id: int, user_id: int) -> bool:
    """Отмечает задачу выполненной. True, если активная задача найдена."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            UPDATE tasks
            SET status = 'done'
            WHERE id = ? AND user_id = ? AND status = 'active'
            """,
            (task_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def mark_task_done(task_id: int, user_id: int) -> bool:
    """То же, что mark_done; оставлено для удобства импорта из bot.py."""
    return await mark_done(task_id, user_id)


def _pack_embedding(vector: list[float]) -> bytes:
    """Упаковывает эмбеддинг в байты float32 для колонки BLOB."""
    return array("f", vector).tobytes()


async def add_document(
    user_id: int,
    source: str,
    kind: str,
    records: list[dict[str, Any]],
) -> int:
    """Сохраняет документ и его куски с эмбеддингами. Возвращает ID документа."""
    records = [r for r in records if str(r.get("text") or "").strip() and r.get("embedding")]
    if not records:
        raise ValueError("Нет кусков текста для сохранения.")

    created_at = datetime.now(MOSCOW_TZ).isoformat(timespec="seconds")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO documents (user_id, source, kind, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, source.strip() or "Без названия", kind, created_at),
        )
        document_id = cursor.lastrowid
        if document_id is None:
            raise RuntimeError("Не удалось получить ID документа.")

        await db.executemany(
            """
            INSERT INTO chunks (
                document_id, user_id, position, page, text, embedding, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    user_id,
                    position,
                    record.get("page"),
                    str(record["text"]).strip(),
                    _pack_embedding(record["embedding"]),
                    created_at,
                )
                for position, record in enumerate(records)
            ],
        )
        await db.commit()
        return document_id


async def get_user_chunks(user_id: int) -> list[dict[str, Any]]:
    """Возвращает все куски пользователя с эмбеддингами для поиска."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT c.id, c.document_id, c.page, c.text, c.embedding,
                   d.source, d.created_at
            FROM chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE c.user_id = ?
            ORDER BY c.id
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_documents(user_id: int) -> list[dict[str, Any]]:
    """Возвращает документы пользователя с количеством кусков."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT d.id, d.source, d.kind, d.created_at, COUNT(c.id) AS chunks
            FROM documents AS d
            LEFT JOIN chunks AS c ON c.document_id = d.id
            WHERE d.user_id = ?
            GROUP BY d.id
            ORDER BY d.id
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_document(document_id: int, user_id: int) -> bool:
    """Удаляет документ пользователя вместе с его кусками."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM chunks WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        )
        cursor = await db.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0
