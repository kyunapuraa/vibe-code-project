from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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


async def init_db() -> None:
    """Создаёт базу и таблицу задач, если их ещё нет."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute(CREATE_TASKS_TABLE)
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
