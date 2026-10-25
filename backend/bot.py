"""Telegram bot for collecting colleague requests and reminding about deadlines."""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from dotenv import load_dotenv

import ml
from db import create_task, get_active_tasks, init_db, mark_done
from keyboards import confirmation_keyboard


MOSCOW_TZ = ZoneInfo("Europe/Moscow")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

dp = Dispatcher()
scheduler = AsyncIOScheduler(timezone=MOSCOW_TZ)

# Неподтверждённые черновики хранятся в памяти.
# Сохранённые задачи при запуске загружаются из SQLite.
pending_tasks: dict[int, dict[str, Any]] = {}


def _parse_deadline(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MOSCOW_TZ)

    return parsed.astimezone(MOSCOW_TZ)


def _format_deadline(value: datetime) -> str:
    return value.strftime("%d.%m.%Y в %H:%M (МСК)")


async def _parse_and_show(
    message: Message,
    user_id: int,
    draft: dict[str, Any],
) -> None:
    try:
        parsed = await asyncio.to_thread(
            ml.parse_task,
            draft["parse_text"],
            datetime.now(MOSCOW_TZ),
        )
    except Exception:
        logger.exception("ML task parsing failed for user_id=%s", user_id)
        pending_tasks[user_id] = draft
        await message.answer(
            "Не получилось разобрать сообщение. Попробуйте ещё раз "
            "или пришлите дедлайн отдельным сообщением, например: "
            "28.10.2026 18:00."
        )
        return

    if not isinstance(parsed, dict):
        logger.error(
            "ml.parse_task returned %s instead of dict",
            type(parsed).__name__,
        )
        pending_tasks[user_id] = draft
        await message.answer(
            "Не получилось разобрать сообщение. "
            "Пришлите его ещё раз, пожалуйста."
        )
        return

    title = str(parsed.get("title") or "Задача без названия").strip()
    priority = str(parsed.get("priority") or "не указана").strip()
    deadline = _parse_deadline(parsed.get("deadline"))

    if deadline is None or deadline <= datetime.now(MOSCOW_TZ):
        draft["title"] = title
        draft["priority"] = priority
        pending_tasks[user_id] = draft
        await message.answer(
            "Не удалось определить будущий дедлайн. Напишите его "
            "отдельным сообщением, например: 28.10.2026 18:00."
        )
        return

    pending_tasks[user_id] = {
        "original_text": draft["original_text"],
        "title": title,
        "deadline": deadline.isoformat(),
        "priority": priority,
    }

    await message.answer(
        "Проверьте задачу:\n"
        f"Название: {title}\n"
        f"Дедлайн: {_format_deadline(deadline)}\n"
        f"Приоритет: {priority}\n\n"
        "Сохранить?",
        reply_markup=confirmation_keyboard(),
    )


async def _handle_incoming_text(message: Message, text: str) -> None:
    user_id = message.from_user.id
    waiting = pending_tasks.get(user_id)

    if waiting and "deadline" not in waiting:
        waiting["parse_text"] += f"\nДедлайн: {text.strip()}"
        await _parse_and_show(message, user_id, waiting)
        return

    draft = {
        "original_text": text.strip(),
        "parse_text": text.strip(),
    }
    await _parse_and_show(message, user_id, draft)


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Перешлите мне сообщение коллеги или отправьте его текстом "
        "или голосовым. Я предложу название, дедлайн и приоритет — "
        "проверьте их и подтвердите сохранение.\n\n"
        "Команды: /tasks — активные задачи, "
        "/done ID — отметить задачу выполненной."
    )


@dp.message(Command("tasks"))
async def tasks_command(message: Message) -> None:
    rows = await get_active_tasks(message.from_user.id)

    if not rows:
        await message.answer("Активных задач пока нет.")
        return

    lines = ["Активные задачи:"]

    for row in rows:
        deadline = _parse_deadline(row["deadline"])
        deadline_text = (
            _format_deadline(deadline) if deadline else row["deadline"]
        )
        lines.append(
            f"#{row['id']} · {row['title']} · до {deadline_text}"
        )

    lines.append("\nЧтобы завершить задачу: /done ID")
    await message.answer("\n".join(lines))


@dp.message(Command("done"))
async def done_command(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer(
            "Укажите номер задачи: /done ID. "
            "Номера можно посмотреть командой /tasks."
        )
        return

    task_id = int(parts[1])
    updated = await mark_done(task_id, message.from_user.id)

    if updated:
        job_id = f"task-{task_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
        await message.answer(f"Задача #{task_id} отмечена выполненной.")
    else:
        await message.answer("Активная задача с таким номером не найдена.")


@dp.message(F.voice)
async def voice_message(message: Message, bot: Bot) -> None:
    voice = message.voice
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="telegram_voice_",
            suffix=".ogg",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        await bot.download(voice.file_id, destination=temp_path)
        transcript = await asyncio.to_thread(ml.transcribe, temp_path)

        if not isinstance(transcript, str) or not transcript.strip():
            await message.answer(
                "Не удалось распознать голосовое. "
                "Пришлите его ещё раз или отправьте текстом."
            )
            return

        await _handle_incoming_text(message, transcript.strip())

    except Exception:
        logger.exception(
            "Voice processing failed for user_id=%s",
            message.from_user.id,
        )
        await message.answer(
            "Не удалось обработать голосовое. "
            "Попробуйте ещё раз или отправьте сообщение текстом."
        )

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove temporary voice file: %s",
                    temp_path,
                )


@dp.message(F.text & ~F.text.startswith("/"))
async def text_message(message: Message) -> None:
    text = (message.text or "").strip()
    if text:
        await _handle_incoming_text(message, text)


@dp.callback_query(F.data == "confirm_task")
async def confirm_task(callback: CallbackQuery, bot: Bot) -> None:
    user_id = callback.from_user.id
    draft = pending_tasks.get(user_id)

    if not draft or "deadline" not in draft:
        await callback.answer(
            "Черновик задачи устарел. Отправьте сообщение ещё раз.",
            show_alert=True,
        )
        return

    task_id = await create_task(
        user_id=user_id,
        original_text=draft["original_text"],
        title=draft["title"],
        deadline=draft["deadline"],
        priority=draft["priority"],
    )
    pending_tasks.pop(user_id, None)

    _schedule_reminder(
        bot,
        task_id,
        user_id,
        draft["deadline"],
        draft["original_text"],
        draft["title"],
    )

    await callback.answer("Задача сохранена")
    if callback.message:
        await callback.message.edit_text(
            f"Задача #{task_id} сохранена. Напомню о ней в дедлайн."
        )


@dp.callback_query(F.data == "cancel_task")
async def cancel_task(callback: CallbackQuery) -> None:
    pending_tasks.pop(callback.from_user.id, None)
    await callback.answer("Не сохраняю")

    if callback.message:
        await callback.message.edit_text(
            "Задача не сохранена. Можете прислать другое сообщение."
        )


async def _send_reminder(
    bot: Bot,
    user_id: int,
    task_id: int,
    title: str,
    original_text: str,
) -> None:
    try:
        await bot.send_message(
            user_id,
            f"⏰ Дедлайн задачи #{task_id}: {title}\n\n"
            f"Исходное сообщение коллеги:\n{original_text}",
        )
    except Exception:
        logger.exception(
            "Could not send reminder for task_id=%s",
            task_id,
        )


def _schedule_reminder(
    bot: Bot,
    task_id: int,
    user_id: int,
    deadline_value: str,
    original_text: str,
    title: str,
) -> None:
    deadline = _parse_deadline(deadline_value)

    if deadline is None:
        logger.error(
            "Invalid deadline for task_id=%s: %r",
            task_id,
            deadline_value,
        )
        return

    run_at = max(deadline, datetime.now(MOSCOW_TZ))
    scheduler.add_job(
        _send_reminder,
        trigger=DateTrigger(run_date=run_at, timezone=MOSCOW_TZ),
        args=[bot, user_id, task_id, title, original_text],
        id=f"task-{task_id}",
        replace_existing=True,
        misfire_grace_time=None,
    )


async def main() -> None:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")

    if not token:
        raise RuntimeError("BOT_TOKEN не найден. Добавьте его в файл .env.")

    await init_db()
    bot = Bot(token=token)

    for task in await get_active_tasks():
        _schedule_reminder(
            bot,
            task["id"],
            task["user_id"],
            task["deadline"],
            task["original_text"],
            task["title"],
        )

    scheduler.start()

    try:
        logger.info("Bot started; timezone=%s", MOSCOW_TZ)
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
