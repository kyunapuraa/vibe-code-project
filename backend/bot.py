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
from db import add_task, get_active_tasks, init_db, mark_done
from keyboards import confirmation_keyboard


MOSCOW_TZ = ZoneInfo("Europe/Moscow")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

dp = Dispatcher()
scheduler = AsyncIOScheduler(timezone=MOSCOW_TZ)

# Черновики ждут подтверждения пользователя в памяти.
pending_tasks: dict[int, dict[str, Any]] = {}


def parse_deadline(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if result.tzinfo is None:
        result = result.replace(tzinfo=MOSCOW_TZ)

    return result.astimezone(MOSCOW_TZ)


def format_deadline(value: datetime) -> str:
    return value.strftime("%d.%m.%Y в %H:%M (МСК)")


async def parse_and_show(
    message: Message,
    user_id: int,
    draft: dict[str, Any],
) -> None:
    try:
        result = await asyncio.to_thread(
            ml.parse_task,
            draft["parse_text"],
            datetime.now(MOSCOW_TZ),
        )
    except Exception:
        logger.exception("Не удалось разобрать задачу через ML")
        draft["awaiting_deadline"] = True
        pending_tasks[user_id] = draft
        await message.answer(
            "Не получилось разобрать сообщение. Попробуйте ещё раз "
            "или пришлите дедлайн отдельным сообщением, например: "
            "28.10.2026 18:00."
        )
        return

    if not isinstance(result, dict):
        draft["awaiting_deadline"] = True
        pending_tasks[user_id] = draft
        await message.answer("Не получилось разобрать сообщение. Попробуйте ещё раз.")
        return

    title = str(result.get("title") or "Задача без названия").strip()
    priority = str(result.get("priority") or "не указана").strip()
    deadline = parse_deadline(result.get("deadline"))

    if deadline is None or deadline <= datetime.now(MOSCOW_TZ):
        draft["title"] = title
        draft["priority"] = priority
        draft["awaiting_deadline"] = True
        pending_tasks[user_id] = draft
        await message.answer(
            "Не удалось определить будущий дедлайн. "
            "Напишите его отдельным сообщением, например: 28.10.2026 18:00."
        )
        return

    pending_tasks[user_id] = {
        "original_text": draft["original_text"],
        "title": title,
        "deadline": deadline.isoformat(),
        "priority": priority,
        "awaiting_deadline": False,
    }

    await message.answer(
        "Проверьте задачу:\n"
        f"Название: {title}\n"
        f"Дедлайн: {format_deadline(deadline)}\n"
        f"Приоритет: {priority}\n\n"
        "Сохранить?",
        reply_markup=confirmation_keyboard(),
    )


async def handle_text(message: Message, text: str) -> None:
    user_id = message.from_user.id
    draft = pending_tasks.get(user_id)

    if draft and draft.get("awaiting_deadline"):
        draft["parse_text"] += f"\nДедлайн: {text.strip()}"
        draft["awaiting_deadline"] = False
        await parse_and_show(message, user_id, draft)
        return

    new_draft = {
        "original_text": text.strip(),
        "parse_text": text.strip(),
        "awaiting_deadline": False,
    }
    await parse_and_show(message, user_id, new_draft)


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Перешлите сообщение коллеги или отправьте его текстом "
        "или голосовым. Я предложу название, дедлайн и приоритет. "
        "Проверьте их и подтвердите сохранение.\n\n"
        "/tasks — список активных задач\n"
        "/done ID — отметить задачу выполненной"
    )


@dp.message(Command("tasks"))
async def show_tasks(message: Message) -> None:
    rows = await get_active_tasks(message.from_user.id)

    if not rows:
        await message.answer("Активных задач пока нет.")
        return

    lines = ["Активные задачи:"]

    for row in rows:
        deadline = parse_deadline(row.get("deadline"))
        deadline_text = format_deadline(deadline) if deadline else "не задан"
        lines.append(
            f"#{row['id']} · {row['title']} · до {deadline_text}"
        )

    lines.append("\nЧтобы завершить задачу: /done ID")
    await message.answer("\n".join(lines))


@dp.message(Command("done"))
async def complete_task(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer(
            "Укажите номер задачи командой /done ID. "
            "Номера можно посмотреть через /tasks."
        )
        return

    task_id = int(parts[1])
    updated = await mark_done(task_id, message.from_user.id)

    if not updated:
        await message.answer("Активная задача с таким номером не найдена.")
        return

    job_id = f"task-{task_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    await message.answer(f"Задача #{task_id} отмечена выполненной.")


@dp.message(F.voice)
async def handle_voice(message: Message, bot: Bot) -> None:
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="telegram_voice_",
            suffix=".ogg",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        await bot.download(message.voice.file_id, destination=temp_path)
        text = await asyncio.to_thread(ml.transcribe, temp_path)

        if not isinstance(text, str) or not text.strip():
            await message.answer(
                "Не удалось распознать голосовое. "
                "Попробуйте ещё раз или отправьте текстом."
            )
            return

        await handle_text(message, text.strip())

    except Exception:
        logger.exception("Не удалось обработать голосовое сообщение")
        await message.answer(
            "Не удалось обработать голосовое. "
            "Попробуйте ещё раз или отправьте текстом."
        )

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Не удалось удалить временный файл %s", temp_path)


@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text_message(message: Message) -> None:
    text = (message.text or "").strip()
    if text:
        await handle_text(message, text)


async def send_reminder(
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
        logger.exception("Не удалось отправить напоминание для задачи #%s", task_id)


def schedule_reminder(
    bot: Bot,
    task_id: int,
    user_id: int,
    deadline_value: str | None,
    original_text: str,
    title: str,
) -> None:
    deadline = parse_deadline(deadline_value)

    if deadline is None:
        logger.error("У задачи #%s некорректный дедлайн", task_id)
        return

    scheduler.add_job(
        send_reminder,
        trigger=DateTrigger(
            run_date=max(deadline, datetime.now(MOSCOW_TZ)),
            timezone=MOSCOW_TZ,
        ),
        args=[bot, user_id, task_id, title, original_text],
        id=f"task-{task_id}",
        replace_existing=True,
        misfire_grace_time=None,
    )


@dp.callback_query(F.data == "confirm_task")
async def confirm_task(callback: CallbackQuery, bot: Bot) -> None:
    user_id = callback.from_user.id
    draft = pending_tasks.get(user_id)

    if not draft or not draft.get("deadline"):
        await callback.answer(
            "Черновик устарел. Отправьте сообщение ещё раз.",
            show_alert=True,
        )
        return

    task_id = await add_task(
        user_id=user_id,
        original_text=draft["original_text"],
        title=draft["title"],
        deadline=draft["deadline"],
        priority=draft["priority"],
    )

    pending_tasks.pop(user_id, None)

    schedule_reminder(
        bot=bot,
        task_id=task_id,
        user_id=user_id,
        deadline_value=draft["deadline"],
        original_text=draft["original_text"],
        title=draft["title"],
    )

    await callback.answer("Задача сохранена")
    if callback.message:
        await callback.message.edit_text(
            f"Задача #{task_id} сохранена. Напомню о ней в дедлайн."
        )


@dp.callback_query(F.data == "cancel_task")
async def cancel_task(callback: CallbackQuery) -> None:
    pending_tasks.pop(callback.from_user.id, None)
    await callback.answer("Задача не сохранена")

    if callback.message:
        await callback.message.edit_text(
            "Задача не сохранена. Можете прислать другое сообщение."
        )


async def main() -> None:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")

    if not token:
        raise RuntimeError("BOT_TOKEN не найден. Добавьте его в файл .env.")

    await init_db()
    bot = Bot(token=token)

    for task in await get_active_tasks():
        schedule_reminder(
            bot=bot,
            task_id=task["id"],
            user_id=task["user_id"],
            deadline_value=task.get("deadline"),
            original_text=task["original_text"],
            title=task["title"],
        )

    scheduler.start()

    try:
        logger.info("Бот запущен; часовой пояс: Europe/Moscow")
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
