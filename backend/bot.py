import asyncio
import logging
import os
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

import knowledge
import ml
from db import (
    add_document,
    add_task,
    delete_document,
    get_active_tasks,
    get_documents,
    get_user_chunks,
    init_db,
    mark_done,
)
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
        "/done ID — отметить задачу выполненной\n\n"
        "База знаний:\n"
        "• пришлите файл (PDF, DOCX, TXT, MD) или фото конспекта — я сохраню его\n"
        "/note текст — сохранить текст как заметку\n"
        "/ask вопрос — найти нужный фрагмент в ваших материалах\n"
        "/docs — список сохранённых материалов\n"
        "/forget ID — удалить материал"
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


MAX_FILE_SIZE = 20 * 1024 * 1024
SNIPPET_LEN = 150


async def save_records(
    message: Message,
    user_id: int,
    source: str,
    kind: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        await message.answer(
            "Не нашёл в материале текста для сохранения. "
            "Проверьте файл или пришлите текст через /note."
        )
        return

    document_id = await add_document(user_id, source, kind, records)
    await message.answer(
        f"Сохранил «{source}» (#{document_id}): {len(records)} фрагм. текста.\n"
        "Теперь можно искать по нему: /ask ваш вопрос"
    )


async def index_telegram_file(
    message: Message,
    bot: Bot,
    file_id: str,
    source: str,
    suffix: str,
) -> None:
    temp_path: str | None = None
    status = await message.answer("Обрабатываю материал, это может занять немного времени…")

    try:
        with tempfile.NamedTemporaryFile(
            prefix="telegram_doc_",
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        await bot.download(file_id, destination=temp_path)
        records = await asyncio.to_thread(knowledge.index_file, temp_path, source)
        await save_records(message, message.from_user.id, source, "file", records)

    except ValueError as error:
        await message.answer(f"Не получилось обработать файл: {error}")

    except Exception:
        logger.exception("Не удалось проиндексировать файл %s", source)
        await message.answer(
            "Не удалось обработать файл. Для PDF-сканов и фото нужно "
            "распознавание текста (Tesseract) на сервере; можно прислать текст через /note."
        )

    finally:
        try:
            await status.delete()
        except Exception:
            pass
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Не удалось удалить временный файл %s", temp_path)


@dp.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    document = message.document
    name = document.file_name or "документ"
    suffix = Path(name).suffix.lower()

    if suffix not in knowledge.SUPPORTED_EXT:
        await message.answer(
            "Этот формат пока не поддерживается. "
            "Подходят: " + ", ".join(knowledge.SUPPORTED_EXT)
        )
        return

    if document.file_size and document.file_size > MAX_FILE_SIZE:
        await message.answer("Файл больше 20 МБ, Telegram не даёт боту его скачать.")
        return

    await index_telegram_file(message, bot, document.file_id, name, suffix)


@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    photo = message.photo[-1]
    source = message.caption.strip() if message.caption else (
        "Фото от " + datetime.now(MOSCOW_TZ).strftime("%d.%m.%Y %H:%M")
    )
    await index_telegram_file(message, bot, photo.file_id, source, ".jpg")


@dp.message(Command("note"))
async def save_note(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        await message.answer("Напишите текст после команды: /note Матрица — это таблица чисел…")
        return

    text = parts[1].strip()
    first_line = text.splitlines()[0]
    source = first_line[:40] + ("…" if len(first_line) > 40 else "")

    try:
        records = await asyncio.to_thread(knowledge.index_text, text, source)
        await save_records(message, message.from_user.id, source, "note", records)
    except Exception:
        logger.exception("Не удалось сохранить заметку")
        await message.answer("Не удалось сохранить заметку. Попробуйте ещё раз.")


def format_source(item: dict[str, Any]) -> str:
    source = item.get("source") or "материал"
    page = item.get("page")
    return f"«{source}», стр. {page}" if page and page > 1 else f"«{source}»"


@dp.message(Command("ask", "search"))
async def ask(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        await message.answer("Напишите вопрос после команды: /ask где я изучал матрицы")
        return

    query = parts[1].strip()
    rows = await get_user_chunks(message.from_user.id)

    if not rows:
        await message.answer(
            "У вас пока нет сохранённых материалов. "
            "Пришлите файл или фото конспекта, либо текст через /note."
        )
        return

    try:
        top = await asyncio.to_thread(knowledge.find, query, rows, 3)
    except Exception:
        logger.exception("Ошибка поиска")
        top = []

    if not top:
        await message.answer("Не удалось ничего найти. Попробуйте переформулировать вопрос.")
        return

    best = top[0]
    lines = [f"Нашёл в {format_source(best)}:", "", best["text"]]

    others = top[1:]
    if others:
        lines += ["", "Ещё похоже:"]
        for item in others:
            snippet = item["text"][:SNIPPET_LEN]
            if len(item["text"]) > SNIPPET_LEN:
                snippet += "…"
            lines.append(f"• {format_source(item)}: {snippet}")

    await message.answer("\n".join(lines)[:4000])


@dp.message(Command("docs"))
async def show_documents(message: Message) -> None:
    documents = await get_documents(message.from_user.id)

    if not documents:
        await message.answer("Сохранённых материалов пока нет.")
        return

    lines = ["Ваши материалы:"]
    for doc in documents:
        lines.append(f"#{doc['id']} · {doc['source']} · фрагментов: {doc['chunks']}")
    lines.append("\nУдалить: /forget ID")
    await message.answer("\n".join(lines))


@dp.message(Command("forget"))
async def forget_document(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer("Укажите номер материала: /forget ID. Номера — в /docs.")
        return

    deleted = await delete_document(int(parts[1]), message.from_user.id)
    await message.answer(
        f"Материал #{parts[1]} удалён." if deleted else "Материал с таким номером не найден."
    )


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
    asyncio.get_running_loop().run_in_executor(None, knowledge.warmup)

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
