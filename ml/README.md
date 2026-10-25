\# ML-модуль бота-напоминалки (ml.py)



Модуль вычленяет задачу и дедлайн из сообщения коллеги (текст или голосовое), определяет приоритет, составляет план шагов и черновик ответа.



Бэкенд импортирует только три функции:



```python

from ml import transcribe, parse\_task, analyze\_task

```



\## Установка (Ubuntu / WSL)



```bash

sudo apt update

sudo apt install -y ffmpeg python3-venv python3-full

cd \~/taskbot

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

```



При первом импорте `ml.py` скачивается модель faster-whisper `small` (около 500 МБ), дальше она берётся из кэша `\~/.cache/huggingface`.



\## Переменные окружения (.env)



| Переменная | Обязательна | По умолчанию | Назначение |

|---|---|---|---|

| `LLM\_API\_KEY` | нет | — | Ключ LLM. Без него работает только разбор правилами |

| `LLM\_BASE\_URL` | нет | `https://api.openai.com/v1` | Любой OpenAI-совместимый API (OpenAI, OpenRouter, DeepSeek, Ollama) |

| `LLM\_MODEL` | нет | `gpt-4o-mini` | Имя модели |

| `ML\_DISABLE\_WHISPER` | нет | — | `1` — не загружать Whisper (для тестов) |



Пример `.env`:



```

BOT\_TOKEN=токен\_бота

LLM\_API\_KEY=ключ

LLM\_BASE\_URL=https://api.openai.com/v1

LLM\_MODEL=gpt-4o-mini

```



\## Контракт



\### transcribe(path: str) -> str



Принимает путь к голосовому `.ogg` (opus) из Telegram, возвращает русский текст. При ошибке возвращает `""` и пишет причину в лог.



\### parse\_task(text: str, now: datetime) -> dict



```python

{"title": "Подготовить отчёт", "deadline": "2026-10-30T18:00:00+03:00", "priority": "normal"}

```



\- `title` — до 80 символов, начинается с глагола;

\- `deadline` — ISO-строка с часовым поясом или `None`, никогда не в прошлом;

\- `priority` — `high` (срочно / сегодня / горит / asap или срок меньше 24 часов), `low` (нет срока и «когда будет время» / «не срочно»), иначе `normal`.



Разбор идёт в два уровня: правила (regex + dateparser + pymorphy3), затем LLM, если задан ключ. Ответ LLM валидируется; при ошибке остаётся результат правил. Время по умолчанию: утро 10:00, вечер 19:00, без времени 18:00.



\### analyze\_task(text: str) -> dict



```python

{"steps": \["шаг 1", "шаг 2", "шаг 3"], "draft\_reply": "Привет! Принял задачу..."}

```



3–6 шагов. Без LLM используются шаблоны по типу задачи (отчёт, презентация, клиент, встреча, макет, код).



Все функции перехватывают исключения и всегда возвращают ответ по контракту.



\## Примеры вызова



```python

import asyncio

from datetime import datetime

from zoneinfo import ZoneInfo

from ml import transcribe, parse\_task, analyze\_task



now = datetime.now(ZoneInfo("Europe/Moscow"))



print(parse\_task("Сделай отчёт до пятницы 18:00", now))

print(analyze\_task("Сделай отчёт до пятницы 18:00"))

print(transcribe("voice.ogg"))

```



В aiogram распознавание и запросы к LLM синхронные и занимают секунды, поэтому вызывать их нужно в отдельном потоке, чтобы бот не зависал:



```python

text = await asyncio.to\_thread(transcribe, path)

task = await asyncio.to\_thread(parse\_task, text, now)

plan = await asyncio.to\_thread(analyze\_task, text)

```



Логи пишутся в логгер `ml`; настройка `logging.basicConfig(...)` делается в `bot.py`.



\## Тесты



```bash

source venv/bin/activate

pytest -v test\_ml.py

```



Тесты не обращаются к реальной LLM и не загружают Whisper.

