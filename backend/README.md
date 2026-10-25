# Telegram-бот для напоминаний

Нужен Python 3.11 или новее. Бот лежит в папке `backend` (`bot.py`, `db.py`, `keyboards.py`), ML-модули — в папке `ml` (`ml.py`, `extract.py`, `embeddings.py`, `knowledge.py`); `bot.py` сам добавляет папку `ml` в путь импорта.

## Запуск в Windows PowerShell

Открой PowerShell в папке проекта и выполни:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r ..\ml\requirements.txt
Copy-Item .env.example .env
notepad .env
```

## База знаний (поиск по конспектам)

Бизнес-логика:

1. Пользователь присылает боту файл (PDF, DOCX, TXT, MD), фото конспекта или текст через `/note`.
2. Бот скачивает материал и привязывает его к `user_id` (таблица `documents`).
3. ML1 (`ml/extract.py`) достаёт текст и делит его на куски.
4. ML2 (`ml/embeddings.py`) превращает каждый кусок в эмбеддинг (`intfloat/multilingual-e5-small`, 384 числа).
5. Куски сохраняются в таблицу `chunks`: `text`, `page`, `embedding` (BLOB float32), `document_id`, `user_id`.
6. По `/ask вопрос` запрос тоже превращается в эмбеддинг, среди кусков **только этого пользователя** ищется ближайший, бот отвечает его текстом и источником.

Связка ML собрана в `ml/knowledge.py`: `index_file`, `index_text`, `find`.

Команды:

- файл или фото — сохранить материал;
- `/note текст` — сохранить текст как заметку;
- `/ask вопрос` (или `/search`) — найти фрагмент;
- `/docs` — список материалов;
- `/forget ID` — удалить материал.

Для фото и PDF-сканов нужен Tesseract с русским языком (в Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-rus poppler-utils`). Обычные PDF, DOCX и TXT работают без него.

Тесты: `pytest backend/test_knowledge_flow.py ml/`.
