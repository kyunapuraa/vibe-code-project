# Telegram-бот для напоминаний

Нужен Python 3.11 или новее. Перед запуском в папке `backend` должны быть файлы `bot.py`, `db.py`, `keyboards.py` и `ml.py`.

## Запуск в Windows PowerShell

Открой PowerShell в папке проекта и выполни:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
