import os

os.environ["ML_DISABLE_WHISPER"] = "1"

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import ml

TZ = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=TZ)


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.mark.parametrize(
    "text, deadline, priority",
    [
        ("Сделай отчёт до пятницы 18:00", "2026-10-30T18:00:00+03:00", "normal"),
        ("Пришли презентацию завтра к 10 утра", "2026-10-26T10:00:00+03:00", "high"),
        ("Через 2 часа созвонимся по проекту", "2026-10-25T14:00:00+03:00", "high"),
        ("Срочно! Ответь клиенту сегодня до 17:30", "2026-10-25T17:30:00+03:00", "high"),
        ("Когда будет время, глянь макет", None, "low"),
        ("Подготовь договор к 28 октября", "2026-10-28T18:00:00+03:00", "normal"),
        ("В понедельник утром отправь счёт", "2026-10-26T10:00:00+03:00", "high"),
        ("Позвони поставщику в среду вечером", "2026-10-28T19:00:00+03:00", "normal"),
        ("Сделай ревью через 3 дня", "2026-10-28T18:00:00+03:00", "normal"),
        ("Напиши пост к 05.11", "2026-11-05T18:00:00+03:00", "normal"),
        ("Проверь отчёт к концу недели", "2026-10-30T18:00:00+03:00", "normal"),
        ("Пришли счёт через полчаса", "2026-10-25T12:30:00+03:00", "high"),
        ("В следующий вторник принеси ноутбук", "2026-11-03T18:00:00+03:00", "normal"),
        ("Посмотри баг, горит", None, "high"),
        ("Нужно обновить сайт", None, "normal"),
        ("Не срочно, но до пятницы скинь фото", "2026-10-30T18:00:00+03:00", "normal"),
        ("Ответь до 3 часов дня", "2026-10-25T15:00:00+03:00", "high"),
    ],
)
def test_deadline_and_priority(text, deadline, priority):
    result = ml.parse_task(text, NOW)
    assert result["deadline"] == deadline
    assert result["priority"] == priority


@pytest.mark.parametrize(
    "text, title",
    [
        ("Сделай отчёт до пятницы 18:00", "Сделать отчёт"),
        ("Пришли презентацию завтра к 10 утра", "Прислать презентацию"),
        ("Через 2 часа созвонимся по проекту", "Созвониться по проекту"),
        ("Срочно! Ответь клиенту сегодня до 17:30", "Ответить клиенту"),
        ("Когда будет время, глянь макет", "Глянуть макет"),
        ("Саша, привет! Нужно проверить код до 15:00", "Проверить код"),
    ],
)
def test_title(text, title):
    assert ml.parse_task(text, NOW)["title"] == title


def test_contract_keys_and_types():
    result = ml.parse_task("Сделай отчёт до пятницы 18:00", NOW)
    assert set(result) == {"title", "deadline", "priority"}
    assert isinstance(result["title"], str) and result["title"]
    assert result["priority"] in {"high", "normal", "low"}
    datetime.fromisoformat(result["deadline"])


def test_title_max_length():
    text = "Подготовь " + "очень подробный аналитический отчёт по продажам " * 5 + "до пятницы"
    assert len(ml.parse_task(text, NOW)["title"]) <= 80


@pytest.mark.parametrize(
    "text",
    ["Сделай это до 10:00", "Отправь до понедельника", "Сделай к 20 октября", "Сдай отчёт сегодня до 10:00"],
)
def test_deadline_never_in_past(text):
    result = ml.parse_task(text, NOW)
    if result["deadline"] is not None:
        assert datetime.fromisoformat(result["deadline"]) > NOW


def test_naive_now_is_moscow():
    result = ml.parse_task("Пришли презентацию завтра к 10 утра", datetime(2026, 10, 25, 12, 0))
    assert result["deadline"] == "2026-10-26T10:00:00+03:00"


@pytest.mark.parametrize("bad", ["", None, "   "])
def test_empty_text_does_not_crash(bad):
    result = ml.parse_task(bad, NOW)
    assert result == {"title": "Выполнить задачу", "deadline": None, "priority": "normal"}


def test_llm_invalid_json_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(ml, "_llm_json", lambda *a, **k: {"title": 123, "deadline": "вчера"})
    result = ml.parse_task("Сделай отчёт до пятницы 18:00", NOW)
    assert result["title"] == "Сделать отчёт"
    assert result["deadline"] == "2026-10-30T18:00:00+03:00"


def test_llm_valid_answer_used(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(
        ml, "_llm_json",
        lambda *a, **k: {"title": "Согласовать бюджет", "deadline": "2026-10-29T12:00:00+03:00"},
    )
    result = ml.parse_task("Как обсуждали на планёрке, там по бюджету надо решить", NOW)
    assert result["title"] == "Согласовать бюджет"
    assert result["deadline"] == "2026-10-29T12:00:00+03:00"


def test_llm_past_deadline_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(
        ml, "_llm_json",
        lambda *a, **k: {"title": "Согласовать бюджет", "deadline": "2026-10-20T12:00:00+03:00"},
    )
    result = ml.parse_task("По бюджету надо решить", NOW)
    assert result["deadline"] is None


def test_llm_unreachable_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9")
    result = ml.parse_task("Срочно! Ответь клиенту сегодня до 17:30", NOW)
    assert result["deadline"] == "2026-10-25T17:30:00+03:00"
    assert result["priority"] == "high"


def test_analyze_task_contract():
    result = ml.analyze_task("Сделай отчёт до пятницы 18:00")
    assert set(result) == {"steps", "draft_reply"}
    assert 3 <= len(result["steps"]) <= 6
    assert all(isinstance(s, str) and s for s in result["steps"])
    assert isinstance(result["draft_reply"], str) and result["draft_reply"]


def test_analyze_task_llm_bad_steps_fallback(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(ml, "_llm_json", lambda *a, **k: {"steps": ["один шаг"], "draft_reply": "ok"})
    result = ml.analyze_task("Глянь макет")
    assert 3 <= len(result["steps"]) <= 6


def test_analyze_task_llm_trims_to_six(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    steps = [f"Шаг {i}" for i in range(10)]
    monkeypatch.setattr(ml, "_llm_json", lambda *a, **k: {"steps": steps, "draft_reply": "Принял!"})
    result = ml.analyze_task("Глянь макет")
    assert len(result["steps"]) == 6
    assert result["draft_reply"] == "Принял!"


def test_analyze_task_empty():
    result = ml.analyze_task("")
    assert 3 <= len(result["steps"]) <= 6


def test_transcribe_missing_file_returns_empty():
    assert ml.transcribe("/no/such/file.ogg") == ""


def test_transcribe_empty_path_returns_empty():
    assert ml.transcribe("") == ""