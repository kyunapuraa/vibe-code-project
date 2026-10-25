import json
import logging
import os
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import dateparser
import requests
from dateparser.search import search_dates
from dotenv import load_dotenv

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except Exception:
    _morph = None

load_dotenv()

logger = logging.getLogger("ml")

TZ = ZoneInfo("Europe/Moscow")
DEFAULT_TIME = time(18, 0)
MORNING_TIME = time(10, 0)
DAY_TIME = time(14, 0)
LUNCH_TIME = time(13, 0)
NOON_TIME = time(12, 0)
EVENING_TIME = time(19, 0)
NIGHT_TIME = time(23, 0)
TITLE_MAX_LEN = 80
LLM_TIMEOUT = 20

__all__ = ["transcribe", "parse_task", "analyze_task"]


def _load_whisper():
    if os.getenv("ML_DISABLE_WHISPER") == "1":
        logger.info("Загрузка faster-whisper отключена (ML_DISABLE_WHISPER=1)")
        return None
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("Модель faster-whisper 'small' загружена")
        return model
    except Exception:
        logger.exception("Не удалось загрузить модель faster-whisper")
        return None


_whisper_model = _load_whisper()


def transcribe(path: str) -> str:
    try:
        if _whisper_model is None:
            logger.error("transcribe: модель распознавания не загружена")
            return ""
        if not path or not os.path.isfile(path):
            logger.error("transcribe: файл не найден: %r", path)
            return ""
        if os.path.getsize(path) == 0:
            logger.error("transcribe: пустой файл: %r", path)
            return ""
        segments, info = _whisper_model.transcribe(
            path, language="ru", beam_size=5, vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            logger.warning("transcribe: речь не распознана в %r", path)
        else:
            logger.info("transcribe: %.1f c аудио -> %d символов", info.duration, len(text))
        return text
    except Exception:
        logger.exception("transcribe: ошибка распознавания %r", path)
        return ""


WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресень": 6,
}

NUM_WORDS = {
    "один": 1, "одну": 1, "одна": 1, "пару": 2, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "десять": 10,
}

URGENT_WORDS = re.compile(
    r"\b(срочн\w*|сегодня|горит|asap|как можно скорее|немедленно|прямо сейчас)\b"
)
LOW_PHRASES = re.compile(
    r"(когда (будет|появится) (время|минутка)|как (будет|появится) (время|минутка)"
    r"|когда освободишься|не срочно|не горит|без спешки|в свободное время)"
)
DEADLINE_HINTS = re.compile(
    r"(\d|сегодня|завтра|послезавтра|через|понедельник|вторник|сред[уаые]\b|четверг"
    r"|пятниц|суббот|воскресень|недел|месяц|утр|вечер|обед|январ|феврал|март|апрел"
    r"|ма[яй]\b|июн|июл|август|сентябр|октябр|ноябр|декабр|конц|срок|дедлайн)"
)

MONTHS_RE = (
    r"(январ[яь]|феврал[яь]|марта?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|августа?"
    r"|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])"
)


def _to_moscow(now: datetime) -> datetime:
    if not isinstance(now, datetime):
        return datetime.now(TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=TZ)
    return now.astimezone(TZ)


def _dp_settings(now: datetime) -> dict:
    return {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": now.replace(tzinfo=None),
        "TIMEZONE": "Europe/Moscow",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }


def _hour_with_suffix(hour: int, suffix: str | None) -> int | None:
    if suffix:
        if suffix.startswith("утр"):
            hour = 0 if hour == 12 else hour
        elif suffix.startswith("дн") or suffix.startswith("вечер"):
            hour = hour + 12 if hour < 12 else hour
        elif suffix.startswith("ноч"):
            hour = 0 if hour == 12 else (hour + 12 if 6 <= hour < 12 else hour)
    if 0 <= hour <= 23:
        return hour
    return None


def _num(token: str | None) -> float:
    if not token:
        return 1
    if token.isdigit():
        return int(token)
    if token == "пол":
        return 0.5
    if token.startswith("полтор"):
        return 1.5
    return NUM_WORDS.get(token, 1)


class _Extractor:
    def __init__(self, text: str, now: datetime):
        self.now = now
        self.s = text.lower().replace("ё", "е")
        self.spans: list[tuple[int, int]] = []
        self.delta: timedelta | None = None
        self.day_offset_delta: timedelta | None = None
        self.date: date | None = None
        self.date_kind: str | None = None
        self.time: time | None = None
        self.part: time | None = None

    def _take(self, m: re.Match, group: int = 0):
        start, end = m.span(group)
        self.spans.append((start, end))
        self.s = self.s[:start] + " " * (end - start) + self.s[end:]

    def _find(self, pattern: str):
        return list(re.finditer(pattern, self.s))

    def run(self):
        self._relative()
        self._time_colon()
        self._time_dot()
        self._text_date()
        self._numeric_date()
        self._hour_suffix()
        self._hour_word()
        self._day_words()
        self._weekday()
        self._end_of()
        self._part_of_day()
        self._bare_hour()
        if not self._found_anything():
            self._dateparser_fallback()
        return self

    def _found_anything(self) -> bool:
        return any([self.delta, self.day_offset_delta, self.date, self.time, self.part])

    def _relative(self):
        pattern = (
            r"\bчерез\s+(?:(\d+|пару|полтор[аы]|один|одну|одна|два|две|три|четыре|пять|шесть|семь|десять)\s*)?"
            r"(пол)?(минут\w*|мин\b|час\w*|сутк\w*|дн\w*|день|недел\w*|месяц\w*)"
        )
        for m in self._find(pattern):
            amount = 0.5 if m.group(2) else _num(m.group(1))
            unit = m.group(3)
            if unit.startswith("мин"):
                self.delta = timedelta(minutes=amount)
            elif unit.startswith("час"):
                self.delta = timedelta(hours=amount)
            elif unit.startswith("сутк"):
                self.delta = timedelta(days=amount)
            elif unit.startswith("недел"):
                self.day_offset_delta = timedelta(weeks=amount)
            elif unit.startswith("месяц"):
                self.day_offset_delta = timedelta(days=30 * amount)
            else:
                self.day_offset_delta = timedelta(days=amount)
            self._take(m)
            return
        m = re.search(r"\bчерез\s+полчаса\b", self.s)
        if m:
            self.delta = timedelta(minutes=30)
            self._take(m)

    def _time_colon(self):
        pattern = r"(?:\b(?:в|к|до|на|около|не позднее)\s+)?\b([01]?\d|2[0-3]):([0-5]\d)\b"
        for m in self._find(pattern):
            self.time = time(int(m.group(1)), int(m.group(2)))
            self._take(m)
            return

    def _time_dot(self):
        pattern = r"\b(?:в|около)\s+([01]?\d|2[0-3])\.([0-5]\d)\b(?!\.\d)"
        for m in self._find(pattern):
            self.time = time(int(m.group(1)), int(m.group(2)))
            self._take(m)
            return

    def _text_date(self):
        pattern = (
            r"(?:\b(?:до|к|ко|на|не позднее|в течение)\s+)?\b(\d{1,2})(?:-?го)?\s+"
            + MONTHS_RE + r"(?:\s+(\d{4})(?:\s*(?:года|г\.?))?)?"
        )
        for m in self._find(pattern):
            fragment = f"{m.group(1)} {m.group(2)}" + (f" {m.group(3)}" if m.group(3) else "")
            parsed = None
            try:
                parsed = dateparser.parse(fragment, languages=["ru"], settings=_dp_settings(self.now))
            except Exception:
                logger.exception("dateparser не разобрал %r", fragment)
            if parsed:
                self.date = parsed.date()
                self.date_kind = "date_with_year" if m.group(3) else "date"
                self._take(m)
                return

    def _numeric_date(self):
        pattern = r"(?:\b(?:до|к|ко|на|не позднее)\s+)?\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b"
        for m in self._find(pattern):
            day, month = int(m.group(1)), int(m.group(2))
            year_raw = m.group(3)
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue
            if year_raw:
                year = int(year_raw)
                year = year + 2000 if year < 100 else year
            else:
                year = self.now.year
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            self.date = d
            self.date_kind = "date_with_year" if year_raw else "date"
            self._take(m)
            return

    def _hour_suffix(self):
        pattern = (
            r"(?:\b(?:в|к|до|на|около|не позднее)\s+)?\b(\d{1,2})(?:[.:](\d{2}))?\s*"
            r"(?:час\w*\s+)?(утра|дня|вечера|ночи)\b"
        )
        for m in self._find(pattern):
            hour = _hour_with_suffix(int(m.group(1)), m.group(3))
            minute = int(m.group(2)) if m.group(2) else 0
            if hour is None or minute > 59:
                continue
            self.time = time(hour, minute)
            self._take(m)
            return

    def _hour_word(self):
        pattern = r"\b(?:в|к|до|около|не позднее)\s+(\d{1,2})\s*(?:час(?:а|ов)?)\b"
        for m in self._find(pattern):
            hour = int(m.group(1))
            if 1 <= hour <= 7:
                hour += 12
            if 0 <= hour <= 23:
                self.time = time(hour, 0)
                self._take(m)
                return
        m = re.search(r"\b(?:в|к|до|около)\s+(час|полдень)\b(?:\s+дня)?", self.s)
        if m:
            self.time = time(13, 0) if m.group(1) == "час" else NOON_TIME
            self._take(m)

    def _day_words(self):
        offsets = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
        for m in self._find(r"(?:\b(?:до|на|к)\s+)?\b(сегодня|завтра|послезавтра)\b"):
            self.date = (self.now + timedelta(days=offsets[m.group(1)])).date()
            self.date_kind = "today" if m.group(1) == "сегодня" else "dayword"
            self._take(m)
            return

    def _weekday(self):
        if self.date is not None:
            return
        pattern = (
            r"(?:\b(?:до|к|ко|в|во|на|по)\s+)?(?:\b(следующ\w*|эт\w+|ближайш\w*)\s+)?"
            r"\b(понедельник\w*|вторник\w*|сред[уаые]|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*)\b"
        )
        for m in self._find(pattern):
            word = m.group(2)
            target = next(v for k, v in WEEKDAYS.items() if word.startswith(k))
            days_ahead = (target - self.now.weekday()) % 7
            if m.group(1) and m.group(1).startswith("следующ"):
                days_ahead = days_ahead + 7 if days_ahead <= 2 else days_ahead
            self.date = (self.now + timedelta(days=days_ahead)).date()
            self.date_kind = "weekday"
            self._take(m)
            return

    def _end_of(self):
        m = re.search(r"(?:\b(?:к|до|на)\s+)?\bконц[ауе]?\s+(дня|недели|месяца|года)\b", self.s)
        if m:
            unit = m.group(1)
            if unit == "дня":
                if self.date is None:
                    self.date = self.now.date()
                    self.date_kind = "today"
            elif unit == "недели":
                days_ahead = (4 - self.now.weekday()) % 7
                self.date = (self.now + timedelta(days=days_ahead)).date()
                self.date_kind = "weekday"
            elif unit == "месяца":
                first_next = (self.now.replace(day=1) + timedelta(days=32)).replace(day=1)
                self.date = (first_next - timedelta(days=1)).date()
                self.date_kind = "date"
            else:
                self.date = date(self.now.year, 12, 31)
                self.date_kind = "date"
            self._take(m)
            return
        m = re.search(r"\bна\s+(этой|текущей)\s+неделе\b", self.s)
        if m and self.date is None:
            days_ahead = (4 - self.now.weekday()) % 7
            self.date = (self.now + timedelta(days=days_ahead)).date()
            self.date_kind = "weekday"
            self._take(m)

    def _part_of_day(self):
        if self.time is not None:
            m = re.search(r"\b(утром|с утра|днем|вечером|ночью)\b", self.s)
            if m:
                self._take(m)
            return
        parts = [
            (r"(?:\b(?:до|к|в)\s+)?\bполд(?:ень|ня)\b", NOON_TIME),
            (r"(?:\b(?:до|к|в|после)\s+)?\bобед[ау]?\b", LUNCH_TIME),
            (r"(?:\b(?:с|до|к)\s+)?\bутр(?:ом|а|у)\b", MORNING_TIME),
            (r"\bднем\b", DAY_TIME),
            (r"(?:\b(?:до|к)\s+)?\bвечер(?:ом|а|у)?\b", EVENING_TIME),
            (r"\bночью\b", NIGHT_TIME),
        ]
        for pattern, value in parts:
            m = re.search(pattern, self.s)
            if m:
                self.part = value
                self._take(m)
                return

    def _bare_hour(self):
        if self.time is not None or self.date is None:
            return
        m = re.search(r"\b(?:в|к|до|около)\s+(\d{1,2})\b(?!\s*[./:%\d])", self.s)
        if m:
            hour = int(m.group(1))
            if 1 <= hour <= 7:
                hour += 12
            if 8 <= hour <= 23:
                self.time = time(hour, 0)
                self._take(m)

    def _dateparser_fallback(self):
        if not DEADLINE_HINTS.search(self.s):
            return
        try:
            results = search_dates(self.s, languages=["ru"], settings=_dp_settings(self.now))
        except Exception:
            logger.exception("search_dates упал")
            return
        if not results:
            return
        for fragment, dt in results:
            if not re.search(r"\d|" + MONTHS_RE, fragment):
                continue
            dt = _to_moscow(dt)
            idx = self.s.find(fragment)
            if idx >= 0:
                self.spans.append((idx, idx + len(fragment)))
            has_time = bool(re.search(r"\d{1,2}[:.]\d{2}|час|утра|вечера|дня", fragment))
            self.date = dt.date()
            self.date_kind = "date"
            if has_time:
                self.time = dt.timetz().replace(tzinfo=None)
            return

    def deadline(self) -> datetime | None:
        now = self.now
        if self.delta is not None:
            return (now + self.delta).replace(second=0, microsecond=0)
        target_date = self.date
        kind = self.date_kind
        if self.day_offset_delta is not None and target_date is None:
            target_date = (now + self.day_offset_delta).date()
            kind = "relative_days"
        clock = self.time or self.part
        if target_date is None:
            if clock is None:
                return None
            target_date = now.date()
            kind = "time_only"
        dt = datetime.combine(target_date, clock or DEFAULT_TIME, tzinfo=TZ)
        return _shift_to_future(dt, now, kind)


def _shift_to_future(dt: datetime, now: datetime, kind: str | None) -> datetime | None:
    if dt > now:
        return dt
    if kind == "time_only":
        dt = dt + timedelta(days=1)
    elif kind == "weekday":
        dt = dt + timedelta(days=7)
    elif kind == "date":
        try:
            dt = dt.replace(year=dt.year + 1)
        except ValueError:
            dt = dt + timedelta(days=365)
    else:
        return None
    return dt if dt > now else None


FILLER_START = re.compile(
    r"^(?:привет\w*|здравствуй\w*|добр\w+\s+(?:день|утро|вечер)|доброго\s+\w+|хай|коллеги|ребят\w*"
    r"|слушай|смотри|кстати|ну|а|и|но|однако|так|короче|пожалуйста|плиз|pls|please|срочно|очень|"
    r"нужно|надо|необходимо|требуется|прошу|просьба|можешь|можете|сможешь|сможете"
    r"|не мог(?:ли)? бы(?:\s+(?:ты|вы))?|не забудь(?:те)?|давай(?:те)?|тебе|вам|нам|мне|ты|вы"
    r"|будь добр\w*|если не сложно|горит|asap)\b[\s,!.:;-]*",
    re.IGNORECASE,
)
NOUN_VERB = {
    "созвон": "Провести", "встреча": "Провести", "встречу": "Провести",
    "звонок": "Провести", "митинг": "Провести", "презентация": "Подготовить",
    "отчет": "Подготовить", "отчёт": "Подготовить",
}


def _verb_to_infinitive(word: str) -> str | None:
    if _morph is None:
        return None
    clean = word.strip(".,!?:;").lower()
    parses = _morph.parse(clean)
    for p in parses:
        if "impr" in p.tag or "INFN" in p.tag:
            return p.normal_form
    for p in parses:
        if "VERB" in p.tag and p.score >= 0.3:
            return p.normal_form
    return None


def _make_title(text: str, spans: list[tuple[int, int]]) -> str:
    try:
        chars = list(text)
        if len(text.lower()) == len(text):
            for start, end in spans:
                for i in range(start, min(end, len(chars))):
                    chars[i] = " "
        cleaned = "".join(chars)
        low_text = cleaned.lower().replace("ё", "е")
        for m in LOW_PHRASES.finditer(low_text):
            cleaned = cleaned[:m.start()] + " " * (m.end() - m.start()) + cleaned[m.end():]
        cleaned = re.sub(
            r"(?i)(?:,\s*)?\b(?:пожалуйста|срочно|горит|asap|нужен|нужна|нужно|нужны|надо)\b", " ", cleaned
        )
        chunks = [c.strip(" ,;:-—\t") for c in re.split(r"[.!?\n]+", cleaned)]
        title = ""
        for chunk in chunks:
            chunk = re.sub(r"\s+", " ", chunk).strip(" ,;:-—")
            prev = None
            while chunk and chunk != prev:
                prev = chunk
                chunk = FILLER_START.sub("", chunk).strip(" ,;:-—")
                chunk = re.sub(r"^[А-ЯЁA-Z][а-яёa-z]+,\s*", "", chunk)
            if len(chunk) >= 3:
                title = chunk
                break
        if not title:
            return "Выполнить задачу"
        title = re.sub(r"\s+(?:до|к|ко|в|во|на|по|около|и|с|не позднее)\s*$", "", title).strip(" ,;:-—")
        title = re.sub(r"\s+([,.;:])", r"\1", title)
        words = title.split(" ")
        first = words[0]
        inf = _verb_to_infinitive(first)
        if inf:
            words[0] = inf
            title = " ".join(words)
        else:
            verb = NOUN_VERB.get(first.lower().strip(".,!?:;"), "Сделать")
            title = verb + " " + first.lower() + (" " + " ".join(words[1:]) if len(words) > 1 else "")
        title = title[0].upper() + title[1:]
        if len(title) > TITLE_MAX_LEN:
            cut = title[:TITLE_MAX_LEN]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            title = cut.rstrip(" ,;:-—")
        return title
    except Exception:
        logger.exception("Не удалось построить title")
        return "Выполнить задачу"


def _priority(text: str, deadline: datetime | None, now: datetime) -> str:
    s = text.lower().replace("ё", "е")
    low_hit = bool(LOW_PHRASES.search(s))
    s_wo_low = LOW_PHRASES.sub(" ", s)
    if URGENT_WORDS.search(s_wo_low):
        return "high"
    if deadline is not None and deadline - now < timedelta(hours=24):
        return "high"
    if deadline is None and low_hit:
        return "low"
    return "normal"


def _parse_rules(text: str, now: datetime) -> tuple[str, datetime | None]:
    ex = _Extractor(text, now).run()
    deadline = ex.deadline()
    title = _make_title(text, ex.spans)
    return title, deadline


def _extract_json(content: str) -> dict | None:
    if not content:
        return None
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    data = json.loads(content[start:end + 1])
    return data if isinstance(data, dict) else None


def _llm_json(system_prompt: str, user_prompt: str) -> dict | None:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=LLM_TIMEOUT)
        if resp.status_code == 400:
            payload.pop("response_format")
            resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = _extract_json(content)
        if data is None:
            logger.warning("LLM вернула не JSON: %r", content[:200])
        return data
    except Exception as e:
        logger.warning("Запрос к LLM не удался: %s", e)
        return None


RU_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

PARSE_SYSTEM = (
    "Ты извлекаешь задачу из сообщения коллеги. Отвечай ТОЛЬКО JSON-объектом без пояснений: "
    '{"title": "...", "deadline": "YYYY-MM-DDTHH:MM:SS+03:00" или null}. '
    "title: короткое название задачи до 80 символов, начинается с глагола в инфинитиве "
    "(например, \"Подготовить отчёт\"). deadline: срок в часовом поясе Europe/Moscow. "
    "Если указан только день без времени, ставь 18:00; утро = 10:00, вечер = 19:00. "
    "Срок не может быть раньше текущего момента. Если срока нет, deadline = null."
)


def _parse_llm(text: str, now: datetime) -> tuple[str | None, datetime | None, bool]:
    user_prompt = (
        f"Сейчас: {now.isoformat(timespec='seconds')} ({RU_WEEKDAYS[now.weekday()]}).\n"
        f"Сообщение:\n{text}"
    )
    data = _llm_json(PARSE_SYSTEM, user_prompt)
    if data is None:
        return None, None, False
    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        logger.warning("LLM: невалидный title: %r", title)
        return None, None, False
    title = re.sub(r"\s+", " ", title).strip(" .")
    if len(title) > TITLE_MAX_LEN:
        title = title[:TITLE_MAX_LEN].rsplit(" ", 1)[0]
    raw = data.get("deadline")
    if raw in (None, "", "null"):
        return title, None, True
    if not isinstance(raw, str):
        logger.warning("LLM: невалидный deadline: %r", raw)
        return None, None, False
    try:
        dt = _to_moscow(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        logger.warning("LLM: deadline не в ISO: %r", raw)
        return None, None, False
    if dt <= now:
        logger.warning("LLM: deadline в прошлом: %r", raw)
        return title, None, True
    return title, dt.replace(second=0, microsecond=0), True


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def parse_task(text: str, now: datetime) -> dict:
    try:
        now = _to_moscow(now)
        text = (text or "").strip()
        if not text:
            return {"title": "Выполнить задачу", "deadline": None, "priority": "normal"}
        title, deadline = _parse_rules(text, now)
        if os.getenv("LLM_API_KEY", "").strip():
            llm_title, llm_deadline, ok = _parse_llm(text, now)
            if ok:
                title = llm_title or title
                if deadline is None and llm_deadline is not None:
                    deadline = llm_deadline
        if deadline is not None and deadline <= now:
            deadline = None
        return {"title": title, "deadline": _iso(deadline), "priority": _priority(text, deadline, now)}
    except Exception:
        logger.exception("parse_task: непредвиденная ошибка")
        return {"title": "Выполнить задачу", "deadline": None, "priority": "normal"}


ANALYZE_SYSTEM = (
    "Ты помощник по задачам. По сообщению коллеги составь план и черновик ответа. "
    "Отвечай ТОЛЬКО JSON-объектом без пояснений: "
    '{"steps": ["шаг 1", "..."], "draft_reply": "..."}. '
    "steps: от 3 до 6 коротких конкретных шагов на русском, каждый начинается с глагола. "
    "draft_reply: короткий вежливый ответ коллеге на русском (2-3 предложения): "
    "подтверди, что задача принята, назови срок, если он есть, предложи уточнить детали."
)

STEP_TEMPLATES = [
    (r"отчет|отчёт|доклад|аналитик", [
        "Уточнить формат и ожидаемый объём отчёта",
        "Собрать нужные данные и материалы",
        "Подготовить черновик отчёта",
        "Проверить цифры и выводы",
        "Отправить отчёт коллеге",
    ]),
    (r"презентац|слайд", [
        "Уточнить цель и аудиторию презентации",
        "Набросать структуру слайдов",
        "Подготовить слайды с ключевыми тезисами",
        "Проверить оформление и опечатки",
        "Отправить презентацию",
    ]),
    (r"клиент|письм|ответ", [
        "Перечитать запрос и историю переписки",
        "Уточнить недостающие детали у коллег",
        "Подготовить текст ответа",
        "Отправить ответ и зафиксировать договорённости",
    ]),
    (r"созвон|встреч|звонок|митинг|созвонимся", [
        "Согласовать время и участников",
        "Подготовить повестку и вопросы",
        "Провести встречу",
        "Разослать итоги и следующие шаги",
    ]),
    (r"макет|дизайн|верстк", [
        "Открыть макет и изучить требования",
        "Отметить замечания и спорные места",
        "Сформулировать комментарии",
        "Отправить фидбек автору макета",
    ]),
    (r"код|баг|ошибк|фич|деплой|ревью|pr\b|пулл", [
        "Воспроизвести проблему или разобрать требования",
        "Внести изменения в код",
        "Проверить работу и прогнать тесты",
        "Отправить изменения на ревью",
    ]),
]

DEFAULT_STEPS = [
    "Уточнить у коллеги ожидаемый результат",
    "Разбить задачу на части и оценить время",
    "Выполнить основную часть работы",
    "Проверить результат и отправить коллеге",
]

RU_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
                 "августа", "сентября", "октября", "ноября", "декабря"]


def _human_deadline(dt: datetime, now: datetime) -> str:
    hm = dt.strftime("%H:%M")
    if dt.date() == now.date():
        return f"сегодня до {hm}"
    if dt.date() == (now + timedelta(days=1)).date():
        return f"завтра до {hm}"
    return f"{dt.day} {RU_MONTHS_GEN[dt.month - 1]} до {hm}"


def _analyze_fallback(text: str) -> dict:
    now = datetime.now(TZ)
    title, deadline = _parse_rules(text, now)
    s = text.lower()
    steps = DEFAULT_STEPS
    for pattern, template in STEP_TEMPLATES:
        if re.search(pattern, s):
            steps = template
            break
    title_lc = title[0].lower() + title[1:] if title else "задачу"
    if deadline is not None:
        when = f"Постараюсь успеть {_human_deadline(deadline, now)}."
    else:
        when = "Сориентирую по срокам, как только оценю объём."
    draft = (
        f"Привет! Принял задачу: {title_lc}. {when} "
        f"Если есть дополнительные пожелания или детали, напиши, пожалуйста."
    )
    return {"steps": list(steps), "draft_reply": draft}


def _validate_analysis(data: dict | None) -> dict | None:
    if not isinstance(data, dict):
        return None
    steps = data.get("steps")
    draft = data.get("draft_reply")
    if not isinstance(steps, list) or not isinstance(draft, str) or not draft.strip():
        return None
    steps = [re.sub(r"\s+", " ", s).strip(" .-•") for s in steps if isinstance(s, str) and s.strip()]
    steps = [re.sub(r"^\d+[.)]\s*", "", s) for s in steps]
    if len(steps) < 3:
        return None
    return {"steps": steps[:6], "draft_reply": draft.strip()}


def analyze_task(text: str) -> dict:
    try:
        text = (text or "").strip()
        if not text:
            return {
                "steps": DEFAULT_STEPS.copy(),
                "draft_reply": "Привет! Задачу получил, уточни, пожалуйста, детали.",
            }
        if os.getenv("LLM_API_KEY", "").strip():
            now = datetime.now(TZ)
            user_prompt = (
                f"Сейчас: {now.isoformat(timespec='seconds')} ({RU_WEEKDAYS[now.weekday()]}).\n"
                f"Сообщение коллеги:\n{text}"
            )
            result = _validate_analysis(_llm_json(ANALYZE_SYSTEM, user_prompt))
            if result:
                return result
            logger.warning("analyze_task: ответ LLM невалиден, использую шаблоны")
        return _analyze_fallback(text)
    except Exception:
        logger.exception("analyze_task: непредвиденная ошибка")
        return {
            "steps": DEFAULT_STEPS.copy(),
            "draft_reply": "Привет! Задачу получил, скоро вернусь с ответом.",
        }