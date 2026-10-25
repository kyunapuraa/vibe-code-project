import logging
import os

import embeddings
import extract as ml1

logger = logging.getLogger("ml.knowledge")

SUPPORTED_EXT = (".pdf", ".docx", ".txt", ".md", ".jpg", ".jpeg", ".png")

__all__ = ["SUPPORTED_EXT", "chunk_file", "chunk_text", "index_file", "index_text", "find", "warmup"]


def chunk_file(path: str, source: str | None = None) -> list[dict]:
    chunks = ml1.extract(path)
    if source:
        for chunk in chunks:
            chunk["source"] = source
    return chunks


def chunk_text(text: str, source: str = "Заметка") -> list[dict]:
    cleaned = ml1.clean(text or "")
    if not cleaned:
        return []
    parts = [p for p in ml1.merge_small(ml1.split_text(cleaned)) if len(p) > 40]
    if not parts:
        parts = [cleaned]
    return [{"text": p, "source": source, "page": 1} for p in parts]


def index_file(path: str, source: str | None = None) -> list[dict]:
    chunks = chunk_file(path, source or os.path.basename(path))
    logger.info("Файл %s: %d кусков", source or path, len(chunks))
    return embeddings.embed_chunks(chunks)


def index_text(text: str, source: str = "Заметка") -> list[dict]:
    chunks = chunk_text(text, source)
    logger.info("Текст %s: %d кусков", source, len(chunks))
    return embeddings.embed_chunks(chunks)


def find(query: str, rows: list, k: int = 3) -> list[dict]:
    top = embeddings.search_top(query, rows, k=k)
    meta = {}
    for row in rows or []:
        if isinstance(row, dict):
            meta[row.get("id")] = row
    for item in top:
        row = meta.get(item["id"], {})
        item["source"] = row.get("source")
        item["page"] = row.get("page")
        item["document_id"] = row.get("document_id")
    return top


def warmup() -> None:
    try:
        embeddings.embed_query("прогрев модели")
        logger.info("Модель эмбеддингов прогрета")
    except Exception:
        logger.exception("Не удалось прогреть модель эмбеддингов")
