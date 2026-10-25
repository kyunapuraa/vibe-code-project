import json
import logging
import os

import numpy as np

logger = logging.getLogger("ml.embeddings")

MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = 384
BATCH_SIZE = 32

__all__ = ["embed", "embed_chunks", "embed_query", "search", "search_top", "EMBED_DIM"]

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Загружаю модель эмбеддингов %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
        logger.info("Модель загружена, размерность %d", (getattr(_model, "get_embedding_dimension", None) or _model.get_sentence_embedding_dimension)())
    return _model


def _with_prefix(texts: list[str], kind: str) -> list[str]:
    if "e5" not in MODEL_NAME.lower():
        return texts
    prefix = "query: " if kind == "query" else "passage: "
    return [prefix + t for t in texts]


def _encode(texts: list[str], kind: str) -> np.ndarray:
    clean = [" ".join((t or "").split()) for t in texts]
    result = np.zeros((len(clean), EMBED_DIM), dtype=np.float32)
    idx = [i for i, t in enumerate(clean) if t]
    if not idx:
        return result
    vectors = _get_model().encode(
        _with_prefix([clean[i] for i in idx], kind),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    result[idx] = vectors.astype(np.float32)
    return result


def embed(text: str) -> list[float]:
    try:
        return _encode([text], "passage")[0].tolist()
    except Exception:
        logger.exception("embed: не удалось получить эмбеддинг")
        return [0.0] * EMBED_DIM


def embed_query(text: str) -> list[float]:
    try:
        return _encode([text], "query")[0].tolist()
    except Exception:
        logger.exception("embed_query: не удалось получить эмбеддинг")
        return [0.0] * EMBED_DIM


def embed_chunks(chunks: list) -> list[dict]:
    try:
        records = []
        for chunk in chunks or []:
            if isinstance(chunk, dict):
                record = dict(chunk)
                record["text"] = str(record.get("text") or "")
            else:
                record = {"text": str(chunk or "")}
            records.append(record)
        if not records:
            return []
        vectors = _encode([r["text"] for r in records], "passage")
        for record, vector in zip(records, vectors):
            record["embedding"] = vector.tolist()
        return records
    except Exception:
        logger.exception("embed_chunks: не удалось получить эмбеддинги")
        return []


def _to_vector(raw) -> np.ndarray | None:
    try:
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray, memoryview)):
            vector = np.frombuffer(bytes(raw), dtype=np.float32)
        elif isinstance(raw, str):
            vector = np.asarray(json.loads(raw), dtype=np.float32)
        else:
            vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != EMBED_DIM or not np.all(np.isfinite(vector)):
            return None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm
    except Exception:
        return None


def _unpack(chunk) -> tuple:
    if isinstance(chunk, dict):
        return chunk.get("id"), chunk.get("text", ""), chunk.get("embedding")
    if isinstance(chunk, (list, tuple)):
        if len(chunk) >= 3:
            return chunk[0], chunk[1], chunk[2]
        if len(chunk) == 2:
            return chunk[0], "", chunk[1]
    return None, "", None


def search_top(query: str, chunks: list, k: int = 3) -> list[dict]:
    try:
        if not query or not str(query).strip():
            return []
        ids, texts, vectors = [], [], []
        for chunk in chunks or []:
            chunk_id, text, raw = _unpack(chunk)
            vector = _to_vector(raw)
            if vector is None:
                logger.warning("search: пропускаю кусок %r с некорректным эмбеддингом", chunk_id)
                continue
            ids.append(chunk_id)
            texts.append(text or "")
            vectors.append(vector)
        if not vectors:
            return []
        query_vector = _encode([str(query)], "query")[0]
        scores = np.vstack(vectors) @ query_vector
        order = np.argsort(-scores)[: max(1, int(k))]
        return [{"id": ids[i], "text": texts[i], "score": round(float(scores[i]), 4)} for i in order]
    except Exception:
        logger.exception("search: ошибка поиска")
        return []


def search(query: str, chunks: list) -> str:
    top = search_top(query, chunks, k=1)
    return top[0]["text"] if top else ""