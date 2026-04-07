"""PDF embedding pipeline for semantic vector search.

Embeds per-page text from indexed PDFs into sqlite-vec for similarity
search. Long pages are chunked with overlap.

Same model and approach as the email MCP server.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np
import structlog

from pdf_mcp.db import Database

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 64
_CHUNK_CHARS = 400
_CHUNK_OVERLAP = 100
_MAX_CHUNKS_PER_PAGE = 5

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIMS = 1024
_QUERY_PREFIX = "query: "
_DOC_PREFIX = "passage: "
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_API_BATCH_SIZE = 100


def _serialize_f32(vector: np.ndarray) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _make_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if len(text) <= _CHUNK_CHARS:
        return [f"{_DOC_PREFIX}{text}"]

    chunks = []
    step = _CHUNK_CHARS - _CHUNK_OVERLAP
    pos = 0
    while pos < len(text):
        chunk = text[pos : pos + _CHUNK_CHARS]
        chunks.append(f"{_DOC_PREFIX}{chunk}")
        pos += step
        if pos + _CHUNK_OVERLAP >= len(text):
            break

    return chunks[:_MAX_CHUNKS_PER_PAGE] or [f"{_DOC_PREFIX}{text}"]


class Embedder:
    """Embed PDF page content and search by vector similarity."""

    def __init__(
        self,
        db: Database,
        model: Any = None,
        model_name: str = MODEL_NAME,
        api_key: str = "",
    ) -> None:
        self._db = db
        self._model_name = model_name
        self._together_key = api_key
        self._local_model = model
        self._reranker = None
        self._ensure_table()

    @staticmethod
    def _load_local_model(model_name: str) -> Any:
        import logging

        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.encode(["warmup"], show_progress_bar=False)
        return model

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        if self._local_model is None:
            logger.info("embedder.loading_local_model")
            self._local_model = self._load_local_model(self._model_name)
        return self._local_model.encode(texts, batch_size=_BATCH_SIZE, show_progress_bar=False)

    def _encode_via_api(self, texts: list[str]) -> np.ndarray:
        import httpx

        all_vectors = []
        for i in range(0, len(texts), _API_BATCH_SIZE):
            batch = texts[i : i + _API_BATCH_SIZE]
            resp = httpx.post(
                "https://api.together.xyz/v1/embeddings",
                headers={"Authorization": f"Bearer {self._together_key}"},
                json={"model": self._model_name, "input": batch},
                timeout=60,
            )
            if resp.status_code != 200:
                detail = resp.json().get("error", {}).get("message", resp.text[:200])
                raise RuntimeError(f"Together API: {detail}")
            data = resp.json()
            all_vectors.append(
                np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
            )
        return np.concatenate(all_vectors)

    def _ensure_table(self) -> None:
        import sqlite_vec

        self._db._conn.enable_load_extension(True)
        sqlite_vec.load(self._db._conn)
        self._db._conn.enable_load_extension(False)

        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS pdf_vectors"
            f" USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{EMBEDDING_DIMS}])"
        )

        # Add embedded column if missing
        existing = {
            row[1] for row in self._db.execute("PRAGMA table_info(pdfs)").fetchall()
        }
        if "embedded" not in existing:
            self._db.execute(
                "ALTER TABLE pdfs ADD COLUMN embedded INTEGER NOT NULL DEFAULT 0"
            )
            self._db.commit()

    def embed_batch(self, filenames: list[str], use_api: bool = False) -> int:
        """Embed pages for a batch of PDFs. Returns count of newly embedded."""
        all_texts = []
        all_chunk_ids = []
        embedded_files: set[str] = set()

        for filename in filenames:
            # Skip already embedded
            pdf = self._db.get_pdf(filename)
            if not pdf:
                continue
            if pdf.get("embedded"):
                continue

            pages = self._db.get_pages(filename)
            if not pages:
                continue

            for page in pages:
                text = f"File: {filename}\nPage {page['page_num']}\n\n{page['text']}"
                chunks = _make_chunks(text)
                for i, chunk_text in enumerate(chunks):
                    all_texts.append(chunk_text)
                    all_chunk_ids.append(f"{filename}:{page['page_num']}:{i}")

            embedded_files.add(filename)

        if not all_texts:
            return 0

        # Encode
        if use_api and self._together_key:
            try:
                vectors = self._encode_via_api(all_texts)
            except Exception as e:
                logger.warning("embedder.api_failed", error=str(e), chunks=len(all_texts))
                return 0
        else:
            vectors = self._encode_local(all_texts)

        # Store vectors
        for chunk_id, vec in zip(all_chunk_ids, vectors):
            vec_f32 = np.asarray(vec, dtype=np.float32)
            self._db.execute(
                "INSERT OR REPLACE INTO pdf_vectors (chunk_id, embedding) VALUES (?, ?)",
                [chunk_id, _serialize_f32(vec_f32)],
            )

        # Mark as embedded
        for filename in embedded_files:
            self._db.execute(
                "UPDATE pdfs SET embedded = 1 WHERE filename = ?", [filename]
            )
        self._db.commit()

        logger.info("embedder.embedded", files=len(embedded_files), chunks=len(all_texts))
        return len(embedded_files)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {filename, page_num, distance}."""
        vec = self._encode_local([f"{_QUERY_PREFIX}{query}"])
        query_vec = np.asarray(vec[0], dtype=np.float32)

        rows = self._db.execute(
            "SELECT chunk_id, distance FROM pdf_vectors"
            " WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance",
            [_serialize_f32(query_vec), limit * 3],
        ).fetchall()

        if not rows:
            return []

        # Dedup to filename+page_num, keep best distance
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for chunk_id, dist in rows:
            parts = chunk_id.rsplit(":", 2)
            key = f"{parts[0]}:{parts[1]}"  # filename:page_num
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "filename": parts[0],
                    "page_num": int(parts[1]),
                    "distance": round(float(dist), 4),
                }
            )
            if len(results) >= limit:
                break

        return results

    def _ensure_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("embedder.loading_reranker")
            self._reranker = CrossEncoder(_RERANKER_MODEL)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[tuple[float, dict[str, Any]]]:
        """Score candidates using cross-encoder. Returns (score, result) sorted desc."""
        if not candidates:
            return []
        self._ensure_reranker()
        assert self._reranker is not None

        pairs = []
        for c in candidates:
            doc = f"File: {c['filename']}\nPage {c['page_num']}\n\n{c.get('snippet', '')}"
            pairs.append([query, doc])

        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        logger.info(
            "embedder.reranked",
            count=len(ranked),
            top_score=f"{ranked[0][0]:.2f}" if ranked else None,
        )
        return ranked

    def get_unembedded(self, limit: int = 1000) -> list[str]:
        """Get filenames that have pages but aren't embedded yet."""
        rows = self._db.execute(
            "SELECT p.filename FROM pdfs p"
            " WHERE p.embedded = 0"
            " AND EXISTS (SELECT 1 FROM pdf_pages pp WHERE pp.filename = p.filename)"
            " ORDER BY p.filename"
            " LIMIT ?",
            [limit],
        ).fetchall()
        return [r[0] for r in rows]
