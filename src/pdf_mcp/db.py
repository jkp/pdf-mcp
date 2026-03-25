"""SQLite database layer for PDF indexing and full-text search.

Stores PDF metadata, per-page extracted text, and an FTS5 index
for full-text search across all indexed PDFs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pdfs (
    filename    TEXT PRIMARY KEY,
    file_hash   TEXT NOT NULL,
    page_count  INTEGER,
    title       TEXT,
    author      TEXT,
    indexed_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS pdf_pages (
    rowid       INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL REFERENCES pdfs(filename) ON DELETE CASCADE,
    page_num    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    UNIQUE(filename, page_num)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_pdfs USING fts5(
    filename UNINDEXED,
    page_num UNINDEXED,
    text,
    content='pdf_pages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pdf_pages BEGIN
    INSERT INTO fts_pdfs(rowid, filename, page_num, text)
    VALUES (new.rowid, new.filename, new.page_num, new.text);
END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pdf_pages BEGIN
    INSERT INTO fts_pdfs(fts_pdfs, rowid, filename, page_num, text)
    VALUES ('delete', old.rowid, old.filename, old.page_num, old.text);
END;
CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pdf_pages BEGIN
    INSERT INTO fts_pdfs(fts_pdfs, rowid, filename, page_num, text)
    VALUES ('delete', old.rowid, old.filename, old.page_num, old.text);
    INSERT INTO fts_pdfs(rowid, filename, page_num, text)
    VALUES (new.rowid, new.filename, new.page_num, new.text);
END;
"""


class Database:
    """SQLite database for PDF metadata and full-text search."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def execute(self, sql: str, params: list[Any] | None = None) -> sqlite3.Cursor:
        return self._conn.execute(sql, params or [])

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── PDFs ──────────────────────────────────────────────────────────────────

    def upsert_pdf(
        self,
        filename: str,
        file_hash: str,
        page_count: int | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO pdfs (filename, file_hash, page_count, title, author)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                file_hash = excluded.file_hash,
                page_count = excluded.page_count,
                title = excluded.title,
                author = excluded.author,
                indexed_at = unixepoch()
            """,
            [filename, file_hash, page_count, title, author],
        )
        self._conn.commit()

    def get_pdf(self, filename: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM pdfs WHERE filename = ?", [filename]
        ).fetchone()
        return dict(row) if row else None

    def list_pdfs(self, pattern: str | None = None) -> list[dict[str, Any]]:
        if pattern:
            rows = self._conn.execute(
                "SELECT * FROM pdfs WHERE filename LIKE ? ORDER BY filename",
                [f"%{pattern}%"],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pdfs ORDER BY filename"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_pdf(self, filename: str) -> None:
        self._conn.execute("DELETE FROM pdfs WHERE filename = ?", [filename])
        self._conn.commit()

    # ── Pages ─────────────────────────────────────────────────────────────────

    def insert_pages(self, filename: str, pages: list[tuple[int, str]]) -> None:
        """Insert or replace page text for a PDF."""
        # Delete existing pages first (triggers FTS cleanup)
        self._conn.execute("DELETE FROM pdf_pages WHERE filename = ?", [filename])
        for page_num, text in pages:
            self._conn.execute(
                "INSERT INTO pdf_pages (filename, page_num, text) VALUES (?, ?, ?)",
                [filename, page_num, text],
            )
        self._conn.commit()

    def get_pages(self, filename: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT page_num, text FROM pdf_pages WHERE filename = ? ORDER BY page_num",
            [filename],
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across all indexed PDFs.

        Returns list of {filename, page_num, snippet} sorted by relevance.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT filename, page_num, snippet(fts_pdfs, 2, '**', '**', '...', 30) as snippet
                FROM fts_pdfs
                WHERE fts_pdfs MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                [query, limit],
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []
