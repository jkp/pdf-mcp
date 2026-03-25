"""PDF text extraction and indexing pipeline.

Scans a vault directory for PDFs, extracts text per page using pymupdf,
and stores in SQLite FTS5 for full-text search.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf
import structlog

from pdf_mcp.db import Database

logger = structlog.get_logger(__name__)


class Indexer:
    """Extract text from PDFs and index into SQLite FTS5."""

    def __init__(self, db: Database, vault: Path) -> None:
        self._db = db
        self._vault = vault

    def index_file(self, filename: str) -> bool:
        """Index a single PDF. Returns True if indexed, False if skipped.

        Raises FileNotFoundError if the file doesn't exist.
        """
        path = self._vault / filename
        if not path.is_file():
            msg = f"PDF not found: {filename}"
            raise FileNotFoundError(msg)

        file_hash = self._hash_file(path)

        # Skip if already indexed with same hash
        existing = self._db.get_pdf(filename)
        if existing and existing["file_hash"] == file_hash:
            return False

        # Extract text
        doc = pymupdf.open(str(path))
        pages: list[tuple[int, str]] = []
        title = doc.metadata.get("title") if doc.metadata else None
        author = doc.metadata.get("author") if doc.metadata else None

        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append((i + 1, text))
        page_count = len(doc)
        doc.close()

        # Store
        self._db.upsert_pdf(
            filename=filename,
            file_hash=file_hash,
            page_count=page_count,
            title=title or None,
            author=author or None,
        )
        self._db.insert_pages(filename, pages)

        logger.info("indexer.indexed", filename=filename, pages=page_count)
        return True

    def index_all(self) -> dict[str, int]:
        """Index all PDFs in the vault. Returns stats.

        - Indexes new/changed PDFs
        - Skips unchanged PDFs
        - Removes PDFs that no longer exist on disk
        """
        indexed = 0
        skipped = 0
        failed = 0
        removed = 0

        # Find all PDFs in vault
        disk_files = {f.name for f in self._vault.glob("*.pdf")}

        # Remove DB entries for deleted files
        db_files = {p["filename"] for p in self._db.list_pdfs()}
        for gone in db_files - disk_files:
            self._db.delete_pdf(gone)
            removed += 1
            logger.info("indexer.removed", filename=gone)

        # Index new/changed
        for filename in sorted(disk_files):
            try:
                if self.index_file(filename):
                    indexed += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                logger.warning("indexer.failed", filename=filename, error=str(e))

        logger.info(
            "indexer.done",
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            removed=removed,
        )
        return {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "removed": removed,
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA256 hash of file contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
