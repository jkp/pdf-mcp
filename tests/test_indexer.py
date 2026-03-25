"""Tests for the PDF indexer."""

import hashlib
from pathlib import Path

import pymupdf
import pytest

from pdf_mcp.db import Database
from pdf_mcp.indexer import Indexer


@pytest.fixture
def db(db_path: Path) -> Database:
    return Database(db_path)


@pytest.fixture
def indexer(db: Database, vault: Path) -> Indexer:
    return Indexer(db=db, vault=vault)


def _make_pdf(vault: Path, name: str, pages: list[str]) -> Path:
    """Create a PDF with the given text on each page."""
    path = vault / name
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    doc.save(str(path))
    doc.close()
    return path


class TestIndexOne:
    def test_indexes_pdf(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "test.pdf", ["Hello world"])
        indexer.index_file("test.pdf")

        pdf = db.get_pdf("test.pdf")
        assert pdf is not None
        assert pdf["page_count"] == 1

        pages = db.get_pages("test.pdf")
        assert len(pages) == 1
        assert "Hello world" in pages[0]["text"]

    def test_extracts_multiple_pages(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "multi.pdf", ["Page one", "Page two", "Page three"])
        indexer.index_file("multi.pdf")

        pdf = db.get_pdf("multi.pdf")
        assert pdf["page_count"] == 3

        pages = db.get_pages("multi.pdf")
        assert len(pages) == 3

    def test_skips_if_hash_unchanged(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "test.pdf", ["Content"])
        indexer.index_file("test.pdf")
        first_indexed = db.get_pdf("test.pdf")["indexed_at"]

        # Index again — should skip
        indexer.index_file("test.pdf")
        second_indexed = db.get_pdf("test.pdf")["indexed_at"]
        assert first_indexed == second_indexed

    def test_reindexes_if_hash_changed(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "test.pdf", ["Old content"])
        indexer.index_file("test.pdf")

        # Recreate with different content
        _make_pdf(vault, "test.pdf", ["New content"])
        indexer.index_file("test.pdf")

        pages = db.get_pages("test.pdf")
        assert "New content" in pages[0]["text"]

    def test_nonexistent_file_raises(self, indexer: Indexer) -> None:
        with pytest.raises(FileNotFoundError):
            indexer.index_file("missing.pdf")


class TestIndexAll:
    def test_indexes_all_pdfs_in_vault(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "a.pdf", ["Alpha"])
        _make_pdf(vault, "b.pdf", ["Beta"])
        _make_pdf(vault, "c.pdf", ["Gamma"])

        stats = indexer.index_all()
        assert stats["indexed"] == 3
        assert stats["skipped"] == 0
        assert len(db.list_pdfs()) == 3

    def test_skips_already_indexed(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "a.pdf", ["Alpha"])
        indexer.index_all()

        _make_pdf(vault, "b.pdf", ["Beta"])
        stats = indexer.index_all()
        assert stats["indexed"] == 1
        assert stats["skipped"] == 1

    def test_removes_deleted_pdfs(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "keep.pdf", ["Keep"])
        _make_pdf(vault, "delete.pdf", ["Delete"])
        indexer.index_all()
        assert len(db.list_pdfs()) == 2

        (vault / "delete.pdf").unlink()
        stats = indexer.index_all()
        assert stats["removed"] == 1
        assert len(db.list_pdfs()) == 1

    def test_handles_empty_vault(self, indexer: Indexer) -> None:
        stats = indexer.index_all()
        assert stats["indexed"] == 0


class TestSearchIntegration:
    def test_indexed_pdfs_are_searchable(self, indexer: Indexer, db: Database, vault: Path) -> None:
        _make_pdf(vault, "invoice.pdf", ["Invoice #12345 for consulting services rendered"])
        _make_pdf(vault, "contract.pdf", ["Employment agreement between parties"])
        indexer.index_all()

        results = db.search("invoice consulting")
        assert len(results) >= 1
        assert results[0]["filename"] == "invoice.pdf"
