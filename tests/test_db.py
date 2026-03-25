"""Tests for the SQLite database layer."""

from pathlib import Path

import pytest

from pdf_mcp.db import Database


@pytest.fixture
def db(db_path: Path) -> Database:
    return Database(db_path)


class TestSchema:
    def test_tables_created(self, db: Database) -> None:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"pdfs", "pdf_pages"} <= tables

    def test_fts_table_created(self, db: Database) -> None:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_pdfs'"
        ).fetchall()
        assert rows

    def test_migrations_idempotent(self, db_path: Path) -> None:
        Database(db_path)
        Database(db_path)  # second open should not raise


class TestPdfs:
    def test_upsert_and_get(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc123", page_count=3, title="Test Doc")
        pdf = db.get_pdf("test.pdf")
        assert pdf is not None
        assert pdf["filename"] == "test.pdf"
        assert pdf["file_hash"] == "abc123"
        assert pdf["page_count"] == 3
        assert pdf["title"] == "Test Doc"

    def test_upsert_updates_existing(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="old")
        db.upsert_pdf("test.pdf", file_hash="new", page_count=5)
        pdf = db.get_pdf("test.pdf")
        assert pdf["file_hash"] == "new"
        assert pdf["page_count"] == 5

    def test_list_all(self, db: Database) -> None:
        db.upsert_pdf("a.pdf", file_hash="1")
        db.upsert_pdf("b.pdf", file_hash="2")
        db.upsert_pdf("c.pdf", file_hash="3")
        pdfs = db.list_pdfs()
        assert len(pdfs) == 3

    def test_delete(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.delete_pdf("test.pdf")
        assert db.get_pdf("test.pdf") is None


class TestPages:
    def test_insert_and_get_pages(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.insert_pages("test.pdf", [(1, "Page one text"), (2, "Page two text")])
        pages = db.get_pages("test.pdf")
        assert len(pages) == 2
        assert pages[0]["text"] == "Page one text"
        assert pages[1]["text"] == "Page two text"

    def test_replace_pages_on_reindex(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.insert_pages("test.pdf", [(1, "Old text")])
        db.insert_pages("test.pdf", [(1, "New text")])
        pages = db.get_pages("test.pdf")
        assert len(pages) == 1
        assert pages[0]["text"] == "New text"

    def test_cascade_delete(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.insert_pages("test.pdf", [(1, "Some text")])
        db.delete_pdf("test.pdf")
        pages = db.get_pages("test.pdf")
        assert pages == []


class TestSearch:
    def test_fts_search(self, db: Database) -> None:
        db.upsert_pdf("invoice.pdf", file_hash="abc")
        db.insert_pages("invoice.pdf", [(1, "Invoice #12345 for consulting services")])
        db.upsert_pdf("contract.pdf", file_hash="def")
        db.insert_pages("contract.pdf", [(1, "Employment contract terms and conditions")])

        results = db.search("invoice consulting")
        assert len(results) >= 1
        assert results[0]["filename"] == "invoice.pdf"

    def test_fts_no_match(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.insert_pages("test.pdf", [(1, "Some content")])
        results = db.search("xyznonexistent")
        assert results == []

    def test_fts_returns_page_info(self, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc")
        db.insert_pages("test.pdf", [(1, "First page"), (2, "Second page with keyword")])
        results = db.search("keyword")
        assert len(results) == 1
        assert results[0]["page_num"] == 2
