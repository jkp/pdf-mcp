"""Tests for PDF embedding pipeline and vector search."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pymupdf
import pytest

from pdf_mcp.db import Database
from pdf_mcp.embedder import Embedder
from pdf_mcp.indexer import Indexer


@pytest.fixture
def db(db_path: Path) -> Database:
    return Database(db_path)


@pytest.fixture
def mock_model():
    """Mock sentence transformer returning deterministic vectors."""
    from pdf_mcp.embedder import EMBEDDING_DIMS

    model = MagicMock()

    def _encode(texts, batch_size=64, show_progress_bar=True):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            v = rng.randn(EMBEDDING_DIMS).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    model.encode = _encode
    return model


@pytest.fixture
def embedder(db: Database, mock_model) -> Embedder:
    return Embedder(db=db, model=mock_model)


def _make_pdf(vault: Path, name: str, pages: list[str]) -> Path:
    path = vault / name
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    doc.save(str(path))
    doc.close()
    return path


class TestEmbedBatch:
    def test_embeds_indexed_pages(self, embedder: Embedder, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc", page_count=2)
        db.insert_pages("test.pdf", [(1, "Hello world"), (2, "Goodbye world")])

        count = embedder.embed_batch(["test.pdf"])
        assert count == 1

        row = db.get_pdf("test.pdf")
        assert row is not None

    def test_skips_already_embedded(self, embedder: Embedder, db: Database) -> None:
        db.upsert_pdf("test.pdf", file_hash="abc", page_count=1)
        db.insert_pages("test.pdf", [(1, "Content")])

        embedder.embed_batch(["test.pdf"])
        # Embed again — should skip
        count = embedder.embed_batch(["test.pdf"])
        assert count == 0

    def test_skips_missing_pages(self, embedder: Embedder, db: Database) -> None:
        db.upsert_pdf("empty.pdf", file_hash="abc", page_count=0)
        count = embedder.embed_batch(["empty.pdf"])
        assert count == 0


class TestVectorSearch:
    def test_finds_embedded_pdfs(self, embedder: Embedder, db: Database, monkeypatch) -> None:
        db.upsert_pdf("a.pdf", file_hash="1", page_count=1)
        db.insert_pages("a.pdf", [(1, "Invoice for consulting services")])
        db.upsert_pdf("b.pdf", file_hash="2", page_count=1)
        db.insert_pages("b.pdf", [(1, "Employment contract agreement")])

        embedder.embed_batch(["a.pdf", "b.pdf"])

        results = embedder.search("invoice", limit=5)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert "filename" in results[0]
        assert "page_num" in results[0]

    def test_returns_empty_for_no_vectors(self, embedder: Embedder) -> None:
        results = embedder.search("anything", limit=5)
        assert results == []

    def test_respects_limit(self, embedder: Embedder, db: Database) -> None:
        for i in range(10):
            db.upsert_pdf(f"doc-{i}.pdf", file_hash=str(i), page_count=1)
            db.insert_pages(f"doc-{i}.pdf", [(1, f"Document number {i} content")])
        embedder.embed_batch([f"doc-{i}.pdf" for i in range(10)])

        results = embedder.search("document", limit=3)
        assert len(results) <= 3


class TestGetUnembedded:
    def test_returns_unembedded(self, embedder: Embedder, db: Database) -> None:
        db.upsert_pdf("a.pdf", file_hash="1", page_count=1)
        db.insert_pages("a.pdf", [(1, "Text")])
        db.upsert_pdf("b.pdf", file_hash="2", page_count=1)
        db.insert_pages("b.pdf", [(1, "Text")])

        unembedded = embedder.get_unembedded()
        assert set(unembedded) == {"a.pdf", "b.pdf"}

    def test_excludes_already_embedded(self, embedder: Embedder, db: Database) -> None:
        db.upsert_pdf("a.pdf", file_hash="1", page_count=1)
        db.insert_pages("a.pdf", [(1, "Text")])
        embedder.embed_batch(["a.pdf"])

        db.upsert_pdf("b.pdf", file_hash="2", page_count=1)
        db.insert_pages("b.pdf", [(1, "Text")])

        unembedded = embedder.get_unembedded()
        assert unembedded == ["b.pdf"]


class TestSearchIntegration:
    def test_fts_and_vector_both_work(
        self, embedder: Embedder, db: Database, vault: Path
    ) -> None:
        """Both FTS5 and vector search should find indexed+embedded PDFs."""
        indexer = Indexer(db=db, vault=vault)
        _make_pdf(vault, "invoice.pdf", ["Invoice #999 for web development services"])
        indexer.index_file("invoice.pdf")
        embedder.embed_batch(["invoice.pdf"])

        # FTS5 search
        fts_results = db.search("invoice")
        assert len(fts_results) >= 1

        # Vector search
        vec_results = embedder.search("invoice", limit=5)
        assert len(vec_results) >= 1
