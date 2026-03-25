"""Shared test fixtures for pdf-mcp tests."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a temporary vault with sample PDFs."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture
def sample_pdf(vault: Path) -> Path:
    """Create a minimal valid PDF in the vault."""
    import pymupdf

    path = vault / "2024-01-15 Test Invoice.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Invoice #12345\nAmount: £500.00\nDue: 2024-02-15")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"
