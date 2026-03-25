"""PDF reading tools."""

from typing import Any

import pymupdf
import structlog

from pdf_mcp.server import db, indexer, mcp, settings

logger = structlog.get_logger()


@mcp.tool(annotations={"readOnlyHint": True, "title": "Read PDF"})
async def read_pdf(
    filename: str,
    pages: str | None = None,
) -> dict[str, Any]:
    """Extract text from a PDF, optionally specific pages.

    Args:
        filename: PDF filename (e.g., "2024-01-15 Invoice.pdf")
        pages: Optional page range (e.g., "1-3", "5", "1,3,5"). Omit for all pages.
    """
    logger.info("tool.read_pdf", filename=filename, pages=pages)

    path = settings.vault / filename
    if not path.is_file():
        return {"error": f"PDF not found: {filename}"}

    doc = pymupdf.open(str(path))
    page_indices = _parse_pages(pages, len(doc)) if pages else range(len(doc))

    result_pages = []
    for i in page_indices:
        if 0 <= i < len(doc):
            text = doc[i].get_text("text").strip()
            result_pages.append({"page": i + 1, "text": text})

    metadata = doc.metadata or {}
    page_count = len(doc)
    doc.close()

    return {
        "filename": filename,
        "page_count": page_count,
        "title": metadata.get("title") or None,
        "pages": result_pages,
    }


@mcp.tool(annotations={"readOnlyHint": True, "title": "Get PDF Info"})
async def get_pdf_info(filename: str) -> dict[str, Any]:
    """Get metadata about a PDF without extracting full text.

    Args:
        filename: PDF filename
    """
    logger.info("tool.get_pdf_info", filename=filename)

    path = settings.vault / filename
    if not path.is_file():
        return {"error": f"PDF not found: {filename}"}

    doc = pymupdf.open(str(path))
    metadata = doc.metadata or {}
    page_count = len(doc)
    file_size = path.stat().st_size
    doc.close()

    return {
        "filename": filename,
        "page_count": page_count,
        "file_size_bytes": file_size,
        "title": metadata.get("title") or None,
        "author": metadata.get("author") or None,
        "subject": metadata.get("subject") or None,
        "creator": metadata.get("creator") or None,
        "creation_date": metadata.get("creationDate") or None,
    }


def _parse_pages(spec: str, total: int) -> list[int]:
    """Parse a page spec like '1-3,5,7' into zero-based indices."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            s = int(start) - 1
            e = int(end)
            indices.extend(range(max(0, s), min(e, total)))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.append(idx)
    return indices
