"""PDF search tools."""

from typing import Any

import structlog

from pdf_mcp.server import db, mcp

logger = structlog.get_logger()


@mcp.tool(annotations={"readOnlyHint": True, "title": "Search PDFs"})
async def search_pdfs(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Full-text search across all indexed PDFs.

    Searches the extracted text of all PDFs using FTS5.
    Returns matching pages with highlighted snippets.

    Args:
        query: Search query (supports FTS5 syntax: AND, OR, NOT, "phrases")
        limit: Maximum number of results
    """
    logger.info("tool.search_pdfs", query=query, limit=limit)
    results = db.search(query, limit=limit)
    logger.info("tool.search_pdfs.done", query=query, count=len(results))
    return results


@mcp.tool(annotations={"readOnlyHint": True, "title": "List PDFs"})
async def list_pdfs(
    pattern: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List PDFs in the filing cabinet, optionally filtered by filename pattern.

    Args:
        pattern: Optional filename substring to filter by (e.g., "mortgage", "2024")
        limit: Maximum number of results
    """
    logger.info("tool.list_pdfs", pattern=pattern)
    results = db.list_pdfs(pattern=pattern)
    return results[:limit]
