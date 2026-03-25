"""PDF search tools — FTS5 + semantic vector search."""

from typing import Any

import structlog

from pdf_mcp.server import db, embedder, mcp

logger = structlog.get_logger()


@mcp.tool(annotations={"readOnlyHint": True, "title": "Search PDFs"})
async def search_pdfs(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search across all indexed PDFs using full-text and semantic search.

    Combines FTS5 keyword matching with vector similarity search for
    best results. Returns matching pages with snippets.

    Args:
        query: Search query (natural language or keywords)
        limit: Maximum number of results
    """
    logger.info("tool.search_pdfs", query=query, limit=limit)

    seen: set[str] = set()  # filename:page_num dedup key
    results: list[dict[str, Any]] = []

    # Phase 1: Vector search (semantic)
    if embedder:
        try:
            vec_results = embedder.search(query, limit=limit)
            for r in vec_results:
                key = f"{r['filename']}:{r['page_num']}"
                if key not in seen:
                    seen.add(key)
                    # Fetch the page text snippet
                    pages = db.get_pages(r["filename"])
                    page_text = ""
                    for p in pages:
                        if p["page_num"] == r["page_num"]:
                            page_text = p["text"][:200]
                            break
                    results.append(
                        {
                            "filename": r["filename"],
                            "page_num": r["page_num"],
                            "snippet": page_text,
                            "source": "semantic",
                        }
                    )
            logger.info("tool.search_pdfs.vector", hits=len(vec_results))
        except Exception as e:
            logger.warning("tool.search_pdfs.vector_error", error=str(e))

    # Phase 2: FTS5 keyword search
    try:
        fts_results = db.search(query, limit=limit)
        for r in fts_results:
            key = f"{r['filename']}:{r['page_num']}"
            if key not in seen:
                seen.add(key)
                results.append(
                    {
                        "filename": r["filename"],
                        "page_num": r["page_num"],
                        "snippet": r.get("snippet", ""),
                        "source": "keyword",
                    }
                )
        logger.info("tool.search_pdfs.fts", hits=len(fts_results))
    except Exception as e:
        logger.warning("tool.search_pdfs.fts_error", error=str(e))

    logger.info("tool.search_pdfs.done", query=query, count=len(results))
    return results[:limit]


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
