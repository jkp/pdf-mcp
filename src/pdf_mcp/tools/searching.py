"""PDF search tools — FTS5 + semantic vector search + reranker + relevance."""

from typing import Any

import structlog

from pdf_mcp.relevance import score_relevance
from pdf_mcp.server import db, embedder, mcp, settings

logger = structlog.get_logger()


@mcp.tool(annotations={"readOnlyHint": True, "title": "Search PDFs"})
async def search_pdfs(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search across all indexed PDFs using full-text and semantic search.

    Combines FTS5 keyword matching with vector similarity search,
    reranks with a cross-encoder, and filters by LLM relevance scoring.

    Args:
        query: Search query (natural language or keywords)
        limit: Maximum number of results
    """
    logger.info("tool.search_pdfs", query=query, limit=limit)

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    # Phase 1: Vector search (semantic)
    if embedder:
        try:
            vec_results = embedder.search(query, limit=limit)
            for r in vec_results:
                key = f"{r['filename']}:{r['page_num']}"
                if key not in seen:
                    seen.add(key)
                    pages = db.get_pages(r["filename"])
                    page_text = ""
                    for p in pages:
                        if p["page_num"] == r["page_num"]:
                            page_text = p["text"][:500]
                            break
                    candidates.append(
                        {
                            "filename": r["filename"],
                            "page_num": r["page_num"],
                            "snippet": page_text,
                        }
                    )
            logger.info("tool.search_pdfs.vector", hits=len(vec_results))
        except Exception as e:
            logger.warning("tool.search_pdfs.vector_error", error=str(e))

    # Phase 2: FTS5 keyword search
    try:
        fts_results = db.search(query, limit=limit * 2)
        for r in fts_results:
            key = f"{r['filename']}:{r['page_num']}"
            if key not in seen:
                seen.add(key)
                candidates.append(
                    {
                        "filename": r["filename"],
                        "page_num": r["page_num"],
                        "snippet": r.get("snippet", ""),
                    }
                )
        logger.info("tool.search_pdfs.fts", hits=len(fts_results))
    except Exception as e:
        logger.warning("tool.search_pdfs.fts_error", error=str(e))

    # Phase 3: Cross-encoder reranker
    if embedder and candidates:
        try:
            ranked = embedder.rerank(query, candidates)
            candidates = [c for _, c in ranked]
            logger.info("tool.search_pdfs.reranked", count=len(candidates))
        except Exception as e:
            logger.warning("tool.search_pdfs.rerank_error", error=str(e))

    results = candidates[:limit]

    # Phase 4: LLM relevance filter
    if results and settings.together_api_key:
        try:
            results = await score_relevance(
                query, results, api_key=settings.together_api_key
            )
        except Exception as e:
            logger.warning("tool.search_pdfs.relevance_error", error=str(e))

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
