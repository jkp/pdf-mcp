"""PDF management tools (rename, reindex)."""

from typing import Any

import structlog

from pdf_mcp.server import indexer, mcp, settings

logger = structlog.get_logger()


@mcp.tool(annotations={"destructiveHint": False, "title": "Rename PDF"})
async def rename_pdf(old_filename: str, new_filename: str) -> dict[str, Any]:
    """Rename a PDF file in the filing cabinet.

    Args:
        old_filename: Current filename
        new_filename: New filename
    """
    logger.info("tool.rename_pdf", old=old_filename, new=new_filename)

    old_path = settings.vault / old_filename
    new_path = settings.vault / new_filename

    if not old_path.is_file():
        return {"error": f"PDF not found: {old_filename}"}
    if new_path.exists():
        return {"error": f"Target already exists: {new_filename}"}

    old_path.rename(new_path)

    # Reindex under the new name
    from pdf_mcp.server import db

    db.delete_pdf(old_filename)
    indexer.index_file(new_filename)

    return {"old": old_filename, "new": new_filename, "status": "renamed"}


@mcp.tool(annotations={"readOnlyHint": True, "title": "Reindex PDFs"})
async def reindex_pdfs() -> dict[str, Any]:
    """Reindex all PDFs in the filing cabinet.

    Use this after adding new PDFs or if search results seem stale.
    """
    logger.info("tool.reindex_pdfs")
    stats = indexer.index_all()
    return stats
