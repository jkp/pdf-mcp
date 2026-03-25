"""FastMCP server instance and entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastmcp import FastMCP

from pdf_mcp.config import Settings
from pdf_mcp.db import Database
from pdf_mcp.indexer import Indexer

settings = Settings()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog._log_levels.NAME_TO_LEVEL.get(settings.log_level.lower(), 20)
    ),
)
logger = structlog.get_logger()

db = Database(settings.database_path)
indexer = Indexer(db=db, vault=settings.vault)


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Index PDFs on startup, then serve."""
    vault = settings.vault
    if not vault.is_dir():
        logger.warning("server.vault_not_found", path=str(vault))
        vault.mkdir(parents=True, exist_ok=True)

    # Index all PDFs on startup
    stats = indexer.index_all()
    logger.info("server.indexed", **stats)

    logger.info("server.ready", vault=str(vault))
    yield
    db.close()
    logger.info("server.shutdown")


def _build_auth():
    """Build OAuth auth provider if GitHub credentials are configured."""
    if not settings.github_client_id or not settings.github_client_secret:
        return None

    from fastmcp.server.auth.providers.github import GitHubProvider

    return GitHubProvider(
        client_id=settings.github_client_id,
        client_secret=settings.github_client_secret,
        base_url=settings.oauth_base_url or f"http://localhost:{settings.port}",
    )


mcp = FastMCP(
    name="PDF Filing Cabinet",
    auth=_build_auth(),
    lifespan=_lifespan,
)

# Import tools to register them
import pdf_mcp.tools.managing  # noqa: F401, E402
import pdf_mcp.tools.reading  # noqa: F401, E402
import pdf_mcp.tools.searching  # noqa: F401, E402


def _build_app():
    """Build the ASGI app for HTTP transport."""
    from starlette.responses import JSONResponse

    _app = mcp.http_app(transport="http", stateless_http=True, path=settings.mcp_path)

    from starlette.routing import Route

    async def oauth_protected_resource(request):
        base = str(settings.oauth_base_url or f"http://{settings.host}:{settings.port}")
        return JSONResponse(
            {
                "resource": base,
                "authorization_servers": [base],
            }
        )

    _app.router.routes.append(
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource)
    )

    return _app


app = _build_app() if settings.transport == "http" else None


def main() -> None:
    """Entry point for the MCP server."""
    logger.info(
        "server.starting",
        transport=settings.transport,
        vault=str(settings.vault),
    )
    if settings.transport == "http":
        import uvicorn

        assert app is not None
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            lifespan="on",
        )
    else:
        mcp.run()
