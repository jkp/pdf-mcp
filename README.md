# pdf-mcp

An MCP server that gives LLMs searchable access to your PDF filing cabinet.

Point it at a folder of PDFs and it indexes every document — extracting text page by page, building a full-text search index, and embedding content for semantic similarity search. Works with Claude.ai, Claude Code, and any MCP-compatible client.

## Features

- **Hybrid search** — FTS5 keyword matching + semantic vector search (e5-large via sqlite-vec), cross-encoder reranking (bge-reranker-v2-m3), and LLM relevance filtering
- **Handles scanned PDFs** — pymupdf extracts text from OCR'd documents
- **Incremental indexing** — SHA256 hash tracking means restarts only process new or changed files
- **Single-file database** — everything stored in one portable SQLite file (FTS5 + vectors)
- **Zero external services** — runs fully local; Together API optional for faster bulk embedding

## Quick start

```bash
# Install
git clone https://github.com/jkp/pdf-mcp.git
cd pdf-mcp
uv sync

# Point at your PDFs
export PDF_MCP_VAULT_PATH=~/Documents/PDFs

# Run (stdio mode — for Claude Code, Cursor, etc.)
uv run pdf-mcp
```

## Configuration

All settings use the `PDF_MCP_` prefix and can be set via environment variables or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_PATH` | `~/Documents/PDFs` | Directory containing your PDFs |
| `DB_PATH` | `~/.local/share/pdf-mcp/pdf.db` | SQLite database location |
| `TRANSPORT` | `stdio` | `stdio` or `http` |
| `PORT` | `10201` | HTTP server port |
| `MCP_PATH` | `/mcp` | MCP endpoint path (set to `/` behind a path-stripping proxy) |
| `TOGETHER_API_KEY` | | Optional — enables faster bulk embedding and LLM relevance scoring |
| `LOG_LEVEL` | `INFO` | Logging level |

### OAuth (for HTTP transport)

| Variable | Description |
|----------|-------------|
| `GITHUB_CLIENT_ID` | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth app client secret |
| `OAUTH_BASE_URL` | Public URL of the server |
| `OAUTH_ALLOWED_USERS` | Comma-separated GitHub usernames |
| `OAUTH_STATE_DIR` | Directory for OAuth state persistence |

## MCP tools

| Tool | Description |
|------|-------------|
| `search_pdfs` | Hybrid full-text + semantic search with reranking and relevance filtering |
| `list_pdfs` | List/filter PDFs by filename pattern |
| `read_pdf` | Extract text from a PDF (all pages or specific range) |
| `get_pdf_info` | PDF metadata — page count, author, title, file size |
| `rename_pdf` | Rename a PDF with automatic reindex |
| `reindex_pdfs` | Manually trigger reindexing of all PDFs |

## Search pipeline

```
query
  │
  ├─► Vector search (e5-large embeddings, sqlite-vec KNN)
  ├─► FTS5 keyword search (porter stemming)
  │
  └─► Union + deduplicate
        │
        ├─► Cross-encoder reranker (bge-reranker-v2-m3)
        └─► LLM relevance filter (scores 1-5, drops noise)
              │
              └─► Results
```

## Deployment

### Docker / Podman

```bash
docker build -t pdf-mcp .
docker run -v ~/Documents/PDFs:/data/vault -v pdf-mcp-data:/data \
  -e PDF_MCP_VAULT_PATH=/data/vault \
  -e PDF_MCP_DB_PATH=/data/pdf.db \
  -e PDF_MCP_TRANSPORT=http \
  -p 10201:10201 \
  pdf-mcp
```

### Claude Code (stdio)

Add to your MCP config:

```json
{
  "mcpServers": {
    "pdf": {
      "command": "uv",
      "args": ["--directory", "/path/to/pdf-mcp", "run", "pdf-mcp"],
      "env": {
        "PDF_MCP_VAULT_PATH": "~/Documents/PDFs"
      }
    }
  }
}
```

### Claude.ai (remote HTTP)

Add as a remote MCP server using the HTTP URL (e.g., `https://your-server/mcp`). OAuth will handle authentication.

## Performance

First startup indexes and embeds all PDFs:
- **Text extraction**: ~2s per PDF (~30 min for 800 PDFs)
- **Embedding**: ~50 PDFs/min local, ~500 PDFs/min via Together API
- **Search**: <1s per query (vector + FTS5 + reranking)

Subsequent startups are instant — hash-based skip.

## Development

```bash
uv sync
uv run pytest -v
```

## License

MIT
