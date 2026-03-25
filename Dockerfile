FROM ghcr.io/astral-sh/uv:0.7.4 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev

CMD ["uv", "run", "pdf-mcp"]
