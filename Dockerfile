FROM ghcr.io/astral-sh/uv:0.7.4 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /usr/local/bin/uv

# tesseract for OCR of scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev

CMD ["uv", "run", "pdf-mcp"]
