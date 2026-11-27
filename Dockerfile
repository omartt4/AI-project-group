# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.9 AS uv-bin

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    STANZA_RESOURCES_DIR="/opt/stanza_resources" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8501

COPY --from=uv-bin /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen \
    && python -c "import os, stanza; stanza.download('ar', processors='tokenize,mwt,pos,lemma,depparse', model_dir=os.environ['STANZA_RESOURCES_DIR'], verbose=False)" \
    && python scripts/check_nlp_runtime.py --require-spacy --require-arabic-dependencies

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
