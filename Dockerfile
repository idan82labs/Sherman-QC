# ShermanChat production image for Render.

FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend/react
COPY frontend/react/package*.json ./
RUN npm ci
COPY frontend/react/ ./
RUN npm run build


FROM python:3.11-slim AS python-builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /install /usr/local

COPY apps/ ./apps/
COPY backend/ ./backend/
COPY contracts/ ./contracts/
COPY domains/ ./domains/
COPY infrastructure/ ./infrastructure/
COPY requirements.txt .

COPY data/manual_assistant/index/ ./data/manual_assistant/index/
COPY data/manual_assistant/page_images/ ./data/manual_assistant/page_images/
COPY data/manual_assistant/crops/ ./data/manual_assistant/crops/
COPY data/manual_assistant/gold_eval_cases.json ./data/manual_assistant/gold_eval_cases.json

COPY --from=frontend-builder /app/frontend/dist/ ./frontend/

RUN mkdir -p /app/data/manual_assistant/uploads /app/uploads /app/output /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    PRODUCTION=true \
    LOG_LEVEL=INFO \
    SHERMAN_CHAT_PROVIDER=mock \
    SHERMAN_CHAT_MODEL=gpt-5.5 \
    SHERMAN_RETRIEVAL_BACKEND=local \
    SHERMAN_MANUAL_DATA_DIR=/app/data/manual_assistant \
    SHERMAN_CHAT_INCLUDE_RETRIEVAL_TRACE=false \
    SHERMAN_CHAT_ALLOW_REMOTE_CODEX=false

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '10000') + '/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
