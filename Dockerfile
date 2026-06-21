# ShermanChat production image for Render.
#
# This image intentionally ships the chat/RAG surface only. The broader QC
# platform imports heavy 3D/vision dependencies that are not needed for the
# public ShermanChat deployment.

FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend/react
COPY frontend/react/package*.json ./
RUN npm ci
COPY frontend/react/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

COPY requirements.chat.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/__init__.py ./apps/__init__.py
COPY apps/api/__init__.py ./apps/api/__init__.py
COPY apps/api/chat_main.py ./apps/api/chat_main.py
COPY apps/api/routes/__init__.py ./apps/api/routes/__init__.py
COPY apps/api/routes/manual_assistant.py ./apps/api/routes/manual_assistant.py
COPY apps/api/services/__init__.py ./apps/api/services/__init__.py
COPY apps/api/services/manual_assistant_service.py ./apps/api/services/manual_assistant_service.py
COPY domains/__init__.py ./domains/__init__.py
COPY domains/manual_assistant/ ./domains/manual_assistant/
COPY infrastructure/__init__.py ./infrastructure/__init__.py
COPY infrastructure/rag/ ./infrastructure/rag/

COPY data/manual_assistant/index/ ./data/manual_assistant/index/
COPY data/manual_assistant/page_images/ ./data/manual_assistant/page_images/
COPY data/manual_assistant/crops/ ./data/manual_assistant/crops/
COPY data/manual_assistant/gold_eval_cases.json ./data/manual_assistant/gold_eval_cases.json

COPY --from=frontend-builder /app/frontend/dist/ ./frontend/

RUN mkdir -p /app/data/manual_assistant/uploads

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
    SHERMAN_CHAT_ALLOW_REMOTE_CODEX=false \
    SHERMAN_CHAT_REQUIRE_AUTH=false

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '10000') + '/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn apps.api.chat_main:app --host 0.0.0.0 --port ${PORT:-10000}"]
