# ResolveIQ API image -- slim (ONNX embeddings, no torch).
# NOTE: authored but not built/tested in the environment it was written in
# (no Docker available there); run `docker build` before relying on it.
FROM python:3.13-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RESOLVEIQ_EMBEDDING_BACKEND=onnx \
    RESOLVEIQ_AUTO_SEED_KNOWLEDGE=true \
    HOME=/home/app

# tesseract: image OCR in the evidence ingestion pipeline (pytesseract).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /srv/resolveiq
COPY requirements-base.txt requirements-cloud.txt ./
RUN pip install -r requirements-base.txt -r requirements-cloud.txt

# Bake the ONNX embedding model into the image so containers don't download
# ~80 MB on every start (Chroma caches it under $HOME/.cache/chroma).
RUN mkdir -p /home/app && HOME=/home/app python -c \
    "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as E; E()(['warm'])" \
    && chown -R app:app /home/app

COPY app ./app
COPY scripts ./scripts
COPY data/sample_knowledge ./data/sample_knowledge
RUN mkdir -p /srv/resolveiq/data && chown -R app:app /srv/resolveiq/data

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker: chat enhancement jobs and TFS/Wiki caches are in-process state.
# Scale out with replicas behind a load balancer (needs RESOLVEIQ_DATABASE_URL
# and RESOLVEIQ_CHROMA_HOST so replicas share state).
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
