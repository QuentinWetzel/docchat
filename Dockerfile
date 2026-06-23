FROM python:3.11-slim

# System deps for torch / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
# CPU-only wheel: the default PyPI torch wheel bundles CUDA libs nobody needs on Railway's
# CPU containers, which bloats the image and risks build timeouts/OOM.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# bge-m3 + reranker (~4.5 GB combined) don't fit in Railway Hobby's 5 GB volume cap, so there's
# no persistent volume here -- HF_HOME just lives on the container's own ephemeral disk and
# re-downloads on every restart/redeploy.
ENV HF_HOME=/models

EXPOSE 8000
# Railway provides $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
