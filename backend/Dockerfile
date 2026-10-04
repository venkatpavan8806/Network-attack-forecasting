# Backend image for Render. Runs the existing FastAPI app unchanged; the
# models and reports are produced at build time by `python -m app.train`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt .
# CPU-only torch first (the default torch wheel bundles ~2 GB of CUDA libraries)
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
RUN OMP_NUM_THREADS=4 python -m app.train

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
