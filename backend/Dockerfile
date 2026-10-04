# Backend API image for Render (or any Docker host).
#
# The trained models are produced INSIDE the image at build time by the
# project's own training pipeline (python -m app.train) -- nothing is
# downloaded or hard-coded. Training takes ~10 minutes, once per deploy.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# Train the models + compute every report the dashboard shows. The local
# SQLite file train.py may leave behind is removed: user data lives in
# Supabase Postgres (DATABASE_URL), never in the image.
RUN OMP_NUM_THREADS=4 python -m app.train \
 && rm -f data/app_state.sqlite3 \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*' --workers 1"]
