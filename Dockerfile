# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app PORT=8000

WORKDIR $APP_HOME
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Download spaCy model at build-time (immutable image)
RUN python -m spacy download en_core_web_sm

COPY . .
# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000
# Healthcheck hits /readyz
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python","-c","import urllib.request,sys; import urllib.error; \
try: \
  r=urllib.request.urlopen('http://127.0.0.1:8083/readyz', timeout=2); \
  sys.exit(0 if r.status==200 else 1); \
except Exception: \
  sys.exit(1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
