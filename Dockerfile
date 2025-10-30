# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the app code
COPY . .

# Non-root user
RUN useradd -m appuser
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
