# Dockerfile
FROM python:3.11-slim

# LightGBM runtime dep
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_VERSION="lgbm_iforest_2025-09-16"

# Work under /app
WORKDIR /app

# 1) install Python deps (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) copy service code
COPY . ./ml-service

# Expose Flask port
EXPOSE 8000

# Run from the service dir
WORKDIR /app/ml-service
CMD ["python", "app.py"]
