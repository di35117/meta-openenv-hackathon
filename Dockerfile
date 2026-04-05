# ASHA Village Health — OpenEnv
# Standalone Dockerfile that does NOT depend on the private openenv-base image.
# Works on Hugging Face Spaces (Docker SDK) and any standard Docker host.

FROM python:3.11-slim

# HF Spaces runs as a non-root user with UID 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# System deps: git (for pip VCS installs) + curl (healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Set PYTHONPATH so 'from models import ...' and 'from server.app import ...' both work
ENV PYTHONPATH="/app"

USER user

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn. 'server.app:app' works because PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
