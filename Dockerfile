# Use slim Python image
FROM python:3.10.16-slim

# Prevent pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first (cache friendly)
COPY pyproject.toml /app/

# Install Python dependencies with timeout
RUN pip install --upgrade pip \
    && pip install -e . --default-timeout=200

# Copy application code
COPY api /app/api
COPY bike_sharing_model /app/bike_sharing_model

# Expose FastAPI port
EXPOSE 8089

# Healthcheck for container
HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -f http://localhost:8089/docs || exit 1

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8089"]
