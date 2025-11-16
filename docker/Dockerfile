# Multi-stage Dockerfile for DeepLOB MLOps Pipeline

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create directories
RUN mkdir -p data/raw data/processed checkpoints logs

# Expose ports
EXPOSE 8000 5000

# Default command
CMD ["python", "-m", "src.serving.api"]


# Training stage
FROM base as training

CMD ["python", "src/training/train.py"]


# Serving stage
FROM base as serving

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
