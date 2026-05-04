# ZAR-1 Inference Server Docker Image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY ./pyproject.toml ./README.md /app/
COPY ./src /app/src
COPY ./configs /app/configs

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e /app && \
    pip install uvicorn[standard] pydantic

# Set environment variables for inference
ENV CUDA_VISIBLE_DEVICES=0
ENV ZAR1_DTYPE=bfloat16

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "zar1.server", "--checkpoint", "/models/latest.pt", "--port", "8000", "--host", "0.0.0.0"]
