FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app
COPY pyproject.toml ./
RUN python -m pip install --upgrade pip wheel && \
    python -m pip install \
        torch>=2.5 \
        transformers>=4.44 \
        datasets>=2.20 \
        einops>=0.8 \
        pyyaml>=6.0 \
        fastapi>=0.115 \
        "uvicorn[standard]>=0.30" \
        pydantic>=2.7 \
        sentencepiece>=0.2

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV ZAR1_CONFIG=/app/configs/zar1_7b.yaml
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "zar1.server:app", "--host", "0.0.0.0", "--port", "8000"]
