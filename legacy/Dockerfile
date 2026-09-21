# Dockerfile -- Tune3 v2 reproducible environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# Sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Python pinned
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip==24.0 setuptools==69.0.3 wheel==0.42.0 \
    && pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -r /workspace/requirements.txt

COPY . /workspace/
RUN pip install -e .

# Usuario nao-root
RUN useradd -m -u 1000 tune3user \
    && chown -R tune3user:tune3user /workspace
USER tune3user

HEALTHCHECK --interval=10m --timeout=60s --start-period=30s --retries=2 \
    CMD python -c "import tune3" || exit 1

CMD ["bash"]