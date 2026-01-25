FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        swig \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps from uv.lock (keeps Docker layer caching stable)
COPY pyproject.toml uv.lock README.md ./
COPY diverserl/ diverserl/
ARG UV_SYNC_ARGS="--frozen --no-dev"
RUN uv sync ${UV_SYNC_ARGS}

# Copy the rest of the repo (scripts, configs, etc.)
COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "run.py"]
