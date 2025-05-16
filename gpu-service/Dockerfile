FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    curl \
    ca-certificates \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --locked

COPY src src

EXPOSE 5001

CMD ["uv", "run", "gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "src.app:app"]
