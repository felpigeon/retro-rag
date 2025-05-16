FROM python:3.12-slim

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

WORKDIR /app
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --locked

COPY src src
COPY baml_src baml_src
RUN uv run baml-cli generate

EXPOSE 5000

CMD ["uv", "run", "gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "src.app:app"]
