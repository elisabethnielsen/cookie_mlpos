# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY uv.lock uv.lock


WORKDIR /
RUN uv sync --locked --no-cache 

RUN mkdir -p /reports/figures
# locked means use the uv.lock file
# no cache means that uv does not use already-installed packages
# it install everything fresh to ensure reproducibility

ENTRYPOINT ["uv", "run", "src/cookie/evaluate.py"]