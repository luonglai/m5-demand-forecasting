# Reproducible local environment: Python + Java + Spark, same as CI.
#   docker build -t m5fc .
#   docker run --rm m5fc                                   # run the test suite
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#       m5fc m5fc --config conf/local.yml --stage all      # full pipeline on the real data
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPARK_DRIVER_MEMORY=6g

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev]"

COPY conf ./conf
COPY tests ./tests

CMD ["pytest", "-q"]
