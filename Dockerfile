FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 assay
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Code only; .dockerignore keeps data/ out. Mount data read-only at /app/data at run time:
#   docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/cases_r2:/app/cases_r2" assay case CASE=HHF-007
COPY . .
RUN mkdir -p /app/data /app/cases_r2 && chown -R assay:assay /app
USER assay
ENV PY=python PYTHONPATH=/app
ENTRYPOINT ["make"]
CMD ["case"]
