FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAGQA_INDEX_DIR=/app/index \
    RAGQA_DATA_DIR=/app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ragqa ./ragqa
COPY data ./data

# Build the sample index at image build time so the API answers immediately,
# then drop root privileges.
RUN python -m ragqa.cli --index /app/index ingest data/sample \
    && useradd --create-home --uid 1000 app \
    && chown -R app /app/index
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
