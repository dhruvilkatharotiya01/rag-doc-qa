FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAGQA_INDEX_DIR=/app/index

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ragqa ./ragqa
COPY data ./data
COPY eval ./eval

# Build the sample index at image build time so the API answers immediately.
RUN python -m ragqa.cli --index /app/index ingest data/sample

EXPOSE 8000
CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
