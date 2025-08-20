FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

WORKDIR /app

# OCR: tesseract + poppler pdf2imagea varten
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-fin poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Asenna python-riippuvuudet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Koodi
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
