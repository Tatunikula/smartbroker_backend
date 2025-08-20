# Playwrightin virallinen python-image -> mukana Chromium + kaikki riippuvuudet
FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

WORKDIR /app

# Järjestelmäpaketit OCR:lle (valinnainen, jos haluat PDF OCR-fallbackin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-fin poppler-utils \
  && rm -rf /var/lib/apt/lists/*

# Python-kirjastot
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sovelluskoodi
COPY . /app

# Render asettaa $PORT:in -> käytä sitä
ENV PYTHONUNBUFFERED=1
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
git init
git add .
git commit -m "initial deploy"

# Luo uusi tyhjä repo GitHubissa nimellä smartbroker_backend (private käy).
# Sitten lisää remote (vaihda <user> sun nimeen):
git branch -M main
git remote add origin https://github.com/<user>/smartbroker_backend.git
git push -u origin main
