FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


COPY . /app

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
