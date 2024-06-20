# Use a lightweight Python image
FROM python:3.9-slim-buster

# Install necessary system packages
RUN apt-get update && apt-get -y install \
    gcc \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libffi-dev \
    libssl-dev \
    libxml2 \
    libgdk-pixbuf2.0-0 \
    libxrender1 \
    libjpeg-dev \
    libgdk-pixbuf2.0-dev \
    libcairo2 \
    libpango1.0-dev \
    librsvg2-dev \
    fonts-liberation \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    wget

# Install the latest version of sqlite3
RUN apt-get remove -y sqlite3 && \
    wget https://www.sqlite.org/2022/sqlite-autoconf-3390400.tar.gz && \
    tar xzf sqlite-autoconf-3390400.tar.gz && \
    cd sqlite-autoconf-3390400 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf sqlite-autoconf-3390400 sqlite-autoconf-3390400.tar.gz

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the models directory
# COPY models /app/models
# COPY models/db_resnet50-79bd7d70.pt /app/models/db_resnet50-79bd7d70.pt
# COPY models/crnn_vgg16_bn-9762b0b0.pt /app/models/crnn_vgg16_bn-9762b0b0.pt

# Copy the models directory (now inside app)
COPY app/models /app/app/models

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
