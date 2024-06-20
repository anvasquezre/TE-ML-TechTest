# DEV_NOTE.md

## Project Overview

This project is designed to extract text and names from PDF documents using OCR, store the extracted text in a vector database (ChromaDB), and implement a Retrieval-Augmented Generation (RAG) strategy using OpenAI's GPT-4 model to answer questions based on the stored text.

## Code Structure

The project is structured as follows:

```bash
    ├── app/
    │ ├── api.py
    │ ├── doctr_module.py
    │ ├── embeddings_model.py
    │ ├── image_processing.py
    │ ├── ner.py
    │ ├── pdf_ocr.py
    │ ├── rag.py
    │ ├── vector_database.py
    │ ├── init.py
    ├── config/
    │ ├── config.py
    │ ├── init.py
    ├── data/
    │ ├── driver_license_data/
    │ ├── residential_contract/
    │ ├── results/
    ├── database/
    │ ├── chroma.sqlite3
    ├── logs/
    │ ├── error.log
    │ ├── info.log
    ├── manual_tests/
    │ ├── print_tree.py
    │ ├── README.md
    │ ├── run.rag.py
    │ ├── run_doctr_ocr.py
    │ ├── run_doctr_ocr_display.py
    │ ├── run_explore_chroma.py
    │ ├── run_fix_orientation.py
    │ ├── run_ner.py
    │ ├── run_pdf_ocr.py
    ├── models/
    │ ├── crnn_vgg16_bn-9762b0b0.pt
    │ ├── db_resnet50-79bd7d70.pt
    ├── requirements.txt
    ├── Dockerfile
    └── .dockerignore
```

## External Libraries and Resources

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **doctr**: A library for OCR using deep learning.
- **OpenAI API**: For GPT-4 model to generate answers based on the context retrieved from the vector database.
- **ChromaDB**: An open-source vector database used to store and query text embeddings.
- **Sentence-Transformers**: For converting text into vector representations.

## Setup and Installation

### Prerequisites

- Docker
- Python 3.9 or later

### Instructions

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv ml_env
    source ml_env/bin/activate  # On Windows, use `ml_env\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the application using Docker:**

    1. **Build the Docker image:**
        ```bash
        docker build -t te-ml-challenge .
        ```

    2. **Run the Docker container:**
        ```bash
        docker run -p 8000:8000 --env-file .env te-ml-challenge
        ```

### Running Locally with FastAPI and Uvicorn

1. **Activate your virtual environment:**
    ```bash
    source ml_env/bin/activate  # On Windows, use `ml_env\Scripts\activate`
    ```

2. **Run the FastAPI application with Uvicorn:**
    ```bash
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
    ```

This will start the FastAPI application on `localhost` at port `8000`. You can then access the API endpoints through your web browser or using tools like `curl` or Postman.

## Usage

### Endpoints

1. **Upload PDF and Store Text in Vector Database:**
    - **Endpoint:** `/upload-pdf/`
    - **Method:** `POST`
    - **Form Data:** 
        - `file`: The PDF file to be uploaded.

2. **Extract Names and Bounding Boxes:**
    - **Endpoint:** `/extract/`
    - **Method:** `POST`
    - **Form Data or JSON:** 
        - `file_location`: The location of the file (image or PDF).
        - `names`: List of names to be extracted.

3. **Ask a Question:**
    - **Endpoint:** `/ask-question/`
    - **Method:** `POST`
    - **JSON:** 
        - `question`: The question to be asked.

## Example Requests

1. **Upload PDF:**
    ```bash
    curl -X POST "http://localhost:8000/upload-pdf/" -F "file=@path_to_your_pdf_file.pdf"
    ```

2. **Extract Names:**
    ```bash
    curl -X POST "http://localhost:8000/extract/" -H "Content-Type: application/json" -d '{
        "file_location": "path_to_your_file",
        "names": ["John Doe", "Jane Smith"]
    }'
    ```

3. **Ask a Question:**
    ```bash
    curl -X POST "http://localhost:8000/ask-question/" -H "Content-Type: application/json" -d '{
        "question": "What are the financing options available in this contract?"
    }'
    ```

## Logging

Logs are stored in the `logs/` directory:
- `info.log`: General information and operational logs.
- `error.log`: Error logs.

## Additional Notes

- Ensure your `.env` file is correctly set up with your OpenAI API key.
- The models for `doctr` are stored in the `models/` directory.

## Troubleshooting

- **Docker Issues:** Ensure Docker is installed and running properly. Follow the installation guide on the [Docker website](https://docs.docker.com/get-docker/).
- **API Issues:** Check the logs in the `logs/` directory for any errors and troubleshoot accordingly.
