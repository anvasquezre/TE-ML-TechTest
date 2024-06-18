from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
from app.services.pdf_processor import process_pdf
import json
import logging

router = APIRouter()

@router.post("/process_pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...),
                               names: str = Form(...)):
    """
    Endpoint to process a PDF and find names, their coordinates, and fuzzy match
    scores.

    Parameters:
    file (UploadFile): The uploaded PDF file.
    names (List[str]): A list of names to compare.

    Returns:
    dict: A dictionary containing the names found, their coordinates, and fuzzy
    match scores.
    """
    logging.info("Received request to process PDF")
    logging.info(f"Nombres originales: {names}")
    if not file.filename.endswith('.pdf'):
        logging.error("Invalid file format")
        raise HTTPException(status_code=400,
                            detail="Invalid file format. Only PDF files are accepted.")

    try:
        # Cambiar la ruta de escritura al directorio actual
        pdf_path = f"{file.filename}"
        with open(pdf_path, "wb") as buffer:
            logging.info("Saving uploaded file")
            buffer.write(file.file.read())

        # Asegurarse de que 'names' no esté vacío y sea una cadena JSON válida
        if not names:
            logging.error("Names list cannot be empty.")
            raise HTTPException(status_code=400,
                                detail="Names list cannot be empty.")

        try:
            logging.info(f"Names received: {names}")
            names_list = [name.strip() for name in names.split(",")]
            logging.info(f"Parsed names list: {names_list}")
        except json.JSONDecodeError:
            logging.error("Names must be a valid JSON list of strings.")
            raise HTTPException(status_code=400,
                                detail="Names must be a valid JSON list of strings.")

        logging.info("Processing PDF")
        result = process_pdf(pdf_path, names_list)
        logging.info("Finished processing PDF")
        return result
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


