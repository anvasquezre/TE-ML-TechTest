import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import shutil
from fuzzywuzzy import fuzz, process
from doctr.models import db_resnet50, crnn_vgg16_bn
from app.doctr_module import load_custom_model, extract_names_and_boxes
from app.pdf_ocr import extract_text_and_boxes_from_pdf
from app.vector_database import store_text_in_vector_db
from app.rag import retrieve_answer
from config.config import logger
from pathlib import Path
import shutil


app = FastAPI()

# Define paths for the model weights
det_weights_path = "models/db_resnet50-79bd7d70.pt"
reco_weights_path = "models/crnn_vgg16_bn-9762b0b0.pt"

# Global variables to hold the models
det_model = None
reco_model = None


# Define a model for input data
class NamePairs(BaseModel):
    file_location: Optional[str] = None
    names: Optional[list] = []


# Custom dependency to handle form data and JSON
async def get_name_pairs(
    request: Request,
    file_location: Optional[str] = Form(None),
    names: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> NamePairs:
    content_type = request.headers.get("content-type")
    if "application/json" in content_type:
        try:
            json_body = await request.json()
            return NamePairs(**json_body)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid JSON input")
    else:
        # Handle form-data
        # If a file is uploaded, use the file's location; otherwise, use the provided file location
        if file:
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_location = temp_file_path

        name_list = names.split(",") if names else []
        return NamePairs(file_location=file_location, names=name_list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global det_model, reco_model
    await load_models()
    yield
    await unload_models()


async def load_models():
    """Load all model configurations at startup."""
    global det_model, reco_model
    try:
        det_model, reco_model = load_custom_model(
            db_resnet50, det_weights_path, crnn_vgg16_bn, reco_weights_path
        )
        logger.info("Models loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Error loading models on startup: {e}")
        raise


async def unload_models():
    """Unload all model configurations at shutdown."""
    global det_model, reco_model
    det_model = None
    reco_model = None
    logger.info("Models unloaded successfully on shutdown.")


app = FastAPI(lifespan=lifespan)


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Received uploaded file and saved to {file_location}")

        # Extract text and boxes from PDF
        _, text_blocks = extract_text_and_boxes_from_pdf(file_location)

        # Store the extracted text in the vector database
        for i, text in enumerate(text_blocks):
            doc_id = f"{file.filename}_page_{i+1}"
            store_text_in_vector_db(doc_id, text)
            logger.info(f"Stored text from page {i+1} in vector database.")

        return {"message": "PDF processed and text stored in vector database."}
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the saved file
        if file_location and Path(file_location).exists():
            Path(file_location).unlink(missing_ok=True)


@app.post("/extract/")
async def extract_names_endpoint(name_pairs: NamePairs = Depends(get_name_pairs)):
    try:
        logger.info(f"Received name_pairs: {name_pairs}")

        file_location = name_pairs.file_location
        name_list = name_pairs.names

        if not file_location:
            raise HTTPException(
                status_code=400, detail="File location must be provided"
            )

        # Assuming extract_names_and_boxes and extract_text_and_boxes_from_pdf are defined
        if file_location.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.info(f"Processing image file: {file_location}")
            extracted_data, _ = extract_names_and_boxes(
                file_location, det_model, reco_model
            )
        elif file_location.lower().endswith(".pdf"):
            logger.info(f"Processing PDF file: {file_location}")
            extracted_data, _ = extract_text_and_boxes_from_pdf(file_location)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        logger.info(f"Extracted data: {extracted_data}")

        matched_names = []
        for full_name in name_list:
            matches = process.extract(
                full_name, [d["name"] for d in extracted_data], scorer=fuzz.ratio
            )
            logger.info(f"Matches for '{full_name}': {matches}")
            for match in matches:
                if match[1] >= 90:
                    for data in extracted_data:
                        if data["name"] == match[0]:
                            matched_names.append(
                                {
                                    "provided_name": full_name,
                                    "extracted_name": data["name"],
                                    "bbox": data["bbox"],
                                    "score": match[1],
                                }
                            )

        logger.info(f"Matched Names: {matched_names}")
        return JSONResponse(content={"matched_names": matched_names})
    except Exception as e:
        logger.error(f"Error in extract_names_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_location and Path(file_location).exists():
            Path(file_location).unlink(missing_ok=True)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        answer = retrieve_answer(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Error retrieving answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
