import logging
from fastapi import FastAPI
from app.endpoints.pdf_processing import router as pdf_processing_router
from app.endpoints.questions import router as question_router

# Configurar el logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Guarda los logs en el archivo app.log
        logging.StreamHandler()  # También muestra los logs en la consola
    ]
)

app = FastAPI()

app.include_router(pdf_processing_router, prefix="/api")
app.include_router(question_router, prefix="/api")