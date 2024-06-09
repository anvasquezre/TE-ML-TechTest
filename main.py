
import os
import shutil
import lib.utils as utils
from openai import OpenAI
from passlib.hash import bcrypt
import lib.processing as processing
import security.security as security
from qdrant_client import QdrantClient
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, FastAPI, HTTPException, status, Path, UploadFile, File, Form

# Load the configuration file
config = utils.load_config('./config.yaml')

# Load documentation of the API development and show the information when is deployed
tag = utils.load_json(r'./data/tags.json') 
tags = [tag[t] for t in tag]

client_ia = OpenAI(api_key=config['openai']['key'])


# Initialize Qdrant client with the host and port according to the configuration file
client_vdb = QdrantClient(config['qdrant']['host'], port=config['qdrant']['port'])

app = FastAPI(title="Machine Learning Engineer Technical Test", openapi_tags= tags)

origins = ["*"]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Route for the home page
@app.get("/", tags=["Home"])
async def home():
    """
    Check if the API is running correctly.

    Returns:
        dict: A dictionary indicating that the API is running correctly.
    """
    return {'API created by Fabian Coy -2024 Machine Engineer Test': 'API to extract data using OCR, load data to QDRANT and get answers using LLM model.'}


#### Testing to upload a file endpoint
@app.post("/extract_data/", tags=["OCR"])
async def extract_data(
    file: UploadFile = File(...),
    name: str = Form(...),
    lastname: str = Form(...),
    current_user: security.User = Depends(security.get_current_active_user)
):
    """
    Extract data from an uploaded file using OCR.

    Args:
        file (UploadFile): The file to be processed.
        name (str): The name of the person.
        lastname (str): The last name of the person.
        current_user (security.User): The current authenticated user.

    Returns:
        JSONResponse: A JSON response indicating the outcome of the data extraction process.
    """
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith('.pdf'):
        file_path, extracted_text = processing.handle_pdf(file_path)

    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path, extracted_text = processing.convert_to_png(file_path)
    else:
        return {"error": "Invalid file type. Please upload a correct file in format '.png', '.jpg', '.jpeg', '.bmp', '.gif' or '.pdf'"}

    try:        
        outcome = processing.process_file(file_path, name, lastname, config, extracted_text)
        return JSONResponse(content=outcome, status_code=200)
    except Exception:
        return {"error": "An error occurred while processing the file."} 
 
# Upload a .pdf file endpoint
@app.post("/upload/", tags=["Upload_file"])
async def upload_file(
    file: UploadFile = File(...),
    current_user: security.User = Depends(security.get_current_active_user)):
    """
    Upload or update a document in Qdrant vector database.

    Args:
        file (UploadFile): The .pdf file to be uploaded.
        current_user (security.User): The current authenticated user.

    Returns:
        JSONResponse: A JSON response indicating the status of the upload process.
    """
    # check if the file is a .docx file 
    if not file.filename.lower().endswith('.pdf') :
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file or provide a valid ID.")

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith('.pdf'):
        file_path, extracted_text = processing.handle_pdf(file_path)

    # read the contents of the file, create embeddings, and upload the document to Qdrant
    try:
 
        embeddings, _, text = processing.create_embeddings(client_ia, extracted_text, config['openai']['tokenizer'], config['openai']['model'])
        for i in range(len(embeddings)):
            processing.upload_documents(config['qdrant']['collection'], client_vdb, embeddings[i], text[i])
        return JSONResponse(content={"the filename": file.filename, "process created": "Document uploaded successfully"}, status_code=200)

    except Exception:
        raise HTTPException(status_code=500, detail=f"Error processing file: {file.filename}")

# Route to search the best document based on the prompt and create an answer using LLM model
@app.get("/search/{prompt}", tags=["Search"])
# async def read_words(prompt: str = Path(..., max_length=config['api']['max_length_prompt']), current_user: security.User = Depends(security.get_current_active_user)):
async def read_words(prompt: str = Path(..., max_length=config['api']['max_length_prompt']),
                     current_user: security.User = Depends(security.get_current_active_user)):
    """
    Search for words or phrases in the documents stored in Qdrant vector database.

    Args:
        prompt (str): The search prompt or query.
        current_user (security.User): The current authenticated user.

    Returns:
        JSONResponse: A JSON response indicating the LLM outcome using the prompt and document.
    """
    try:
        response = processing.get_answer_RAG(config, client_vdb, client_ia, prompt)
        return JSONResponse(content={"LLM answer": response}, status_code=200)
    
    except Exception:
        raise HTTPException(status_code=500, detail="Error conextion with LLM")


# Route to generate a token for accessing the API securely
@app.post("/token", response_model=security.Token, tags=["Generate Token"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Generate a token for accessing the API securely.

    Args:
        form_data (OAuth2PasswordRequestForm): The form data containing the username and password.

    Returns:
        dict: A dictionary containing the access token and token type.
    """
    user = security.authenticate_user(security.db_dummy, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config['secure']['ACCESS_TOKEN_EXPIRE_MINUTES'])
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


