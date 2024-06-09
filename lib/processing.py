import lib.utils as utils
import re
import json
import cv2
import fitz 
import tiktoken
import pytesseract
import numpy as np
from PIL import Image
from openai import OpenAI
from qdrant_client import QdrantClient
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# def get_answer_llm(config, client_vdb, client_ia, prompt):     
def get_answer_llm(config, data_extraction, prompt_template = 'prompt_template'):
    """
    Provides an answer from a Language Model (LLM) based on the similarity between the prompt and documents.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
    - prompt (str): The prompt to be used for generating the answer.

    Returns:
    - response (str): The generated answer from the LLM.
    """
    # Create an instance of the ChatOpenAI class for the Language Model
    llm = ChatOpenAI(temperature=0, openai_api_key=config['openai']['key'], model_name=config['openai']['llm_model'])

    prompt_template = PromptTemplate.from_template(config['llm'][prompt_template])
    # filled_prompt = prompt_template.format(question=data_extraction, content=config['llm']['prompt_template'])
    filled_prompt = prompt_template.format(text=data_extraction)

    # Generate the answer from the Language Model using the filled prompt
    response = llm.call_as_llm(filled_prompt)
    return response

# Create a dictionary from a JSON string
def convert_str_to_dict(json_str):
    """
    Converts a JSON string to a dictionary.

    Parameters:
    - json_str (str): The JSON string to be converted.

    Returns:
    - json_dict (dict): The converted dictionary.
    - None: If there is an error decoding the JSON string.
    """
    try:
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError:
        return None

# Extract text from an image using Tesseract OCR and return the extracted text and bounding boxes
def extract_data(image_path, angle=-90):
    """
    Extracts text from an image using Tesseract OCR.

    Parameters:
    - image_path (str): The path to the image file.
    - angle (int, optional): The rotation angle of the image. Defaults to -90.

    Returns:
    - extracted_text: The extracted text from the image.
    - boxes: The bounding boxes of the text regions.
    """
    # Load the image
    image = cv2.imread(image_path)
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0), (image.shape[1], image.shape[0]))
    # Extract the text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(rotated)

    # Get the bounding boxes of the text regions
    boxes = pytesseract.image_to_boxes(rotated)

    return extracted_text, boxes

#calculate the similarity between two words using the Levenshtein distance algorithm to OCR endpoint
def calculate_similarity(word1, word2):
    """
    Calculates the similarity between two words using the Levenshtein distance algorithm.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - similarity (float): The similarity score between the two words.
    """
    if word1 is None or word2 is None:
        return 0

    else:
        # Convert the words to lowercase
        word1 = word1.lower()
        word2 = word2.lower()

        # Calculate the Levenshtein distance to calculate and compare letter by letter each word
        distance = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            distance[i][0] = i
        for j in range(len(word2) + 1):
            distance[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[i][j] = min(distance[i - 1][j] + 1, distance[i][j - 1] + 1, distance[i - 1][j - 1] + cost)

        # Calculate the similarity score
        max_length = max(len(word1), len(word2))
        similarity = 1 - (distance[len(word1)][len(word2)] / max_length)

        return similarity

# Create option to rotate the image and extract the data using Tesseract OCR
def modify_angle_extract_data(config, image_path, example_name, example_lastname, highest_name_simitality, highest_lastname_simitality):
    """
    Modifies the angle of the image and extracts data using Tesseract OCR.
    Calculates the similarity between the extracted data and example name and last name.
    Returns the highest similarity scores and corresponding names.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
    - image_path (str): The path to the image file.
    - example_name (str): The example name for comparison.
    - example_lastname (str): The example last name for comparison.
    - highest_name_simitality (float): The highest similarity score for name.
    - highest_lastname_simitality (float): The highest similarity score for last name.

    Returns:
    - highest_name_simitality (float): The highest similarity score for name.
    - highest_lastname_simitality (float): The highest similarity score for last name.
    - highest_name (str): The name with the highest similarity score.
    - highest_lastname (str): The last name with the highest similarity score.
    - extracted_text (str): The extracted text from the image.
    - bounding_boxes (list): The bounding boxes of the text regions.
    - name_similarity (float): The similarity score between the extracted name and example name.
    - lastname_similarity (float): The similarity score between the extracted last name and example last name.
    """
    for i in range(0, 360, 5):
        extracted_text, bounding_boxes = extract_data(image_path, angle=-i)
        answer = get_answer_llm(config, extracted_text)
        dictionary = convert_str_to_dict(answer)
        name_similarity = calculate_similarity(dictionary["name"], example_name)
        lastname_similarity = calculate_similarity(dictionary["last_name"], example_lastname)

        if name_similarity > highest_name_simitality:
            highest_name_simitality = name_similarity
            highest_name = dictionary["name"]

        if lastname_similarity > highest_lastname_simitality:
            highest_lastname_simitality = lastname_similarity
            highest_lastname = dictionary["last_name"]

        if dictionary["name"] and name_similarity > 0.9 and lastname_similarity > 0.9:
            break
    return highest_name_simitality, highest_lastname_simitality, highest_name, highest_lastname, extracted_text, bounding_boxes, name_similarity, lastname_similarity

# Get the bounding boxes of a word in a given list of bounding boxes
def get_matching_bounding_boxes(word, bounding_boxes):
    """
    Get the bounding boxes of a word in a given list of bounding boxes.

    Parameters:
    - word (str): The word to search for.
    - bounding_boxes (str): The string representation of the bounding boxes.

    Returns:
    - list: A list of bounding boxes representing the characters of the word.
    - str: "Sequence not found" if the word is not found in the bounding boxes.
    """
    # Convert word to lower case to handle case insensitivity
    word = word.lower()
    # List that represents only the characters of the bounding boxes
    text_extracted = []
    # List of the bounding boxes represented by each character, including only the coordinates of the bounding boxes
    total_data = []
    for box in bounding_boxes.split('\n'):
        if box is not None:
            text_extracted.append(box.split(' ')[0].lower())
            total_data.append(box.split(' ')[1:])

    letters_str = ''.join(text_extracted)
    start_index = letters_str.find(word)
    if start_index == -1:
        return "Sequence not found"
    # The end index is the start index plus the length of the word
    end_index = start_index + len(word)
    
    list_bb_word = total_data[start_index:end_index]
    return list_bb_word

# convert a file in png the same format to extract data
def convert_to_png(file_path):
    """
    Converts a file to PNG format.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - new_path (str): The path to the converted PNG file.
    - text (None): None.

    Raises:
    - None.
    """
    img = Image.open(file_path)
    new_path = file_path.rsplit('.', 1)[0] + '.png'
    img.save(new_path, format='PNG')
    text = None
    return new_path, text

# Extract text from a PDF file using PyMuPDF and convert it to images if the outcome of PDF is empty to extract text using tesseract OCR
def handle_pdf(file_path):
    """
    Handles a PDF file by extracting text from it.

    Parameters:
    - file_path (str): The path to the PDF file.

    Returns:
    - image_paths (list): A list of paths to the extracted images if the PDF is empty.
    - text (str): The extracted text from the PDF if it is not empty.
    """
    # Extract text from PDF
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()

    if len(text) == 0:
        # Convert PDF to image
        images = convert_from_path(file_path)
        image_paths = []
        base_path = file_path.rsplit('.', 1)[0]
        for i, image in enumerate(images):
            image_path = f"{base_path}_page_{i}.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)

        return image_paths, text
    else:
        return None, text

# Process the file to extract the text and bounding boxes
def process_file(file_path, name, lastname, config, text=None):
    highest_name_simitality = 0
    highest_lastname_simitality = 0
    if text:    
        answer = get_answer_llm(config, text, 'prompt_template_1')
        dictionary = convert_str_to_dict(answer)
        name_similarity = calculate_similarity(dictionary["name"], name)
        lastname_similarity = calculate_similarity(dictionary["last_name"], lastname)

        return {"Name_extracted": dictionary["name"], 
                "last_name_extracted": dictionary["last_name"], 
                "Bounding_boxes_name": "No available bounding boxes because the text is provided directly of pdf file", 
                "Bounding_boxes_last_name": "No available bounding boxes because the text is provided directly of pdf file",
                "fuzzy_matching_name": name_similarity * 100,
                "fuzzy_matching_lastname": lastname_similarity * 100}

    else:
        extracted_text, bounding_boxes = extract_data(file_path)
        if not extracted_text:
            highest_name_simitality, highest_lastname_simitality, highest_name, highest_lastname, extracted_text, bounding_boxes, name_similarity, lastname_similarity = modify_angle_extract_data(config, file_path, name, lastname, highest_name_simitality, highest_lastname_simitality)
            if name_similarity > 0.9 and lastname_similarity > 0.9:
                return {"Name_extracted": highest_name, 
                        "last_name_extracted": highest_lastname, 
                        "Bounding_boxes_name": get_matching_bounding_boxes(highest_name, bounding_boxes), 
                        "Bounding_boxes_last_name": get_matching_bounding_boxes(highest_lastname, bounding_boxes),
                        "fuzzy_matching_name": name_similarity * 100,
                        "fuzzy_matching_lastname": lastname_similarity * 100}
            else:
                return {"outcome": f"the highest name similarity is: {highest_name_simitality} with the extract name {highest_name}  and the highest lastname similarity is: {highest_lastname_simitality} with the extract lastname {highest_lastname}"}
        else:
            answer = get_answer_llm(config, extracted_text)
            dictionary = convert_str_to_dict(answer)
            name_similarity = calculate_similarity(dictionary["name"], name)
            lastname_similarity = calculate_similarity(dictionary["last_name"], lastname)
            if name_similarity > 0.9 and lastname_similarity > 0.9:
                return {"Name_extracted": dictionary["name"], 
                        "last_name_extracted": dictionary["last_name"], 
                        "Bounding_boxes_name": get_matching_bounding_boxes(dictionary["name"], bounding_boxes), 
                        "Bounding_boxes_last_name": get_matching_bounding_boxes(dictionary["last_name"], bounding_boxes),
                        "fuzzy_matching_name": name_similarity * 100,
                        "fuzzy_matching_lastname": lastname_similarity * 100}
            else:
                highest_name_simitality, highest_lastname_simitality, highest_name, highest_lastname, extracted_text, bounding_boxes, name_similarity, lastname_similarity = modify_angle_extract_data(config, file_path, name, lastname, highest_name_simitality, highest_lastname_simitality)
                if name_similarity > 0.9 and lastname_similarity > 0.9:
                    return {"Name_extracted": highest_name, 
                            "last_name_extracted": highest_lastname, 
                            "Bounding_boxes_name": get_matching_bounding_boxes(highest_name, bounding_boxes), 
                            "Bounding_boxes_last_name": get_matching_bounding_boxes(highest_lastname, bounding_boxes),
                            "fuzzy_matching_name": name_similarity * 100,
                            "fuzzy_matching_lastname": lastname_similarity * 100}
                else:
                    return {"outcome": f"the highest name similarity is: {highest_name_simitality} with the extract name {highest_name}  and the highest lastname similarity is: {highest_lastname_simitality} with the extract lastname {highest_lastname}"}


## FUNTIONS TO CREATE A RAG 
def normalize_text(s):
    """
    Normalize the text by removing special characters and multiple spaces.
    
    Args:
        s (str): The input text.
    
    Returns:
        str: The normalized text.
    """
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = s.replace("  ", " ")
    s = s.replace("_", "")
    s = s.replace("#", "")
    s = s.strip()
    
    return s

# Create an generate embeddings
def create_embeddings(client, text, tokenizer, model):
    """
    Create embeddings for a given document file.
    
    Args:
        client (OpenAI): The OpenAI client.
        file_path (str): The path to the document file.
    
    Returns:
        tuple: A tuple containing the embedding, tokens, and normalized text. 
               Returns a string when the file format is invalid.
    """
    if len(text) > 0:
        
        # Normalize the text
        normalized_text = normalize_text(text)
        # Tokenize the normalized text
        tokenizer = tiktoken.get_encoding(tokenizer)
        tokens = tokenizer.encode(normalized_text)

        text = list(normalized_text.split(" "))
        embeddings_list = []
        chunk_text = []
        #Create embeddings using OpenAI client
        if len(tokens) > 500:
            total_chunks = len(tokens) // 500
            # text = list(normalized_text.split(" "))
            for i in range(total_chunks):
                embedding = client.embeddings.create(input=' '.join(text[i*500:(i+1)*500]), model=model).data[0].embedding
                embeddings_list.append(embedding)
                chunk_text.append(' '.join(map(str,text[i*500:(i+1)*500])))
        else:
            embedding = client.embeddings.create(input=normalized_text, model=model).data[0].embedding
            embeddings_list.append(embedding)
            chunk_text.append(' '.join(map(str, text)))

        return embeddings_list, tokens, chunk_text
    else:
        return "Error: Invalid file format. Only .docx files are supported.", None, None
    

## Methods to use QDRANT API, create a collection and index a document
# Get all collections in Qdrant just to check if the collection already exists
def get_all_collections(client_vdb):
    """
    Get all collections in Qdrant.

    Args:
        client_vdb (QdrantClient): The Qdrant client.

    Returns:
        list: A list of collection names.

    Raises:
        Exception: If there is an error fetching collections from Qdrant.
    """
    try:
        collections_list = []
        collections = client_vdb.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collections_list.append(c.name)
        return collections_list
    except Exception as e:
        print(f"Error fetching collections from Qdrant: {e}")

# Upload or update a document in the Qdrant vector database with the embeddings and text
def upload_documents(collection_name, client_vdb, embeddings, text):
    """
    Upload or update a document in the Qdrant vector database.

    Args:
        collection_name (str): The name of the collection in Qdrant.
        client_vdb (QdrantClient): The Qdrant client.
        embeddings (list): The embeddings of the document.
        text: List of text divided in each chunk size defined.
        id (int, optional): The ID of the document. Defaults to 0.

    Returns:
        str: A success message indicating the status of the document upload or update.

    Raises:
        Exception: If there is an error during the document upload or update.
    """
    try:
        if collection_name in get_all_collections(client_vdb):
            # Get the information of the collection and the last id
            information = client_vdb.get_collection(collection_name=collection_name)
            last_id = information.points_count

            client_vdb.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(id=last_id+1, vector=embeddings, payload={"Document_text": text})
                ]
            )
            return "Document upload successful"
    
        else:
            # Create a collection with the given name and vector size in Qdrant
            client_vdb.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            client_vdb.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(id=0, vector=embeddings, payload={"Document_text": text})
                ]
            )
            return "Collection creation and document upload successful"
    except Exception as e:
        raise ValueError(f"Error during document upload or update: {e}")

## Methods to use OPENAI API, process text, and get an answer from a Language Model
def get_answer_RAG(config, client_vdb, client_ia, prompt):
    """
    Provides an answer from a Language Model (LLM) based on the similarity between the prompt and documents.

    Parameters:
    - config (dict): A dictionary containing configuration settings.
    - client_vdb: The client for interacting with a Vector Database.
    - client_ia: The client for interacting with an Intelligent Assistant.
    - prompt (str): The prompt to be used for generating the answer.

    Returns:
    - response (str): The generated answer from the LLM.
    """
    # Create an instance of the ChatOpenAI class for the Language Model
    llm = ChatOpenAI(temperature=0, openai_api_key=config['openai']['key'], model_name=config['openai']['llm_model'])

    # Normalize the prompt text
    normalized_prompt = normalize_text(prompt)

    # Create embeddings for the normalized prompt using the Intelligent Assistant client
    embedding = client_ia.embeddings.create(input=normalized_prompt, model=config['openai']['model']).data[0].embedding

    # Search for the best matching document in the Vector Database based on the prompt embedding
    best_document = client_vdb.search(
        collection_name=config['qdrant']['collection'], query_vector=embedding, limit=1
    )

    ## If the best matching document has a score greater than or equal to the threshold, generate the answer
    if best_document[0].score >= config['llm']['threshold']:
        
        # Get the text of the best matching document
        text = best_document[0].payload['Document_text']

        # Fill the prompt template with the question and the content of the best matching document
        prompt_template = PromptTemplate.from_template(config['llm']['prompt_template_RAG'])
        filled_prompt = prompt_template.format(question=normalized_prompt, context=text)

        # Generate the answer from the Language Model using the filled prompt
        response = llm.call_as_llm(filled_prompt)
        response = normalize_text(response)
        return response
    else:
        return "I don´t have data that matching with a document according with your search."
