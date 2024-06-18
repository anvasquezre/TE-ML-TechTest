from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
import fitz

# Configuration for Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def partition_text(text, chunk_size=100):
    """
    Partitions text into chunks of specified size.

    Parameters:
    text (str): The text to be partitioned.
    chunk_size (int): The size of each text chunk.

    Returns:
    list: A list of text chunks.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def generate_embeddings(text_chunks):
    """
    Generates embeddings for each text chunk using Sentence-BERT.

    Parameters:
    text_chunks (list): A list of text chunks.

    Returns:
    list: A list of embeddings.
    """
    embeddings = model.encode(text_chunks)
    return embeddings


def create_collection(collection_name):
    """
    Creates a collection in the Qdrant database.

    Parameters:
    collection_name (str): The name of the collection.

    Returns:
    None
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE)
    )


def store_embeddings(collection_name, text_chunks, embeddings):
    """
    Stores embeddings and their metadata in the Qdrant collection.

    Parameters:
    collection_name (str): The name of the collection.
    text_chunks (list): A list of text chunks.
    embeddings (list): A list of embeddings.

    Returns:
    None
    """
    points = [
        rest.PointStruct(id=i, vector=embedding.tolist(), payload={"text": text_chunk})
        for i, (embedding, text_chunk) in enumerate(zip(embeddings, text_chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)


def process_and_store_text(pdf_path, collection_name):
    """
    Processes text from a PDF and stores embeddings in Qdrant.

    Parameters:
    pdf_path (str): The path to the PDF file.
    collection_name (str): The name of the collection.

    Returns:
    None
    """
    # Extract text from the PDF
    document = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(document)):
        page = document[page_num]
        full_text += page.get_text()

    # Partition text and generate embeddings
    text_chunks = partition_text(full_text)
    embeddings = generate_embeddings(text_chunks)

    # Create collection and store embeddings
    create_collection(collection_name)
    store_embeddings(collection_name, text_chunks, embeddings)


#==============================================================
# Function to recover the most relevant text fragments

def query_embeddings(query, collection_name='contract_text_embeddings', top_k=5):
    """
    Queries the Qdrant database to find the most relevant text fragments for a given query.

    Parameters:
    query (str): The query to find relevant text fragments for.
    collection_name (str): The name of the collection to query.
    top_k (int): The number of top results to retrieve.

    Returns:
    list: A list of relevant text fragments.
    """
    query_embedding = model.encode([query])
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        top=top_k
    )

    relevant_texts = [hit.payload['text'] for hit in search_result]
    return relevant_texts


def search_similar_texts(collection_name, query_text, top_k=5):
    """
    Searches for the top K most similar texts in the specified Qdrant collection.

    Parameters:
    collection_name (str): The name of the collection.
    query_text (str): The query text to search for.
    top_k (int): The number of top similar texts to retrieve.

    Returns:
    list: A list of the top K most similar texts and their scores.
    """
    query_vector = model.encode([query_text])[0]
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    similar_texts = []
    for result in search_result:
        similar_texts.append({
            "text": result.payload["text"],
            "score": result.score
        })

    return similar_texts

#===============================================================
# Functions to inspect the contents of the database

def list_collections():
    """
    Lists all collections in the Qdrant database.

    Returns:
    list: A list of collection names.
    """
    collections = client.get_collections()
    return [collection.name for collection in collections.collections]


def count_points(collection_name):
    """
    Counts the number of points in a specific collection.

    Parameters:
    collection_name (str): The name of the collection.

    Returns:
    int: The number of points in the collection.
    """
    response = client.count(collection_name)
    return response.count


def get_point(collection_name, point_id):
    """
    Retrieves a point by its ID from a specific collection.

    Parameters:
    collection_name (str): The name of the collection.
    point_id (int): The ID of the point to retrieve.

    Returns:
    dict: The retrieved point.
    """
    try:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        if points:
            return points[0]
        else:
            return {"message": "Point not found"}
    except Exception as e:
        return {"error": str(e)}


def get_random_point(collection_name):
    """
    Retrieves a random point from a specific collection.

    Parameters:
    collection_name (str): The name of the collection.

    Returns:
    dict: A random point from the collection.
    """
    try:
        points = client.retrieve(
            collection_name=collection_name,
            limit=1
        )
        if points:
            return points[0]
        else:
            return {"message": "No points found"}
    except Exception as e:
        return {"error": str(e)}
