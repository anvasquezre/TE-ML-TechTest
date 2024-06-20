import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config.config import DATABASE_DIR, logger

# Initialize the vector database client with persistence
client = chromadb.PersistentClient(path=str(DATABASE_DIR))

# Define the default embedding function using Chroma's built-in SentenceTransformer
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# Function to create or get the collection with the embedding function
def get_or_create_collection(name="pdf_text"):
    try:
        collection = client.get_or_create_collection(
            name=name, embedding_function=default_ef, metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        logger.error(f"Error creating or getting collection: {e}")
        return None


# Create or get the collection
collection = get_or_create_collection()


def store_text_in_vector_db(doc_id, text):
    try:
        if collection is None:
            raise ValueError("Collection is not initialized.")
        collection.add(documents=[text], ids=[doc_id])
        logger.info("Text successfully stored in database.")
    except Exception as e:
        logger.error(f"Error storing text in vector database: {e}")


def query_vector_db(query_text):
    try:
        if collection is None:
            raise ValueError("Collection is not initialized.")
        # Embed the query text
        query_embedding = default_ef([query_text])
        # Perform the query
        results = collection.query(
            query_embeddings=query_embedding, n_results=5
        )  # Adjust limit as needed
        logger.info(f"Query results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        return None


def list_collections():
    try:
        collections = client.list_collections()
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []


def get_collection_info(collection_name):
    try:
        coll = client.get_collection(collection_name)
        info = {
            "name": coll.name,
            "metadata": coll.metadata,
            "document_count": coll.count(),
        }
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return None
