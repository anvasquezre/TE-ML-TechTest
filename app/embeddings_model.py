from sentence_transformers import SentenceTransformer
from config.config import logger

# Load the Sentence-BERT model
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Sentence-BERT model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Sentence-BERT model: {e}")


def encode_text(text):
    """Encodes text using the loaded Sentence-BERT model and returns the embedding."""
    try:
        embeddings = model.encode([text])
        logger.info("Text successfully encoded into embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
