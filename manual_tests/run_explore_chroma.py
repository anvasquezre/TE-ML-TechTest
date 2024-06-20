from app.vector_database import list_collections, get_collection_info
from config.config import logger

if __name__ == "__main__":
    # List all collections
    collections = list_collections()
    logger.info(f"Collections: {collections}")

    # Get detailed info for each collection
    for collection in collections:
        collection_name = collection.name
        info = get_collection_info(collection_name)
        logger.info(f"Collection info for '{collection_name}': {info}")
