import openai
from qdrant_client import QdrantClient

# Configuración de la API de OpenAI
openai.api_key = 'your-openai-api-key'

# Inicialización del cliente de Qdrant
qdrant_client = QdrantClient("http://localhost:6333")
