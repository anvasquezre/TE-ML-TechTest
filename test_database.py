# Script de prueba
from app.services.database_manager import (
    list_collections, count_points, get_point, get_random_point
)

pdf_path = 'data/AS-IS-Residential-Contract-for-Sale-And-Purchase (1).pdf'
collection_name = 'contract_text_embeddings'

# Procesa el contrato, lo parte en chuncks lo vectoriza y
# almacena en la colección indicada
# process_and_store_text(pdf_path, collection_name)

# Listar todas las colecciones en la base de datos
collections = list_collections()
print("Colecciones:", collections)

# Contar puntos en una colección específica
collection_name = 'contract_text_embeddings'
point_count = count_points(collection_name)
print(f"Número de puntos en la colección '{collection_name}':", point_count)

# Obtener un punto específico por su ID
point_id = 8
point = get_point(collection_name, point_id)
print(f"Punto con ID {point_id}:", point)

