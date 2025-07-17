# src/data_ingestion/run_ingestion.py

import os
import sys
import logging

# --- IMPORTANTE: Ajustar el sys.path para importaciones relativas al proyecto ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..', '..')
sys.path.insert(0, project_root)
# --- FIN DEL AJUSTE DE sys.path ---

from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
# from src.utils import setup_logging # Si tienes setup_logging en utils.py, descomenta

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
# ¡CAMBIO AQUÍ! Si tus documentos están en final_nlp/data/docs, entonces la ruta relativa a la raíz es 'data/docs'
DATA_DIR_RELATIVE_TO_PROJECT_ROOT = "docs"  
FAISS_INDEX_DIR = "vector_store/faiss_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    logger.info("Iniciando el proceso de ingesta de datos para crear/actualizar el índice FAISS...")
    logger.info("DocumentLoader inicializado.")

    loader = DocumentLoader() 

    # --- CAMBIO CLAVE AQUÍ: Construir la ruta completa al directorio de datos ---
    full_data_path = os.path.join(project_root, DATA_DIR_RELATIVE_TO_PROJECT_ROOT)
    
    if not os.path.exists(full_data_path):
        logger.error(f"El directorio de datos no existe: {full_data_path}. Asegúrate de que la carpeta 'data/docs' exista y contenga tus documentos.")
        return

    file_paths_to_load = []
    for root, _, files in os.walk(full_data_path):
        for file_name in files:
            if file_name.endswith((".pdf", ".md")): 
                file_paths_to_load.append(os.path.join(root, file_name))

    if not file_paths_to_load:
        logger.warning(f"No se encontraron archivos PDF o Markdown en '{full_data_path}'. Asegúrate de que los archivos estén allí.")
        return

    logger.info(f"Se encontraron {len(file_paths_to_load)} archivos para cargar en '{full_data_path}'.")

    # 1. Cargar documentos usando el método load_documents_from_paths
    raw_documents = loader.load_documents_from_paths(file_paths_to_load)
    logger.info(f"Cargados {len(raw_documents)} documentos raw de '{full_data_path}'.")

    # 2. Dividir documentos en chunks
    text_splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(raw_documents)
    logger.info(f"Documentos divididos en {len(chunks)} chunks.")

    # 3. Inicializar el generador de embeddings
    try:
        embeddings_generator = EmbeddingsGenerator(model_name=EMBEDDING_MODEL_NAME)
        embeddings = embeddings_generator.get_embeddings()
        logger.info(f"Modelo de embeddings '{EMBEDDING_MODEL_NAME}' cargado exitosamente.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo de embeddings: {e}")
        logger.error("Asegúrate de que el modelo esté disponible o tengas conexión a internet para descargarlo.")
        return

    # 4. Crear o actualizar la base de datos vectorial FAISS
    faiss_index_full_path = os.path.join(project_root, FAISS_INDEX_DIR)
    os.makedirs(faiss_index_full_path, exist_ok=True)
    
    faiss_store = FAISSStore(embeddings=embeddings, faiss_path=faiss_index_full_path)
    
    if os.path.exists(os.path.join(faiss_index_full_path, "index.faiss")):
        logger.info("Se encontró un índice FAISS existente. Cargando y agregando nuevos chunks...")
        faiss_store.add_documents(chunks) 
    else:
        logger.info("No se encontró un índice FAISS existente. Creando uno nuevo...")
        faiss_store.create_and_save_vector_store(chunks)

    logger.info(f"Proceso de ingesta y guardado/actualización de índice FAISS completado en '{faiss_index_full_path}'.")
    logger.info("¡Ingesta de datos finalizada con éxito!")

if __name__ == "__main__":
    main()