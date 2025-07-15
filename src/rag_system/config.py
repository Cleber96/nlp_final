import os
import logging

# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """
    Clase para gestionar la configuración global del sistema RAG.
    Define rutas de archivos, nombres de modelos y otros parámetros importantes.
    """
    # Rutas base
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index") # Directorio para guardar el índice FAISS

    # Configuración de Ingesta de Datos
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Configuración de Embeddings
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # Puedes ajustar el dispositivo para los embeddings: 'cpu' o 'cuda'
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'} 
    EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}

    # Configuración del LLM
    # Para GPT-NeoX-20B-sft, necesitarías descargar el modelo o configurar un endpoint.
    # Aquí se usa GPT4All como un ejemplo más accesible para pruebas locales.
    LLM_MODEL_NAME = "GPT4All" # O "GPT-NeoX-20B-sft" si lo configuras
    # Ruta al archivo del modelo GPT4All (ej. ggml-gpt4all-j-v1.3-groovy.bin)
    # DEBES DESCARGAR ESTE ARCHIVO MANUALMENTE Y ESPECIFICAR LA RUTA CORRECTA.
    # Por ejemplo, puedes descargarlo de https://gpt4all.io/index.html
    LLM_MODEL_PATH = os.path.join(DATA_DIR, "ggml-gpt4all-j-v1.3-groovy.bin") 
    LLM_ALLOW_DOWNLOAD = False # No permitir descarga automática para modelos grandes

    # Configuración del Retriever
    RETRIEVER_K = 4 # Número de documentos a recuperar

    # Configuración de Evaluación (opcional, para benchmarks)
    QA_TEST_SET_PATH = os.path.join(DATA_DIR, "qa_test_set.json")
    BENCHMARK_RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")

    def __init__(self):
        # Asegurarse de que los directorios necesarios existan
        os.makedirs(self.FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(self.BENCHMARK_RESULTS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        logger.info("Configuración cargada y directorios verificados.")
        logger.info(f"Ruta del índice FAISS: {self.FAISS_INDEX_DIR}")
        logger.info(f"Ruta del modelo LLM: {self.LLM_MODEL_PATH}")

# Instancia global de configuración
config = Config()

if __name__ == "__main__":
    print("--- Probando Config ---")
    print(f"Ruta raíz del proyecto: {config.PROJECT_ROOT}")
    print(f"Directorio de documentos: {config.DOCS_DIR}")
    print(f"Directorio del índice FAISS: {config.FAISS_INDEX_DIR}")
    print(f"Tamaño de chunk: {config.CHUNK_SIZE}")
    print(f"Nombre del modelo de embeddings: {config.EMBEDDING_MODEL_NAME}")
    print(f"Nombre del modelo LLM: {config.LLM_MODEL_NAME}")
    print(f"Ruta del modelo LLM: {config.LLM_MODEL_PATH}")
    print(f"Número de documentos a recuperar (k): {config.RETRIEVER_K}")

    # Intentar crear un archivo de prueba para verificar permisos
    try:
        test_file_path = os.path.join(config.FAISS_INDEX_DIR, "test.txt")
        with open(test_file_path, "w") as f:
            f.write("Test file")
        print(f"Archivo de prueba creado en {test_file_path}")
        os.remove(test_file_path)
        print("Archivo de prueba eliminado.")
    except Exception as e:
        print(f"Error al crear/eliminar archivo de prueba en {config.FAISS_INDEX_DIR}: {e}")