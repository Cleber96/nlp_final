import argparse
import os
import logging
from typing import List

# Importar módulos del proyecto
from data_ingestion.document_loader import DocumentLoader
from data_ingestion.text_splitter import TextSplitter
from data_ingestion.data_processor import DataProcessor # Asumimos que DataProcessor orquesta estos
from vector_store.embeddings_generator import EmbeddingsGenerator
from vector_store.faiss_store import FAISSStore
# from vector_store.vector_utils import VectorUtils # Si hay funciones auxiliares en vector_utils
from rag_system.prompt_template import PromptTemplateManager
from rag_system.llm_model import LLMModel
from rag_system.retriever import CustomRetriever
from rag_system.rag_chain import RAGChain
from rag_system.config import (
    DOCS_DIR, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME, LLM_MODEL_PATH, LLM_MODEL_NAME, RETRIEVER_K
)

# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Clase principal para orquestar el sistema RAG (Retrieval-Augmented Generation).
    Maneja la ingesta de datos, la creación/carga del VectorStore,
    la inicialización del LLM y la ejecución de la cadena RAG.
    """
    def __init__(self):
        """
        Inicializa los componentes clave del sistema RAG.
        Los componentes se inicializan solo cuando se necesitan.
        """
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embeddings_generator = EmbeddingsGenerator(model_name=EMBEDDING_MODEL_NAME)
        self.faiss_store = FAISSStore(embeddings=self.embeddings_generator.embeddings)
        
        self.llm_model = None
        self.retriever = None
        self.rag_chain = None

        logger.info("Sistema RAG inicializado. Componentes cargados bajo demanda.")

    def ingest_documents(self, docs_path: str = DOCS_DIR):
        """
        Carga, divide y procesa documentos desde el directorio especificado,
        luego crea y guarda un índice FAISS.
        """
        logger.info(f"Iniciando ingesta de documentos desde: {docs_path}")

        all_documents = []
        for root, _, files in os.walk(docs_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    if file_name.lower().endswith(".pdf"):
                        logger.info(f"Cargando PDF: {file_name}")
                        docs = self.document_loader.load_pdf(file_path)
                        all_documents.extend(docs)
                    elif file_name.lower().endswith((".md", ".markdown")):
                        logger.info(f"Cargando Markdown: {file_name}")
                        docs = self.document_loader.load_markdown(file_path)
                        all_documents.extend(docs)
                    else:
                        logger.warning(f"Tipo de archivo no soportado, omitiendo: {file_name}")
                except Exception as e:
                    logger.error(f"Error al cargar el archivo {file_name}: {e}")
        
        if not all_documents:
            logger.warning(f"No se encontraron documentos soportados en {docs_path}. No se creará el índice FAISS.")
            return

        logger.info(f"Total de documentos cargados: {len(all_documents)}")
        
        logger.info("Dividiendo documentos en fragmentos (chunks)...")
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"Total de fragmentos generados: {len(chunks)}")

        logger.info("Creando y guardando el VectorStore FAISS...")
        try:
            self.faiss_store.create_from_documents(chunks, faiss_path=FAISS_INDEX_PATH)
            logger.info(f"Proceso de ingesta completado. Índice FAISS guardado en {FAISS_INDEX_PATH}.")
        except Exception as e:
            logger.error(f"Error al crear el VectorStore FAISS: {e}")

    def initialize_qa_system(self):
        """
        Carga el VectorStore FAISS existente, inicializa el LLM y construye la cadena RAG.
        """
        logger.info("Inicializando sistema de QA...")
        
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.error(f"Índice FAISS no encontrado en {FAISS_INDEX_PATH}. Por favor, ejecute la ingesta de datos primero (`python src/main.py --ingest`).")
            raise FileNotFoundError(f"Índice FAISS no encontrado: {FAISS_INDEX_PATH}")

        try:
            logger.info(f"Cargando VectorStore FAISS desde: {FAISS_INDEX_PATH}")
            self.faiss_store.load_local(faiss_path=FAISS_INDEX_PATH)
            
            logger.info("Inicializando LLM (GPT-NeoX o sustituto)...")
            self.llm_model = LLMModel(model_path=LLM_MODEL_PATH, model_name=LLM_MODEL_NAME)
            llm_instance = self.llm_model.get_llm()

            logger.info(f"Configurando Retriever con k={RETRIEVER_K}...")
            self.retriever = CustomRetriever(faiss_path=FAISS_INDEX_PATH, 
                                             embeddings=self.embeddings_generator.embeddings, 
                                             k=RETRIEVER_K).get_langchain_retriever()
            
            logger.info("Obteniendo template de prompt...")
            prompt_manager = PromptTemplateManager()
            qa_prompt = prompt_manager.get_qa_prompt()

            logger.info("Construyendo cadena RAG...")
            self.rag_chain = RAGChain(retriever=self.retriever, llm=llm_instance, prompt=qa_prompt)
            logger.info("Sistema de QA inicializado y listo para responder preguntas.")

        except Exception as e:
            logger.critical(f"Error fatal al inicializar el sistema de QA: {e}")
            raise

    def query(self, question: str) -> str:
        """
        Ejecuta una consulta contra el sistema RAG y retorna la respuesta.

        Args:
            question (str): La pregunta del usuario.

        Returns:
            str: La respuesta generada por el sistema RAG.
        """
        if self.rag_chain is None:
            logger.error("La cadena RAG no ha sido inicializada. Ejecute initialize_qa_system() primero.")
            return "Error: El sistema QA no está listo."
        
        logger.info(f"Recibida pregunta: '{question}'")
        try:
            response = self.rag_chain.invoke(question)
            logger.info(f"Respuesta generada: '{response}'")
            return response
        except Exception as e:
            logger.error(f"Error al invocar la cadena RAG para la pregunta '{question}': {e}")
            return "Lo siento, hubo un error al procesar tu pregunta."

def main():
    """
    Función principal para parsear argumentos y ejecutar el sistema RAG.
    """
    parser = argparse.ArgumentParser(description="Sistema RAG para preguntas y respuestas sobre documentación técnica.")
    parser.add_argument(
        "--ingest", 
        action="store_true", 
        help="Ejecutar el pipeline de ingesta de documentos y crear el índice FAISS."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Proporcionar una pregunta para el sistema RAG. (Requiere que el índice FAISS exista o se haya ingestado primero)."
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Iniciar un modo interactivo de preguntas y respuestas."
    )
    parser.add_argument(
        "--docs_path",
        type=str,
        default=DOCS_DIR,
        help=f"Ruta al directorio de documentos para la ingesta. Por defecto: {DOCS_DIR}"
    )

    args = parser.parse_args()

    rag_system = RAGSystem()

    if args.ingest:
        rag_system.ingest_documents(docs_path=args.docs_path)
    elif args.query or args.interactive:
        try:
            rag_system.initialize_qa_system()
            if args.query:
                response = rag_system.query(args.query)
                print("\n--- Respuesta del Sistema RAG ---")
                print(response)
                print("----------------------------------")
            
            if args.interactive:
                print("\n--- Modo Interactivo del Sistema RAG ---")
                print("Escribe tu pregunta o 'salir' para terminar.")
                while True:
                    user_question = input("\nTu pregunta: ")
                    if user_question.lower() == 'salir':
                        print("Saliendo del modo interactivo. ¡Adiós!")
                        break
                    if not user_question.strip():
                        print("Por favor, introduce una pregunta válida.")
                        continue
                    
                    response = rag_system.query(user_question)
                    print("\nRespuesta: " + response)
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Por favor, ejecute `python src/main.py --ingest` primero para crear el índice FAISS.")
        except Exception as e:
            logger.critical(f"Ha ocurrido un error inesperado en el sistema de QA: {e}")
            
    else:
        print("Por favor, especifique una acción: --ingest, --query <pregunta>, o --interactive.")
        parser.print_help()

if __name__ == "__main__":
    # Asegúrate de que las rutas relativas funcionen correctamente
    # Modifica el directorio de trabajo si es necesario para acceder a 'docs/' y 'faiss_index/'
    # Esto es una consideración importante si main.py se ejecuta desde un directorio diferente
    # Por ejemplo, si ejecutas desde 'proyecto8_rag/', y DOCS_DIR es 'docs/':
    # os.chdir(os.path.dirname(os.path.abspath(__file__))) # Cambia al directorio src/
    # os.chdir('..') # Luego sube al directorio raíz del proyecto
    
    # Para simplicidad en este ejemplo, asumimos que se ejecuta desde el raíz del proyecto.
    # Si no es así, las rutas en config.py deberán ser absolutas o manejarse dinámicamente.
    main()