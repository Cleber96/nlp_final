# demo_script.py
import os
import logging

# Configurar el logging para ver los mensajes de los módulos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar los componentes del sistema RAG
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
from src.rag_system.llm_model import LLMModel
from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain

# --- Configuración (ajusta según tu bench_qa_performance.py) ---
DOCS_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_NAME = "EleutherAI/gpt-neox-20b"
LLM_MODEL_LOCAL_PATH = "models/EleutherAI/gpt-neox-20b"
LLM_DEVICE = "auto" # "auto", "cpu", "cuda"
K_VALUE_DEMO = 3 # Un valor de k para la demostración

def run_demo():
    print("\n--- INICIANDO DEMOSTRACIÓN DEL SISTEMA RAG ---")

    # 1. Ingesta y Creación/Carga de FAISS
    print("\nPASO 1: Ingesta de Documentos y Creación/Carga del Vector Store (FAISS)")
    embeddings_generator = EmbeddingsGenerator()
    embeddings = embeddings_generator.embeddings
    faiss_store = FAISSStore(embeddings=embeddings)

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_store.load_local(FAISS_INDEX_PATH)
            print(f"ndice FAISS cargado exitosamente desde {FAISS_INDEX_PATH}.")
        except Exception as e:
            print(f"Error al cargar el índice FAISS: {e}. Procediendo a crearlo.")
            # Borrar el índice defectuoso si existe
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
            create_faiss_index(faiss_store)
    else:
        print(f"Índice FAISS no encontrado en {FAISS_INDEX_PATH}. Procediendo a crearlo.")
        create_faiss_index(faiss_store)

    # 2. Configuración del Sistema RAG
    print(f"\nPASO 2: Configurando el sistema RAG con k={K_VALUE_DEMO}")
    retriever = CustomRetriever(faiss_path=FAISS_INDEX_PATH, embeddings=embeddings, k=K_VALUE_DEMO)
    langchain_retriever = retriever.get_langchain_retriever()
    print("Retriever configurado.")

    llm_model = LLMModel(model_name=LLM_MODEL_NAME, model_path=LLM_MODEL_LOCAL_PATH, device=LLM_DEVICE)
    llm = llm_model.get_llm()
    print(f"LLM ({LLM_MODEL_NAME}) cargado.")

    prompt_manager = PromptTemplateManager()
    qa_prompt = prompt_manager.get_qa_prompt()
    print("Prompt template cargado.")

    rag_chain = RAGChain(retriever=langchain_retriever, llm=llm, prompt=qa_prompt)
    print("Cadena RAG construida.")

    # 3. Pregunta de Demostración
    print("\nPASO 3: Demostración de Pregunta y Respuesta (RAG en acción)")
    print("Ingresa 'salir' para terminar la demo.")

    while True:
        user_question = input("\nTu pregunta al RAG: ")
        if user_question.lower() == 'salir':
            print("Saliendo de la demostración interactiva.")
            break
        
        print(f"Procesando pregunta: '{user_question}'...")
        try:
            # Obtener la respuesta del RAG
            response = rag_chain.invoke(user_question)
            print("\n--- RESPUESTA GENERADA ---")
            print(response)
            print("-------------------------")
        except Exception as e:
            print(f"Error al generar respuesta: {e}")

def create_faiss_index(faiss_store):
    print(f"Cargando y dividiendo documentos desde {DOCS_PATH}...")
    document_loader = DocumentLoader()
    text_splitter = TextSplitter()
    
    all_documents = []
    for root, _, files in os.walk(DOCS_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                docs = document_loader.load_pdf(file_path)
                if docs:
                    all_documents.extend(docs)
                else:
                    logger.warning(f"No se pudieron cargar documentos de {file_path}. ¿Está vacío o corrupto?")
            elif file.endswith(".md"):
                docs = document_loader.load_markdown(file_path)
                if docs:
                    all_documents.extend(docs)
                else:
                    logger.warning(f"No se pudieron cargar documentos de {file_path}. ¿Está vacío o corrupto?")
    
    if not all_documents:
        raise ValueError(f"No se encontraron documentos válidos en {DOCS_PATH}. Asegúrate de tener archivos PDF o Markdown con contenido.")

    chunks = text_splitter.split_documents(all_documents)
    print(f"Documentos cargados y divididos en {len(chunks)} fragmentos.")
    
    faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH)
    print("Índice FAISS creado y guardado.")

if __name__ == "__main__":
    run_demo()