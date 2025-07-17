# demo_script.py
import os
import sys # <--- ¡Añadir sys!
import logging
import shutil # Importar shutil para el borrado recursivo

# --- IMPORTANTE: Ajustar el sys.path para importaciones relativas al proyecto ---
# Como este script está en la raíz del proyecto, añadimos la raíz misma al sys.path.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)
# --- FIN DEL AJUSTE DE sys.path ---

# Configurar el logging para ver los mensajes de los módulos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar los componentes del sistema RAG
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
# Importaciones para GPT4All
from gpt4all import GPT4All as GPT4AllModelLoader # Importar GPT4All para cargar el modelo (gpt4all.GPT4All)
from langchain_community.llms.gpt4all import GPT4All as LangChainGPT4All # Importar el wrapper de LangChain para GPT4All
from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain


# --- Configuración para la DEMO RÁPIDA (usando GPT4All) ---
# --- ¡Asegúrate de que estas rutas sean correctas para TU PROYECTO! ---
DOCS_PATH = "data/docs/" # <--- ¡AJUSTADO! Si tus documentos están en final_nlp/data/docs/
FAISS_INDEX_PATH = "vector_store/faiss_index" # <--- ¡AJUSTADO! Para guardar el índice en vector_store/
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # El mismo que usaste para FAISS
CHUNK_SIZE = 500 # <--- Ajuste del TextSplitter
CHUNK_OVERLAP = 50 # <--- Ajuste del TextSplitter

GPT4ALL_DEMO_MODEL_NAME = "orca-mini-3b-gguf2.q4_0.gguf"
# La ruta del modelo es relativa a la raíz del proyecto.
# Asegúrate de que el modelo se descargue o exista en 'final_nlp/models/orca-mini-3b-gguf2.q4_0.gguf'
GPT4ALL_DEMO_MODEL_PATH = os.path.join("models", GPT4ALL_DEMO_MODEL_NAME) 

K_VALUE_DEMO = 3 # Número de documentos más relevantes a recuperar por el retriever

def run_demo():
    print("\n--- INICIANDO DEMOSTRACIÓN RÁPIDA DEL SISTEMA RAG ---")
    print(f"**ATENCIÓN**: Esta demo utiliza el modelo GPT4All '{GPT4ALL_DEMO_MODEL_NAME}' para una ejecución rápida.")

    # 1. Ingesta y Creación/Carga de FAISS
    print("\nPASO 1: Ingesta de Documentos y Creación/Carga del Vector Store (FAISS)")
    
    # Inicializar EmbeddingsGenerator con el nombre del modelo
    embeddings_generator = EmbeddingsGenerator(model_name=EMBEDDING_MODEL_NAME) 
    embeddings = embeddings_generator.get_embeddings() # Asegúrate que tu EmbeddingsGenerator tenga un get_embeddings()
    
    faiss_store = FAISSStore(embeddings=embeddings, faiss_path=FAISS_INDEX_PATH) # Pasa faiss_path al constructor

    # Verificar y crear el directorio para FAISS si no existe
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    # Intentar cargar o crear el índice FAISS
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        try:
            faiss_store.load_local() # Tu load_local probablemente no necesita el path si ya está en el constructor
            print(f"Índice FAISS cargado exitosamente desde {FAISS_INDEX_PATH}.")
        except Exception as e:
            print(f"Error al cargar el índice FAISS desde {FAISS_INDEX_PATH}: {e}. Procediendo a crearlo.")
            # Borrar el índice defectuoso si existe
            if os.path.exists(FAISS_INDEX_PATH):
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

    # --- CARGA DEL LLM GPT4ALL PARA LA DEMO RÁPIDA ---
    print(f"Cargando modelo LLM GPT4All: {GPT4ALL_DEMO_MODEL_NAME}...")
    model_dir = os.path.dirname(GPT4ALL_DEMO_MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True) # Asegurarse de que el directorio exista

    if not os.path.exists(GPT4ALL_DEMO_MODEL_PATH):
        print(f"Modelo GPT4All no encontrado en {GPT4ALL_DEMO_MODEL_PATH}. Intentando descargarlo...")
        try:
            # gpt4all.GPT4All maneja la descarga si no existe y lo guarda en model_dir
            # Solo necesitamos una instancia para la descarga, no para inferencia aquí
            _ = GPT4AllModelLoader(model_name=GPT4ALL_DEMO_MODEL_NAME, model_path=model_dir, allow_download=True)
            print(f"Modelo GPT4All '{GPT4ALL_DEMO_MODEL_NAME}' descargado con éxito en {model_dir}.")
        except Exception as e:
            print(f"ERROR: No se pudo descargar el modelo GPT4all. Asegúrate de tener conexión a internet o descárgalo manualmente en {GPT4ALL_DEMO_MODEL_PATH}. Error: {e}")
            return # Salir de la demo si no se puede cargar el LLM
    else:
        print(f"Modelo GPT4all '{GPT4ALL_DEMO_MODEL_NAME}' encontrado localmente.")


    # Inicializar el LLM de LangChain con el modelo GPT4All
    llm = LangChainGPT4All(
        model=GPT4ALL_DEMO_MODEL_PATH,
        max_tokens=512, # Ajustar la longitud máxima de la respuesta
        temp=0.1,        # Temperatura baja para respuestas más directas
        verbose=False    # Opcional: para reducir la verbosidad de GPT4All
    )
    print(f"LLM ({GPT4ALL_DEMO_MODEL_NAME}) cargado para la demo.")

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
            response = rag_chain.invoke({"query": user_question}) # <--- Asegúrate que RAGChain.invoke espera un diccionario
            # La respuesta de una cadena RetrievalQA o similar suele estar en 'result'
            print("\n--- RESPUESTA GENERADA ---")
            print(response.get("result", response)) # Accede a 'result' si existe, si no, imprime la respuesta completa
            # Opcional: Mostrar documentos fuente si tu RAGChain los devuelve
            if "source_documents" in response:
                print("\n--- DOCUMENTOS FUENTE ---")
                for i, doc in enumerate(response["source_documents"]):
                    print(f"  Documento {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}):")
                    print(f"    Contenido: {doc.page_content[:250]}...")
                    print("-" * 25)
            print("-------------------------")

        except Exception as e:
            print(f"Error al generar respuesta: {e}")
            logger.exception("Detalles del error al generar respuesta:") # Para ver el traceback completo

def create_faiss_index(faiss_store: FAISSStore): # Tipo de faiss_store para claridad
    print(f"Cargando y dividiendo documentos desde {DOCS_PATH}...")
    
    document_loader = DocumentLoader() # No necesita argumentos si no tiene __init__
    # Pasa los parámetros al constructor de TextSplitter
    text_splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) 
    
    all_documents = []
    # Usar os.path.join para construir la ruta completa al directorio de docs
    full_docs_path = os.path.join(current_script_dir, DOCS_PATH) 
    
    if not os.path.exists(full_docs_path):
        raise ValueError(f"El directorio de documentos no existe: {full_docs_path}. Crea la carpeta 'data/docs' y coloca tus archivos.")

    for root, _, files in os.walk(full_docs_path):
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
            else:
                logger.info(f"Saltando archivo no soportado o reconocido: {file_path}")
    
    if not all_documents:
        raise ValueError(f"No se encontraron documentos válidos en {full_docs_path}. Asegúrate de tener archivos PDF o Markdown con contenido.")

    chunks = text_splitter.split_documents(all_documents)
    print(f"Documentos cargados y divididos en {len(chunks)} fragmentos.")
    
    # create_from_documents recibe el path para guardar el índice
    faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH) 
    print(f"Índice FAISS creado y guardado en {FAISS_INDEX_PATH}.")

if __name__ == "__main__":
    run_demo()