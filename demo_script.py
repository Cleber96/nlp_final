# demo_script.py
import os
import logging
import shutil # Importar shutil para el borrado recursivo

# Configurar el logging para ver los mensajes de los módulos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar los componentes del sistema RAG
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
# CAMBIO CLAVE: Vamos a usar el LLMModel que creamos para GPT4All
# para la demo rápida, o adaptar el actual para cargar GPT4All.
# Necesitamos que LLMModel sea flexible para cargar GPT4All o HuggingFacePipeline.
# Para simplificar la demo, podemos definir un LLMModel_Demo especial o ajustar el existente.
# Por simplicidad para esta demo rápida, vamos a re-introducir la lógica de carga de GPT4All aquí
# o ajustar LLMModel.py para que pueda manejar ambos fácilmente.
# La opción más rápida es añadir la lógica GPT4All directamente para la demo.

# Para esta demo rápida, usaremos una versión simplificada del LLMModel para GPT4All.
# Opcional: Podríamos hacer que LLMModel.py soporte ambos modos (GPT4All y HF)
# pero eso complicaría LLMModel.py. Para esta demo rápida, es mejor esto:

from gpt4all import GPT4All as GPT4AllModelLoader # Importar GPT4All para cargar el modelo
from langchain_community.llms.gpt4all import GPT4All as LangChainGPT4All # Importar el wrapper de LangChain para GPT4All
# NOTA: Asegúrate que tus embeddings y retrievers funcionen con langchain-community.

# Esto significa que src/rag_system/llm_model.py no se usará para esta demo rápida.
# Si quieres mantener una única clase LLMModel, tendríamos que refactorizarla para
# que acepte diferentes tipos de modelos (GPT4All, HF, etc.) basados en los parámetros.
# Por ahora, para que sea lo más rápido de probar:


from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain


# --- Configuración para la DEMO RÁPIDA (usando GPT4All) ---
DOCS_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index"

# --- CAMBIOS AQUI PARA USAR GPT4ALL PEQUEÑO ---
GPT4ALL_DEMO_MODEL_NAME = "orca-mini-3b-gguf2.q4_0.gguf" # Un modelo GPT4All muy ligero
GPT4ALL_DEMO_MODEL_PATH = f"models/{GPT4ALL_DEMO_MODEL_NAME}" # Ruta local
K_VALUE_DEMO = 3 # Un valor de k para la demostración

def run_demo():
    print("\n--- INICIANDO DEMOSTRACIÓN RÁPIDA DEL SISTEMA RAG ---")
    print(f"**ATENCIÓN**: Esta demo utiliza el modelo GPT4All '{GPT4ALL_DEMO_MODEL_NAME}' para una ejecución rápida.")

    # 1. Ingesta y Creación/Carga de FAISS
    print("\nPASO 1: Ingesta de Documentos y Creación/Carga del Vector Store (FAISS)")
    embeddings_generator = EmbeddingsGenerator()
    embeddings = embeddings_generator.embeddings
    faiss_store = FAISSStore(embeddings=embeddings)

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_store.load_local(FAISS_INDEX_PATH)
            print(f"Índice FAISS cargado exitosamente desde {FAISS_INDEX_PATH}.")
        except Exception as e:
            print(f"Error al cargar el índice FAISS: {e}. Procediendo a crearlo.")
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
            # gpt4all.GPT4All maneja la descarga si no existe
            gpt4all_instance = GPT4AllModelLoader(model_name=GPT4ALL_DEMO_MODEL_NAME, model_path=model_dir, allow_download=True)
            # El constructor ya se asegura de que esté en el model_path, no necesitamos renombrar.
            print(f"Modelo GPT4All '{GPT4ALL_DEMO_MODEL_NAME}' descargado con éxito.")
        except Exception as e:
            print(f"ERROR: No se pudo descargar el modelo GPT4all. Asegúrate de tener conexión a internet o descárgalo manualmente en {GPT4ALL_DEMO_MODEL_PATH}. Error: {e}")
            return # Salir de la demo si no se puede cargar el LLM

    # Inicializar el LLM de LangChain con el modelo GPT4All
    llm = LangChainGPT4All(
        model=GPT4ALL_DEMO_MODEL_PATH,
        max_tokens=512, # Ajustar la longitud máxima de la respuesta
        temp=0.1,       # Temperatura baja para respuestas más directas
        verbose=False   # Opcional: para reducir la verbosidad de GPT4All
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