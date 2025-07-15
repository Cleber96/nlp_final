import streamlit as st
import os
import logging

# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar los componentes de tu sistema RAG
# Asegúrate de que las rutas relativas funcionen o ajusta los imports
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
from src.rag_system.llm_model import LLMModel
from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain

# --- Configuración (igual que en demo_script.py y bench_qa_performance.py) ---
DOCS_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_NAME = "EleutherAI/gpt-neox-20b"
LLM_MODEL_LOCAL_PATH = "models/EleutherAI/gpt-neox-20b"
LLM_DEVICE = "auto"
K_VALUE_DEMO = 3 # Un valor de k para la demostración

# Cargar el sistema RAG una sola vez (cachear para Streamlit)
@st.cache_resource
def load_rag_system():
    print("Cargando sistema RAG para la aplicación web...")
    # Lógica de carga de FAISS o creación
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
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)
            # Crear el índice si no se pudo cargar o no existe
            all_documents = []
            for root, _, files in os.walk(DOCS_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".pdf"):
                        all_documents.extend(DocumentLoader().load_pdf(file_path))
                    elif file.endswith(".md"):
                        all_documents.extend(DocumentLoader().load_markdown(file_path))
            if not all_documents:
                st.error(f"No se encontraron documentos válidos en {DOCS_PATH}. Asegúrate de tener archivos PDF o Markdown con contenido.")
                st.stop() # Detiene la ejecución de Streamlit
            chunks = TextSplitter().split_documents(all_documents)
            faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH)
            print("Índice FAISS creado y guardado.")
    else:
        print(f"Índice FAISS no encontrado en {FAISS_INDEX_PATH}. Procediendo a crearlo.")
        all_documents = []
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    all_documents.extend(DocumentLoader().load_pdf(file_path))
                elif file.endswith(".md"):
                    all_documents.extend(DocumentLoader().load_markdown(file_path))
        if not all_documents:
            st.error(f"No se encontraron documentos válidos en {DOCS_PATH}. Asegúrate de tener archivos PDF o Markdown con contenido.")
            st.stop() # Detiene la ejecución de Streamlit
        chunks = TextSplitter().split_documents(all_documents)
        faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH)
        print("Índice FAISS creado y guardado.")


    retriever = CustomRetriever(faiss_path=FAISS_INDEX_PATH, embeddings=embeddings, k=K_VALUE_DEMO)
    langchain_retriever = retriever.get_langchain_retriever()

    llm_model = LLMModel(model_name=LLM_MODEL_NAME, model_path=LLM_MODEL_LOCAL_PATH, device=LLM_DEVICE)
    llm = llm_model.get_llm()

    prompt_manager = PromptTemplateManager()
    qa_prompt = prompt_manager.get_qa_prompt()

    rag_chain = RAGChain(retriever=langchain_retriever, llm=llm, prompt=qa_prompt)
    print("Sistema RAG completamente cargado para Streamlit.")
    return rag_chain

# --- Interfaz de Usuario de Streamlit ---
st.title("🤖 Demostración de RAG con GPT-NeoX")
st.write("Pregunta sobre tus documentos técnicos y el sistema de Recuperación Aumentada por Generación (RAG) te responderá.")

# Cargar el sistema RAG una vez al inicio
try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error(f"Error crítico al iniciar el sistema RAG: {e}. Por favor, revisa tus configuraciones y documentos.")
    st.stop()

# Campo de entrada para la pregunta del usuario
user_question = st.text_input("Ingresa tu pregunta aquí:", key="user_question_input")

if user_question:
    with st.spinner("Buscando y generando respuesta..."):
        try:
            response = rag_chain.invoke(user_question)
            st.subheader("Respuesta Generada:")
            st.info(response)
        except Exception as e:
            st.error(f"Hubo un error al procesar tu pregunta: {e}")

st.markdown("---")
st.markdown("Este sistema RAG utiliza:")
st.markdown(f"- **LLM:** {LLM_MODEL_NAME} (ejecutándose localmente)")
st.markdown(f"- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`")
st.markdown(f"- **Vector Store:** FAISS (índice local)")
st.markdown(f"- **Contexto de búsqueda (k):** {K_VALUE_DEMO}")