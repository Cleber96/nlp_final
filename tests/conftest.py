# tests/conftest.py
import pytest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# Importar las clases de src para las que crearemos fixtures
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
from src.rag_system.llm_model import LLMModel
from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain
from src.evaluation.qa_metrics import QAMetrics

# Mocks de Langchain para evitar dependencias externas reales en pruebas unitarias
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings # ¡IMPORTANTE: Debe estar aquí!
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


@pytest.fixture(scope="session")
def temp_test_dir():
    """
    Crea un directorio temporal para todos los archivos de prueba
    y lo elimina después de que todas las pruebas de la sesión terminen.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def test_docs_dir(temp_test_dir):
    """
    Crea un directorio temporal para documentos de prueba específicos de cada test
    y lo llena con archivos de prueba.
    """
    docs_path = os.path.join(temp_test_dir, "test_docs")
    os.makedirs(docs_path, exist_ok=True)

    dummy_pdf_path = os.path.join(docs_path, "dummy.pdf")
    with open(dummy_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj<<>>endobj\nxref\n0 1\n0000000000 65535 f\ntrailer<<>>startxref\n0\n%%EOF")
    
    md_content = "# Título del Documento\n\nEste es un **documento** de prueba en Markdown.\n\nContiene varias líneas de texto para ser dividido."
    md_path = os.path.join(docs_path, "test_document.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    empty_md_path = os.path.join(docs_path, "empty.md")
    with open(empty_md_path, "w", encoding="utf-8") as f:
        f.write("")

    yield docs_path

    if os.path.exists(docs_path):
        shutil.rmtree(docs_path)

@pytest.fixture
def test_faiss_path(temp_test_dir):
    """
    Ruta temporal para el índice FAISS.
    """
    faiss_path = os.path.join(temp_test_dir, "test_faiss_index")
    yield faiss_path
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)

# --- Fixtures para instancias de clases de src ---

@pytest.fixture
def document_loader():
    return DocumentLoader()

@pytest.fixture
def text_splitter():
    return TextSplitter(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def mock_embeddings_model():
    """
    Mockea la instancia de HuggingFaceEmbeddings.
    Proporciona un comportamiento predecible para `embed_documents` y `embed_query`.
    """
    mock_model = MagicMock(spec=HuggingFaceEmbeddings)
    mock_model.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Simular embeddings con una dimensión de 768 (un valor común para all-MiniLM-L6-v2)
    mock_model.embed_documents.side_effect = lambda texts: [
        [float(i % 768) for i in range(768)] for _ in texts
    ]
    mock_model.embed_query.side_effect = lambda text: [float(i % 768) for i in range(768)]
    return mock_model

@pytest.fixture
def embeddings_generator(mock_embeddings_model):
    """
    Retorna una instancia de EmbeddingsGenerator, asegurando que use el mock.
    Esto se logra parcheando HuggingFaceEmbeddings en el nivel del módulo
    donde EmbeddingsGenerator lo importa.
    """
    # Patch el constructor de HuggingFaceEmbeddings para que siempre devuelva nuestro mock.
    # Esto asegura que EmbeddingsGenerator (si lo crea internamente en su __init__) use el mock.
    with patch('src.vector_store.embeddings_generator.HuggingFaceEmbeddings', return_value=mock_embeddings_model):
        yield EmbeddingsGenerator() # Crea la instancia de EmbeddingsGenerator aquí


@pytest.fixture
def mock_embeddings():
    """
    Retorna una instancia mock de HuggingFaceEmbeddings.
    Esta fixture es para ser usada por otras fixtures que necesiten un mock de embeddings simple.
    """
    mock_embed = MagicMock(spec=HuggingFaceEmbeddings)
    # Asegúrate de que embed_documents devuelva algo que parezca embeddings
    mock_embed.embed_documents.return_value = [[0.1]*384, [0.2]*384]
    mock_embed.embed_query.return_value = [0.1]*384
    return mock_embed

@pytest.fixture
def faiss_store(mock_embeddings):
    # Asumiendo que FAISSStore acepta la instancia de embeddings en su constructor
    return FAISSStore(embeddings=mock_embeddings)

@pytest.fixture
def prompt_template_manager():
    return PromptTemplateManager()

@pytest.fixture
def mock_llm():
    """
    Retorna un mock para la instancia del LLM.
    """
    mock_llm_instance = MagicMock(spec=BaseLLM)
    # Configura el método .invoke para retornar una respuesta simulada
    mock_llm_instance.invoke.return_value = "Respuesta simulada del LLM."
    return mock_llm_instance

@pytest.fixture
def llm_model(mock_llm):
    """
    Retorna una instancia de LLMModel que devuelve el mock_llm.
    """
    class MockLLMModel(LLMModel):
        def __init__(self, *args, **kwargs):
            # No llamar al init original para evitar cargar modelos reales
            self._mock_llm_instance = mock_llm
            self.model_path = kwargs.get('model_path', 'mock_path')
            self.model_name = kwargs.get('model_name', 'mock_model')

        def get_llm(self) -> BaseLLM:
            return self._mock_llm_instance
    
    return MockLLMModel(model_path="mock_path", model_name="mock_model")

@pytest.fixture
def mock_vectorstore_retriever(mock_faiss_store):
    """
    Retorna un mock para un VectorStoreRetriever.
    """
    mock_retriever = MagicMock(spec=VectorStoreRetriever)
    mock_retriever.invoke.return_value = [
        Document(page_content="Contexto relevante 1."),
        Document(page_content="Contexto relevante 2.")
    ]
    return mock_retriever


@pytest.fixture
def rag_chain(mock_vectorstore_retriever, mock_llm, prompt_template_manager):
    """
    Retorna una instancia de RAGChain con mocks para sus dependencias.
    """
    qa_prompt = prompt_template_manager.get_qa_prompt()
    return RAGChain(retriever=mock_vectorstore_retriever, llm=mock_llm, prompt=qa_prompt)

@pytest.fixture
def qa_metrics():
    return QAMetrics()

@pytest.fixture
def mock_faiss_store(mock_embeddings):
    """
    Crea un mock de FAISSStore que simula un vectorstore cargado.
    """
    mock_store = MagicMock(spec=FAISSStore)
    mock_faiss_instance = MagicMock(spec=FAISS)
    mock_faiss_instance.as_retriever.return_value = MagicMock(spec=VectorStoreRetriever)
    mock_faiss_instance.similarity_search.return_value = [
        Document(page_content="Esto es un documento de prueba relevante A"),
        Document(page_content="Este es otro documento de prueba relevante B")
    ]
    
    mock_store.vectorstore = mock_faiss_instance # Establecer el vectorstore interno
    mock_store.load_local.return_value = None # Simular carga exitosa
    mock_store.as_retriever.return_value = mock_faiss_instance.as_retriever() # Retornar el retriever del mock interno
    mock_store.similarity_search.side_effect = lambda query, k: mock_faiss_instance.similarity_search(query, k)

    return mock_store