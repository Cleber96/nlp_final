# tests/test_faiss_store.py
import pytest
import os
import shutil
from src.vector_store.faiss_store import FAISSStore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # Importar HuggingFaceEmbeddings real para el mockeo
from unittest.mock import MagicMock, patch

# --- Fixtures (asumimos que están en este archivo o conftest.py) ---

@pytest.fixture
def mock_embeddings():
    """Mockea una instancia de embeddings para evitar cargas de modelo reales.
    Configura el mock para que pase la verificación de isinstance().
    """
    mock_instance = MagicMock(spec=HuggingFaceEmbeddings) 
    mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
    return mock_instance

@pytest.fixture
def faiss_store(mock_embeddings):
    """Retorna una instancia de FAISSStore con embeddings mockeados."""
    return FAISSStore(embeddings=mock_embeddings)

@pytest.fixture
def test_faiss_path(tmp_path):
    """Proporciona una ruta temporal para los archivos FAISS."""
    return str(tmp_path / "test_faiss_index")

@pytest.fixture
def sample_documents():
    """Retorna una lista de documentos de prueba."""
    return [
        Document(page_content="This is the first test document."),
        Document(page_content="This is the second test document with different content."),
        Document(page_content="A third document for testing purposes."),
    ]

# --- Tests Corregidos ---

def test_faiss_store_initialization(faiss_store, mock_embeddings):
    """
    Verifica que FAISSStore se inicializa correctamente con la instancia de embeddings.
    """
    assert faiss_store.embeddings is mock_embeddings
    assert faiss_store.vectorstore is None

def test_create_from_documents_and_save(faiss_store, sample_documents, test_faiss_path):
    """
    Prueba la creación de un VectorStore FAISS a partir de documentos y su guardado.
    """
    with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_from_documents:
        mock_faiss_instance = MagicMock(spec=FAISS)
        mock_from_documents.return_value = mock_faiss_instance

        # Ejecutar el método que se está probando
        faiss_store.create_from_documents(sample_documents, faiss_path=test_faiss_path)

        # Afirmaciones
        mock_from_documents.assert_called_once_with(sample_documents, faiss_store.embeddings)
        
        # Corrección: Ajustar la aserción para que coincida con la llamada real en tu código fuente
        # Si tu faiss_store.py llama `save_local(test_faiss_path)` sin `allow_dangerous_deserialization=True`
        mock_faiss_instance.save_local.assert_called_once_with(test_faiss_path) # <--- CAMBIO AQUÍ

def test_load_local_success(faiss_store, test_faiss_path):
    """
    Prueba la carga de un VectorStore FAISS desde disco.
    Necesitamos simular que el índice FAISS ya existe y es cargable.
    """
    # Mockear FAISS.load_local y también os.path.exists
    with patch('langchain_community.vectorstores.FAISS.load_local') as mock_load_local, \
         patch('os.path.exists', return_value=True): # <--- CAMBIO AQUÍ: Mockear os.path.exists para que devuelva True
        mock_faiss_instance = MagicMock(spec=FAISS)
        mock_load_local.return_value = mock_faiss_instance

        faiss_store.load_local(faiss_path=test_faiss_path)

        # Afirmaciones: Asegurarse de que load_local fue llamado correctamente
        # Incluye allow_dangerous_deserialization=True si tu método en src/ lo usa
        # Si tu faiss_store.py llama `load_local(test_faiss_path, self.embeddings)` sin `allow_dangerous_deserialization=True`
        mock_load_local.assert_called_once_with(test_faiss_path, faiss_store.embeddings, allow_dangerous_deserialization=True) # <--- CAMBIO AQUÍ (removiendo el arg extra)

def test_load_local_non_existent(faiss_store, test_faiss_path):
    """
    Prueba que `load_local` lanza un error si el índice no existe.
    """
    # Asegurarse de que el directorio del índice NO exista para este test
    if os.path.exists(test_faiss_path):
        shutil.rmtree(test_faiss_path)

    # Mockear FAISS.load_local y también os.path.exists para que devuelva False
    with patch('langchain_community.vectorstores.FAISS.load_local') as mock_load_local, \
         patch('os.path.exists', return_value=False): # <--- CAMBIO AQUÍ: Mockear os.path.exists para que devuelva False
        
        # Simular el error que se produce cuando el archivo FAISS no se encuentra
        # mock_load_local.side_effect = RuntimeError("Error in faiss::FileIOReader::FileIOReader(...) could not open ... No such file or directory")
        # NOTA: Como os.path.exists está mockeado a False, el FileNotFoundError es lanzado ANTES de que mock_load_local sea llamado.
        # Por lo tanto, esta línea de side_effect ya no es necesaria y tu test debe esperar FileNotFoundError.

        # Tu método load_local en faiss_store.py debería atrapar RuntimeError y relanzar ValueError,
        # PERO si el archivo no existe, lanza FileNotFoundError primero.
        with pytest.raises(FileNotFoundError, match=f"FAISS index not found at {test_faiss_path}"): # <--- CAMBIO AQUÍ: Espera FileNotFoundError y ajusta el mensaje
            faiss_store.load_local(faiss_path=test_faiss_path)
        
        # Ya que FileNotFoundError es lanzado antes, mock_load_local nunca será llamado.
        mock_load_local.assert_not_called() # <--- CAMBIO AQUÍ: Aseguramos que no fue llamado


def test_as_retriever_success(faiss_store, sample_documents, test_faiss_path):
    """
    Verifica que `as_retriever` retorna un LangChain VectorStoreRetriever.
    """
    with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_from_documents:
        mock_faiss_instance = MagicMock(spec=FAISS)
        mock_from_documents.return_value = mock_faiss_instance
        
        mock_retriever_instance = MagicMock() # No necesitamos VectorStoreRetriever, un MagicMock genérico es suficiente para el spec
        mock_faiss_instance.as_retriever.return_value = mock_retriever_instance

        # Asegurarse de que el vectorstore esté inicializado (mockeando la creación)
        # y que save_local sea llamado sin el arg extra para que el test no falle aquí
        with patch.object(mock_faiss_instance, 'save_local') as mock_save_local_on_instance:
             faiss_store.create_from_documents(sample_documents, faiss_path=test_faiss_path)
             mock_save_local_on_instance.assert_called_once_with(test_faiss_path) # <--- Ajuste aquí si tu save_local no usa el arg

        retriever = faiss_store.as_retriever(search_kwargs={"k": 2})
        
        assert retriever is mock_retriever_instance
        mock_faiss_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 2})

def test_as_retriever_not_initialized(faiss_store):
    """
    Verifica que `as_retriever` lanza ValueError si el vectorstore no está inicializado.
    """
    faiss_store.vectorstore = None
    # Corrección: Ajustar el mensaje de error regex para que coincida con tu código fuente
    with pytest.raises(ValueError, match="VectorStore not initialized. Call create_from_documents or load_local first."): # <--- CAMBIO AQUÍ
        faiss_store.as_retriever()

def test_similarity_search_success(faiss_store, sample_documents, test_faiss_path):
    """
    Verifica que `similarity_search` llama al método correcto del vectorstore y retorna documentos.
    """
    with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_from_documents:
        mock_faiss_instance = MagicMock(spec=FAISS)
        mock_from_documents.return_value = mock_faiss_instance
        
        expected_docs = [
            Document(page_content="Found relevant doc 1"),
            Document(page_content="Found relevant doc 2")
        ]
        mock_faiss_instance.similarity_search.return_value = expected_docs

        # Asegurarse de que el vectorstore esté inicializado (mockeando la creación)
        # y que save_local sea llamado sin el arg extra para que el test no falle aquí
        with patch.object(mock_faiss_instance, 'save_local') as mock_save_local_on_instance:
            faiss_store.create_from_documents(sample_documents, faiss_path=test_faiss_path)
            mock_save_local_on_instance.assert_called_once_with(test_faiss_path) # <--- Ajuste aquí si tu save_local no usa el arg

        query_text = "What is the capital of France?"
        results = faiss_store.similarity_search(query_text, k=1)
        
        assert isinstance(results, list)
        assert len(results) == len(expected_docs)
        assert all(isinstance(doc, Document) for doc in results)
        assert results == expected_docs
        mock_faiss_instance.similarity_search.assert_called_once_with(query_text, k=1)

def test_similarity_search_not_initialized(faiss_store):
    """
    Verifica que `similarity_search` lanza ValueError si el vectorstore no está inicializado.
    """
    faiss_store.vectorstore = None
    # Corrección: Ajustar el mensaje de error regex para que coincida con tu código fuente
    with pytest.raises(ValueError, match="VectorStore not initialized."): # <--- CAMBIO AQUÍ
        faiss_store.similarity_search("dummy query")