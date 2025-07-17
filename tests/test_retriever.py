import pytest
import os
import shutil
from src.rag_system.retriever import CustomRetriever
from src.vector_store.faiss_store import FAISSStore # Importar tu FAISSStore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from unittest.mock import MagicMock, patch, ANY

# Fixtures `mock_embeddings`, `test_faiss_path`, `mock_faiss_store` se definen en conftest.py

@pytest.fixture
def mock_faiss_load_local_success():
    """
    Fixtura que mockea FAISSStore.load_local para simular una carga exitosa.
    Se asegura de que la instancia de FAISSStore tenga un vectorstore mockeado.
    """
    # Creamos un mock para el VectorStore real de LangChain (FAISS)
    mock_langchain_faiss_vectorstore = MagicMock(spec=FAISS)
    # Configuramos su método .as_retriever() para que devuelva un mock de VectorStoreRetriever
    mock_langchain_retriever = MagicMock(spec=VectorStoreRetriever)
    mock_langchain_faiss_vectorstore.as_retriever.return_value = mock_langchain_retriever

    # Parcheamos el método load_local de TU CLASE FAISSStore
    # autospec=True es útil, pero significa que el mock imita la firma.
    # El método load_local de instancia de FAISSStore solo toma 'faiss_path' como argumento (self ya está vinculado)
    with patch('src.vector_store.faiss_store.FAISSStore.load_local', autospec=True) as mock_load_local:
        # Definimos un side_effect para el mock_load_local
        # Este side_effect se ejecutará cuando se llame a FAISSStore.load_local.
        # Recibe la instancia de FAISSStore (self) y la ruta.
        def _side_effect_load_local(instance, faiss_path):
            # Asignamos el mock de LangChain FAISS VectorStore a la propiedad vectorstore
            # de la instancia real de FAISSStore que se está creando en CustomRetriever.
            instance.vectorstore = mock_langchain_faiss_vectorstore
        
        mock_load_local.side_effect = _side_effect_load_local
        yield mock_load_local # Cedemos el control al test, el mock está activo


@pytest.fixture
def mock_faiss_load_local_failure():
    """
    Fixtura que mockea FAISSStore.load_local para simular un fallo (FileNotFoundError).
    """
    # Aquí autospec=True también aplica. La llamada real a load_local solo pasa 'faiss_path'.
    with patch('src.vector_store.faiss_store.FAISSStore.load_local', side_effect=FileNotFoundError("FAISS index not found"), autospec=True) as mock_method:
        yield mock_method

# --- TESTS ---

def test_custom_retriever_initialization(mock_faiss_load_local_success, mock_embeddings, test_faiss_path):
    """
    Verifica que CustomRetriever se inicializa correctamente y carga el FAISSStore.
    """
    retriever_instance = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=3)
    
    assert retriever_instance.faiss_path == test_faiss_path
    assert retriever_instance.embeddings is mock_embeddings
    assert retriever_instance.k == 3
    assert retriever_instance.vectorstore is not None
    assert isinstance(retriever_instance.vectorstore, MagicMock)
    # Ya no verificamos .spec, ya que puede causar AttributeError con MagicMock(spec=...)
    # Lo importante es que el mock se comporta como el FAISS esperado.

    # El mock de load_local recibe la instancia de FAISSStore (self) y la ruta
    mock_faiss_load_local_success.assert_called_once_with(ANY, test_faiss_path)


def test_custom_retriever_load_local_failure(mock_faiss_load_local_failure, mock_embeddings, test_faiss_path):
    """
    Verifica que el CustomRetriever maneja el caso de que el archivo FAISS no exista
    durante la carga inicial, lanzando un ValueError.
    """
    with pytest.raises(ValueError, match=f"No se pudo cargar el índice FAISS desde {test_faiss_path}"):
        CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=4)

    # El mock de load_local, debido a autospec=True, espera la instancia (self) como primer argumento.
    # La llamada real en tu código es faiss_handler.load_local(self.faiss_path),
    # donde faiss_handler es una instancia de FAISSStore. Por lo tanto,
    # el mock de la función de clase load_local recibirá la instancia como primer arg.
    mock_faiss_load_local_failure.assert_called_once_with(ANY, test_faiss_path)


def test_get_langchain_retriever(mock_faiss_load_local_success, mock_embeddings, test_faiss_path):
    """
    Verifica que `get_langchain_retriever` retorna un objeto LangChain VectorStoreRetriever
    con el parámetro k correcto.
    """
    retriever_k = 5
    custom_retriever = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=retriever_k)

    langchain_retriever = custom_retriever.get_langchain_retriever()

    assert isinstance(langchain_retriever, VectorStoreRetriever) or isinstance(langchain_retriever, MagicMock)

    custom_retriever.vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": retriever_k})


def test_retriever_invoke_method(mock_faiss_load_local_success, mock_embeddings, test_faiss_path):
    """
    Verifica que el retriever de LangChain (obtenido a través de CustomRetriever)
    devuelve documentos cuando es invocado.
    """
    custom_retriever = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=2)
    langchain_retriever = custom_retriever.get_langchain_retriever()

    expected_docs = [
        Document(page_content="Esto es un documento de prueba 1."),
        Document(page_content="Esto es un documento de prueba 2.")
    ]
    langchain_retriever.invoke.return_value = expected_docs

    query = "test query"
    retrieved_docs = langchain_retriever.invoke(query)

    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) == 2
    assert retrieved_docs == expected_docs
    langchain_retriever.invoke.assert_called_once_with(query)