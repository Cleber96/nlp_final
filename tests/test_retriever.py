import pytest
import os
from src.rag_system.retriever import CustomRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from unittest.mock import MagicMock, patch

# Fixtures `mock_embeddings`, `test_faiss_path`, `mock_faiss_store` se definen en conftest.py

@pytest.fixture
def mock_faiss_load_local_success(mock_faiss_store):
    """
    Fixture que mockea FAISS.load_local para siempre retornar una instancia de FAISS mockeada
    que ya tiene el comportamiento de .as_retriever() definido.
    """
    with patch('langchain_community.vectorstores.FAISS.load_local', return_value=mock_faiss_store.vectorstore) as mock_method:
        yield mock_method

def test_custom_retriever_initialization(mock_faiss_load_local_success, mock_embeddings, test_faiss_path):
    """
    Verifica que CustomRetriever se inicializa correctamente y carga el FAISSStore.
    """
    retriever_instance = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=3)
    
    assert retriever_instance.faiss_path == test_faiss_path
    assert retriever_instance.embeddings is mock_embeddings
    assert retriever_instance.k == 3
    assert retriever_instance.vectorstore is not None
    # Verificar que FAISS.load_local fue llamado durante la inicialización
    mock_faiss_load_local_success.assert_called_once_with(
        test_faiss_path, mock_embeddings, allow_dangerous_deserialization=True
    )

def test_custom_retriever_load_local_failure(mock_embeddings, test_faiss_path):
    """
    Verifica que el CustomRetriever maneja el caso de que el archivo FAISS no exista
    durante la carga inicial.
    """
    # Asegurarse de que el directorio FAISS no exista
    if os.path.exists(test_faiss_path):
        shutil.rmtree(test_faiss_path)

    # Mockear FAISS.load_local para que lance FileNotFoundError
    with patch('langchain_community.vectorstores.FAISS.load_local', side_effect=FileNotFoundError("FAISS index not found")) as mock_load:
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=4)
        mock_load.assert_called_once()

def test_get_langchain_retriever(mock_faiss_load_local_success, mock_embeddings, test_faiss_path, mock_faiss_store):
    """
    Verifica que `get_langchain_retriever` retorna un objeto LangChain VectorStoreRetriever
    con el parámetro k correcto.
    """
    retriever_k = 5
    custom_retriever = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=retriever_k)
    
    langchain_retriever = custom_retriever.get_langchain_retriever()
    
    assert isinstance(langchain_retriever, MagicMock) # Debe ser el mock de VectorStoreRetriever
    # Asegurarse de que as_retriever del mock de FAISS fue llamado con el search_kwargs correcto
    mock_faiss_store.vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": retriever_k})

def test_retriever_invoke_method(mock_faiss_load_local_success, mock_embeddings, test_faiss_path, mock_faiss_store):
    """
    Aunque CustomRetriever no tiene un método 'invoke' directamente,
    su `get_langchain_retriever` sí lo tiene. Aquí probamos que el retriever de LangChain
    devuelve documentos.
    """
    custom_retriever = CustomRetriever(faiss_path=test_faiss_path, embeddings=mock_embeddings, k=2)
    langchain_retriever = custom_retriever.get_langchain_retriever()

    query = "technical documentation"
    retrieved_docs = langchain_retriever.invoke(query) # o .get_relevant_documents()

    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) == 2 # Definido en mock_vectorstore_retriever
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    assert "Esto es un documento de prueba relevante A" in retrieved_docs[0].page_content
    # Verificar que el método subyacente del mock FAISS fue llamado
    mock_faiss_store.vectorstore.as_retriever.return_value.invoke.assert_called_once_with(query)