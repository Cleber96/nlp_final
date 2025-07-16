# tests/test_embeddings_generator.py
import pytest
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from langchain.schema import Document
from unittest.mock import MagicMock, patch

# La fixture 'embeddings_generator' se definirá en conftest.py
# y proporcionará una instancia de EmbeddingsGenerator con un mock de HuggingFaceEmbeddings.

def test_embeddings_generator_initialization(embeddings_generator):
    """
    Verifica que el EmbeddingsGenerator se inicializa correctamente
    y que tiene una instancia de HuggingFaceEmbeddings (mockeada).
    """
    assert isinstance(embeddings_generator, EmbeddingsGenerator)
    # Asegúrate de que embeddings_generator.embeddings es el mock inyectado.
    assert isinstance(embeddings_generator.embeddings, MagicMock)
    assert embeddings_generator.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

def test_embed_texts_returns_list_of_lists(embeddings_generator):
    """
    Verifica que `embed_texts` retorna una lista de listas de floats
    (la representación de embeddings) y llama al método correcto del modelo (embed_documents).
    """
    texts = ["Texto de ejemplo 1", "Texto de ejemplo 2"]
    embeddings = embeddings_generator.embed_texts(texts) # <-- ¡Método correcto!
    
    assert isinstance(embeddings, list)
    assert all(isinstance(e, list) for e in embeddings)
    assert all(isinstance(val, float) for e in embeddings for val in e)
    assert len(embeddings) == len(texts)
    
    # Verificar que el mock interno de HuggingFaceEmbeddings fue llamado con los textos
    embeddings_generator.embeddings.embed_documents.assert_called_once_with(texts)

def test_embed_texts_dimension(embeddings_generator):
    """
    Verifica que los embeddings generados por `embed_texts` tienen la dimensión esperada.
    La dimensión esperada (768) se define en la fixture `mock_embeddings_model` en conftest.py.
    """
    texts = ["Un solo texto"]
    embeddings = embeddings_generator.embed_texts(texts) # <-- ¡Método correcto!
    
    expected_dim = 768 # Definido en la fixture mock_embeddings_model en conftest.py
    assert len(embeddings[0]) == expected_dim

def test_embed_texts_empty_list(embeddings_generator):
    """
    Verifica que `embed_texts` maneja correctamente una lista de textos vacía.
    """
    embeddings = embeddings_generator.embed_texts([]) # <-- ¡Método correcto!
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0
    # Asegurarse de que el método mockeado fue llamado, incluso con una lista vacía
    embeddings_generator.embeddings.embed_documents.assert_not_called()

def test_embed_documents_placeholder(embeddings_generator):
    """
    Verifica que `embed_documents` de EmbeddingsGenerator retorna los documentos sin cambios.
    Este método es un placeholder en tu diseño actual, ya que FAISS maneja la creación de embeddings
    directamente.
    """
    documents = [
        Document(page_content="Contenido del documento 1."),
        Document(page_content="Contenido del documento 2.")
    ]
    
    result_documents = embeddings_generator.embed_documents(documents)
    
    assert result_documents is documents # Debería retornar la misma lista de objetos por referencia
    assert len(result_documents) == 2
    # Asegurarse de que el método `embed_documents` del modelo de embeddings NO fue llamado,
    # ya que tu `EmbeddingsGenerator.embed_documents` es un placeholder que no genera embeddings directamente.
    embeddings_generator.embeddings.embed_documents.assert_not_called()