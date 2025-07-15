import pytest
from src.data_ingestion.text_splitter import TextSplitter
from langchain.schema import Document

def test_text_splitter_initialization():
    """
    Verifica que el TextSplitter se inicializa correctamente con los parámetros dados.
    """
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    assert splitter.splitter.chunk_size == 500
    assert splitter.splitter.chunk_overlap == 100

def test_split_documents_basic(text_splitter):
    """
    Asegura que la división de texto produce el número esperado de chunks y
    que el tamaño y solapamiento son correctos para un texto simple.
    """
    long_text = "Esto es una frase corta. " * 20 # 20 * 25 chars = 500 chars
    documents = [Document(page_content=long_text, metadata={"source": "test"})]
    
    # Configurar el splitter para que cada chunk sea pequeño
    text_splitter.splitter.chunk_size = 50
    text_splitter.splitter.chunk_overlap = 10

    chunks = text_splitter.split_documents(documents)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 10 # Debería haber muchos chunks de 50 chars
    assert all(isinstance(chunk, Document) for chunk in chunks)
    
    # Verificar que el tamaño de los chunks es aproximadamente el esperado
    # RecursiveCharacterTextSplitter intenta mantener chunks lógicos, por lo que no será exacto.
    # Pero el contenido de los chunks debería venir del texto original.
    for chunk in chunks:
        assert len(chunk.page_content) <= text_splitter.splitter.chunk_size + text_splitter.splitter.chunk_overlap # Puede exceder por el overlap
        assert "Esto es una frase corta." in chunk.page_content # Verificar que el contenido es del original

def test_split_documents_short_text(text_splitter):
    """
    Prueba con un texto muy corto, que debería producir un solo chunk.
    """
    short_text = "Esto es un texto muy corto."
    documents = [Document(page_content=short_text)]
    
    # Asegurarse de que el chunk_size sea mayor que el texto
    text_splitter.splitter.chunk_size = 1000
    text_splitter.splitter.chunk_overlap = 0

    chunks = text_splitter.split_documents(documents)
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text

def test_split_documents_empty_list(text_splitter):
    """
    Verifica que el splitter maneja una lista de documentos vacía.
    """
    chunks = text_splitter.split_documents([])
    assert isinstance(chunks, list)
    assert len(chunks) == 0

def test_split_documents_empty_document(text_splitter):
    """
    Verifica que el splitter maneja un documento con contenido vacío.
    Debería producir un chunk vacío.
    """
    documents = [Document(page_content="")]
    chunks = text_splitter.split_documents(documents)
    assert len(chunks) == 1
    assert chunks[0].page_content == ""

def test_split_documents_metadata_preservation(text_splitter):
    """
    Asegura que los metadatos se preservan en los chunks divididos.
    """
    text_with_meta = "Primer párrafo de un documento con metadatos. Segundo párrafo."
    documents = [Document(page_content=text_with_meta, metadata={"source": "test_file.txt", "page": 1})]
    
    text_splitter.splitter.chunk_size = 30
    text_splitter.splitter.chunk_overlap = 0

    chunks = text_splitter.split_documents(documents)
    
    assert len(chunks) > 1 # Debería dividirse
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test_file.txt"
        assert "page" in chunk.metadata
        assert chunk.metadata["page"] == 1