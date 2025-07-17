import pytest
from src.data_ingestion.text_splitter import TextSplitter
from langchain.schema import Document

def test_text_splitter_initialization():
    """
    Verifica que el TextSplitter se inicializa correctamente con los parámetros dados.
    """
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    # ACCIÓN: Cambiar a _chunk_size y _chunk_overlap
    assert splitter.splitter._chunk_size == 500
    assert splitter.splitter._chunk_overlap == 100

def test_split_documents_basic():
    """
    Asegura que la división de texto produce el número esperado de chunks y
    que el tamaño y solapamiento son correctos para un texto simple.
    """
    long_text = "Esto es una frase corta. " * 20 # 20 * 25 chars = 500 chars

    documents = [Document(page_content=long_text, metadata={"source": "test"})]
    
    splitter_for_test = TextSplitter(chunk_size=50, chunk_overlap=10)

    chunks = splitter_for_test.split_documents(documents)
    
    assert isinstance(chunks, list)
    assert 8 <= len(chunks) <= 12 # Rango razonable de chunks
    assert all(isinstance(chunk, Document) for chunk in chunks)
    
    for chunk in chunks:
        # ACCIÓN: Cambiar a _chunk_size (o usar el valor fijo del test si se prefiere)
        assert len(chunk.page_content) <= splitter_for_test.splitter._chunk_size * 1.5 
        assert "Esto es una frase corta." in chunk.page_content

def test_split_documents_short_text():
    """
    Prueba con un texto muy corto, que debería producir un solo chunk.
    """
    short_text = "Esto es un texto muy corto."
    documents = [Document(page_content=short_text)]
    
    splitter_for_test = TextSplitter(chunk_size=1000, chunk_overlap=0)

    chunks = splitter_for_test.split_documents(documents)
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text

def test_split_documents_empty_list(text_splitter):
    """
    Verifica que el splitter maneja una lista de documentos vacía.
    """
    chunks = text_splitter.split_documents([])
    assert isinstance(chunks, list)
    assert len(chunks) == 0

def test_split_documents_empty_document():
    """
    Verifica que el splitter maneja un documento con contenido vacío.
    Debería producir 0 chunks ya que RecursiveCharacterTextSplitter no crea chunks de texto vacío.
    """
    documents = [Document(page_content="")]
    splitter_for_test = TextSplitter(chunk_size=100, chunk_overlap=0)

    chunks = splitter_for_test.split_documents(documents)
    # ACCIÓN: Esperar 0 chunks para un documento vacío.
    assert len(chunks) == 0

def test_split_documents_metadata_preservation():
    """
    Asegura que los metadatos se preservan en los chunks divididos.
    También se asegura de que la división realmente ocurra.
    """
    long_text_with_meta = "Este es el primer fragmento del documento. " \
                          "Es lo suficientemente largo como para ser dividido. " \
                          "Aquí viene el segundo fragmento de texto. " \
                          "Y este es el tercer fragmento, para asegurar la división."
    
    documents = [Document(page_content=long_text_with_meta, metadata={"source": "test_file.txt", "page": 1})]
    
    splitter_for_test = TextSplitter(chunk_size=30, chunk_overlap=0)

    chunks = splitter_for_test.split_documents(documents)
    
    assert len(chunks) > 1 
    assert all(isinstance(chunk, Document) for chunk in chunks)
    
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test_file.txt"
        assert "page" in chunk.metadata
        assert chunk.metadata["page"] == 1