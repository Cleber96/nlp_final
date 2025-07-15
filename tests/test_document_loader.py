# tests/test_document_loader.py
import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
# Asegúrate de que las importaciones de loaders sean correctas
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

# Asegúrate de importar tu clase DocumentLoader correctamente
from src.data_ingestion.document_loader import DocumentLoader

# --- Fixtures ---

@pytest.fixture
def document_loader():
    """Retorna una instancia de DocumentLoader."""
    return DocumentLoader()

@pytest.fixture
def test_docs_dir(tmp_path):
    """Crea un directorio temporal para documentos de prueba y lo limpia después."""
    docs_path = tmp_path / "test_docs"
    docs_path.mkdir()

    # Crea un archivo markdown real
    md_content = """# Título del Documento

Este es un **documento** de prueba en Markdown.

Contiene varias líneas de texto para ser dividido."""
    with open(docs_path / "test_document.md", "w") as f:
        f.write(md_content)

    yield str(docs_path)

    # La limpieza la maneja tmp_path automáticamente

@pytest.fixture
def dummy_pdf_path(test_docs_dir):
    """
    Crea un archivo dummy.pdf que no es un PDF válido,
    pero su existencia es necesaria para os.path.exists.
    La lógica de carga de PDF se mockeará en el test.
    """
    pdf_path = os.path.join(test_docs_dir, "dummy.pdf")
    with open(pdf_path, "w") as f:
        f.write("Esto no es un PDF real, es un placeholder.") # Contenido irrelevante, solo para que exista
    return pdf_path


# --- Tests ---

def test_load_pdf_success(document_loader, dummy_pdf_path):
    """
    Verifica que `load_pdf` carga un PDF y retorna una lista de Documentos.
    Mockeamos PyPDFLoader para simular una carga exitosa de un PDF válido.
    """
    assert os.path.exists(dummy_pdf_path) 
    
    # Importante: el patch debe apuntar a la ruta donde PyPDFLoader es importado dentro de document_loader.py
    with patch('src.data_ingestion.document_loader.PyPDFLoader') as MockPyPDFLoader:
        mock_loader_instance = MockPyPDFLoader.return_value
        # Aseguramos que el mock devuelva una lista de documentos
        mock_loader_instance.load.return_value = [Document(page_content="Contenido de prueba de PDF.")]
        
        documents = document_loader.load_pdf(dummy_pdf_path)
        
        assert isinstance(documents, list)
        assert len(documents) > 0 
        assert isinstance(documents[0], Document)
        assert "Contenido de prueba de PDF." in documents[0].page_content
        
        # Verificar que PyPDFLoader fue instanciado con la ruta correcta
        MockPyPDFLoader.assert_called_once_with(dummy_pdf_path)
        # Verificar que el método load() fue llamado en la instancia del loader
        mock_loader_instance.load.assert_called_once()

def test_load_pdf_non_existent(document_loader, test_docs_dir):
    """
    Verifica que `load_pdf` maneja archivos PDF inexistentes
    retornando una lista vacía, según la implementación actual.
    """
    non_existent_pdf_path = os.path.join(test_docs_dir, "non_existent.pdf")
    documents = document_loader.load_pdf(non_existent_pdf_path)
    assert isinstance(documents, list)
    assert len(documents) == 0 # <--- CAMBIO AQUÍ: Esperar lista vacía
    # Opcional: Puedes verificar que el logger.error fue llamado si quieres testear el logging
    # with patch('src.data_ingestion.document_loader.logger.error') as mock_logger_error:
    #     documents = document_loader.load_pdf(non_existent_pdf_path)
    #     mock_logger_error.assert_called_once()


def test_load_markdown_success(document_loader, test_docs_dir):
    """
    Verifica que `load_markdown` carga un archivo Markdown y retorna una lista de Documentos.
    """
    md_path = os.path.join(test_docs_dir, "test_document.md")
    assert os.path.exists(md_path)
    
    documents = document_loader.load_markdown(md_path)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    assert "Título del Documento" in documents[0].page_content
    # CORRECCIÓN: el cargador de Markdown eliminará el formato "**"
    assert "Este es un documento de prueba en Markdown." in documents[0].page_content

def test_load_markdown_non_existent(document_loader, test_docs_dir):
    """
    Verifica que `load_markdown` maneja archivos Markdown inexistentes
    retornando una lista vacía, según la implementación actual.
    """
    non_existent_md_path = os.path.join(test_docs_dir, "non_existent.md")
    documents = document_loader.load_markdown(non_existent_md_path)
    assert isinstance(documents, list)
    assert len(documents) == 0 # <--- CAMBIO AQUÍ: Esperar lista vacía
    # Opcional: Puedes verificar que el logger.error fue llamado si quieres testear el logging
    # with patch('src.data_ingestion.document_loader.logger.error') as mock_logger_error:
    #     documents = document_loader.load_markdown(non_existent_md_path)
    #     mock_logger_error.assert_called_once()


# Renombrado y ajustado para load_documents_from_paths
def test_load_documents_from_paths_success(document_loader, test_docs_dir, dummy_pdf_path):
    """
    Verifica que `load_documents_from_paths` carga todos los tipos de documentos soportados.
    """
    # Mockear tanto PyPDFLoader como UnstructuredMarkdownLoader
    with patch('src.data_ingestion.document_loader.PyPDFLoader') as MockPyPDFLoader, \
         patch('src.data_ingestion.document_loader.UnstructuredMarkdownLoader') as MockUnstructuredMarkdownLoader:

        # Configurar mock para PDF
        mock_pdf_loader_instance = MockPyPDFLoader.return_value
        mock_pdf_loader_instance.load.return_value = [Document(page_content="Contenido PDF mockeado.")]

        # Configurar mock para Markdown
        mock_md_loader_instance = MockUnstructuredMarkdownLoader.return_value
        mock_md_loader_instance.load.return_value = [Document(page_content="Contenido Markdown mockeado.")]

        # Lista de rutas a pasar al método, incluyendo uno no existente para probar el manejo
        file_paths_to_load = [
            dummy_pdf_path, # Path al PDF dummy existente
            os.path.join(test_docs_dir, "test_document.md"), # Path al MD existente
            os.path.join(test_docs_dir, "non_existent.pdf"), # Path a un PDF no existente
            os.path.join(test_docs_dir, "non_existent.md"), # Path a un MD no existente
            os.path.join(test_docs_dir, "unsupported.txt") # Path a un tipo de archivo no soportado
        ]

        documents = document_loader.load_documents_from_paths(file_paths_to_load) # <--- CAMBIO AQUÍ: Llamar al método correcto

        assert isinstance(documents, list)
        # Esperamos 2 documentos (1 del PDF mockeado, 1 del MD mockeado).
        # Los no existentes y no soportados resultarán en listas vacías o advertencias.
        assert len(documents) == 2 

        # Verificar que los loaders fueron llamados con las rutas correctas
        MockPyPDFLoader.assert_called_once_with(dummy_pdf_path)
        mock_pdf_loader_instance.load.assert_called_once()
        
        MockUnstructuredMarkdownLoader.assert_called_once_with(os.path.join(test_docs_dir, "test_document.md"))
        mock_md_loader_instance.load.assert_called_once()
        
        # Opcional: Verificar que el logger.error/warning fue llamado para los archivos no existentes/no soportados
        # Esto requeriría mockear el logger en este test también.

        # Verificar que el contenido esperado está presente
        pdf_found = any("Contenido PDF mockeado." in doc.page_content for doc in documents)
        md_found = any("Contenido Markdown mockeado." in doc.page_content for doc in documents)
        assert pdf_found
        assert md_found