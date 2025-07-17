# src/data_ingestion/document_loader.py
import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.schema import Document

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DocumentLoader:
    """
    Clase para cargar documentos de diferentes formatos (PDF, Markdown)
    y convertirlos en objetos Document de LangChain.
    """
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Carga un documento PDF desde la ruta especificada y retorna una lista de Documentos.

        Args:
            file_path (str): Ruta al archivo PDF.

        Returns:
            List[Document]: Una lista de objetos Document, cada uno representando una página del PDF.
                            Retorna una lista vacía si el archivo no existe o hay un error.
        """
        if not os.path.exists(file_path):
            logger.error(f"Error: El archivo PDF no se encontró en la ruta: {file_path}")
            return []
        try:
            logger.info(f"Cargando documento PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Se cargaron {len(documents)} páginas del PDF.")
            return documents
        except Exception as e:
            logger.error(f"Error al cargar el PDF {file_path}: {e}")
            return []

    def load_markdown(self, file_path: str) -> List[Document]:
        """
        Carga un documento Markdown desde la ruta especificada y retorna una lista de Documentos.

        Args:
            file_path (str): Ruta al archivo Markdown.

        Returns:
            List[Document]: Una lista de objetos Document, donde el contenido del Markdown
                            se trata como un único documento. Retorna una lista vacía
                            si el archivo no existe o hay un error.
        """
        if not os.path.exists(file_path):
            logger.error(f"Error: El archivo Markdown no se encontró en la ruta: {file_path}")
            return []
        try:
            logger.info(f"Cargando documento Markdown: {file_path}")
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            logger.info(f"Se cargó 1 documento Markdown (posiblemente dividido internamente por UnstructuredLoader).")
            return documents
        except Exception as e:
            logger.error(f"Error al cargar el Markdown {file_path}: {e}")
            return []

    def load_documents_from_paths(self, file_paths: List[str]) -> List[Document]:
        """
        Carga una lista de documentos de diferentes tipos (PDF, Markdown)
        desde las rutas especificadas.

        Args:
            file_paths (List[str]): Una lista de rutas a los archivos.

        Returns:
            List[Document]: Una lista consolidada de objetos Document de todos los archivos cargados.
        """
        all_documents: List[Document] = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            try: # Envuelve la llamada para capturar errores y continuar
                if ext == '.pdf':
                    all_documents.extend(self.load_pdf(file_path))
                elif ext == '.md':
                    all_documents.extend(self.load_markdown(file_path))
                else:
                    logger.warning(f"Tipo de archivo no soportado o reconocido: {file_path}. Saltando.")
            except FileNotFoundError as e: # Captura si el archivo no existe
                logger.warning(f"Archivo no encontrado, se saltará: {file_path} - {e}")
            except Exception as e: # Captura otros errores durante la carga (ej. PDF corrupto)
                logger.error(f"Error al procesar {file_path}: {e}. Se saltará.")
        logger.info(f"Carga completa. Total de documentos/páginas cargados: {len(all_documents)}")
        return all_documents