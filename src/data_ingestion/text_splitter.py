import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TextSplitter:
    """
    Clase para dividir documentos de LangChain en fragmentos (chunks) más pequeños.
    Utiliza RecursiveCharacterTextSplitter para una división inteligente.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el divisor de texto.

        Args:
            chunk_size (int): El tamaño máximo de cada fragmento de texto.
            chunk_overlap (int): El número de caracteres de solapamiento entre fragmentos consecutivos.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser un entero positivo.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap no puede ser negativo.")
        if chunk_overlap >= chunk_size:
            logger.warning("chunk_overlap es igual o mayor que chunk_size, esto puede llevar a chunks redundantes.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Usa len para contar caracteres
            is_separator_regex=False, # No usar regex para separadores por defecto
            separators=["\n\n", "\n", " ", ""] # Separadores comunes para texto
        )
        logger.info(f"TextSplitter inicializado con chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide una lista de objetos Document de LangChain en fragmentos más pequeños.

        Args:
            documents (List[Document]): Una lista de documentos a dividir.

        Returns:
            List[Document]: Una lista de objetos Document, cada uno representando un fragmento.
                            Retorna una lista vacía si no se proporcionan documentos.
        """
        if not documents:
            logger.warning("No se proporcionaron documentos para dividir. Retornando lista vacía.")
            return []

        logger.info(f"Iniciando división de {len(documents)} documentos...")
        all_chunks: List[Document] = []
        for i, doc in enumerate(documents):
            try:
                chunks = self.splitter.split_documents([doc])
                all_chunks.extend(chunks)
                logger.debug(f"Documento {i+1}/{len(documents)} dividido en {len(chunks)} fragmentos.")
            except Exception as e:
                logger.error(f"Error al dividir el documento {i+1}: {e}")
                continue # Continuar con el siguiente documento

        logger.info(f"División completa. Total de fragmentos generados: {len(all_chunks)}")
        return all_chunks