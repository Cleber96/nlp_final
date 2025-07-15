import logging
import os
from typing import List

from langchain.schema import Document

from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataProcessor:
    """
    Orquestador para el proceso de ingesta de datos:
    Carga documentos de diversas fuentes y los divide en fragmentos.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el DataProcessor con un cargador de documentos y un divisor de texto.

        Args:
            chunk_size (int): Tamaño de los fragmentos de texto.
            chunk_overlap (int): Solapamiento entre fragmentos de texto.
        """
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"DataProcessor inicializado con chunk_size={chunk_size} y chunk_overlap={chunk_overlap}.")

    def ingest_data(self, source_paths: List[str]) -> List[Document]:
        """
        Procesa una lista de rutas de archivos, cargándolos y dividiéndolos en fragmentos.

        Args:
            source_paths (List[str]): Lista de rutas a los documentos (PDF, Markdown).

        Returns:
            List[Document]: Una lista de objetos Document, cada uno representando un fragmento de texto
                            listo para ser embebido y almacenado.
        """
        if not source_paths:
            logger.warning("No se proporcionaron rutas de origen para la ingesta de datos.")
            return []

        logger.info(f"Iniciando ingesta de datos para las rutas: {source_paths}")

        # 1. Cargar documentos
        loaded_documents = self.document_loader.load_documents_from_paths(source_paths)
        if not loaded_documents:
            logger.error("No se pudieron cargar documentos de las rutas proporcionadas. Finalizando ingesta.")
            return []

        # 2. Dividir documentos en fragmentos
        processed_chunks = self.text_splitter.split_documents(loaded_documents)
        if not processed_chunks:
            logger.error("No se generaron fragmentos después de la división. Finalizando ingesta.")
            return []

        logger.info(f"Proceso de ingesta completado. Total de fragmentos listos: {len(processed_chunks)}")
        return processed_chunks

# Ejemplo de uso (esto iría en un script principal como main.py o en un notebook)
if __name__ == "__main__":
    # Asegúrate de tener algunos archivos de prueba en una carpeta 'docs'
    # Por ejemplo: 'docs/example.pdf', 'docs/notes.md'
    # Crea un directorio 'docs' si no existe
    if not os.path.exists('docs'):
        os.makedirs('docs')
        print("Directorio 'docs' creado. Por favor, añade algunos archivos PDF y Markdown para probar.")
    
    # Crea un archivo PDF de prueba si no existe (usando un truco simple)
    # NOTA: PyPDFLoader necesita un PDF real. Para una prueba simple, puedes usar un PDF existente
    # o crear uno con librerías como reportlab si quieres automatizarlo.
    # Por simplicidad, asumiremos que ya existe un 'example.pdf' y 'example.md'
    
    # Simula la creación de un archivo Markdown para pruebas
    md_content = "# Título del Documento\n\nEste es un párrafo de ejemplo en un documento Markdown. " \
                 "Contiene información técnica sobre el **Proyecto 8 de RAG**.\n\n" \
                 "## Sección Importante\n\nAquí hay más detalles sobre la **ingesta de datos** y el " \
                 "proceso de división en *chunks*. Es crucial para el rendimiento."
    with open("docs/example.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Archivo 'docs/example.md' creado para pruebas.")

    # Simula la creación de un archivo PDF de prueba (requiere un PDF real para PyPDFLoader)
    # Si no tienes un PDF, esta parte fallará o no cargará documentos.
    # Puedes descargar un PDF de prueba o crear uno manualmente para el test.
    # Por ejemplo, puedes poner un PDF llamado 'example.pdf' en la carpeta 'docs'.
    pdf_path = "docs/example.pdf"
    if not os.path.exists(pdf_path):
        print(f"Advertencia: No se encontró '{pdf_path}'. Por favor, coloca un PDF para una prueba completa.")

    processor = DataProcessor(chunk_size=200, chunk_overlap=50)
    
    # Define las rutas de los documentos que quieres procesar
    document_paths = [
        "docs/example.md",
        "docs/example.pdf" # Asegúrate de que este archivo exista para una prueba completa
    ]

    processed_documents = processor.ingest_data(document_paths)

    if processed_documents:
        print(f"\nSe generaron {len(processed_documents)} fragmentos.")
        print("\nPrimer fragmento:")
        print(processed_documents[0].page_content[:500]) # Mostrar los primeros 500 caracteres
        print(f"Metadata del primer fragmento: {processed_documents[0].metadata}")
        
        if len(processed_documents) > 1:
            print("\nSegundo fragmento:")
            print(processed_documents[1].page_content[:500])
            print(f"Metadata del segundo fragmento: {processed_documents[1].metadata}")
    else:
        print("\nNo se generaron fragmentos. Revisa los logs para ver errores.")