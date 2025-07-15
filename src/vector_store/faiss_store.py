import logging
import os
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class FAISSStore:
    """
    Clase para interactuar con un VectorStore FAISS.
    Permite crear, guardar, cargar y realizar búsquedas de similitud.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        """
        Inicializa el FAISSStore con una instancia de HuggingFaceEmbeddings.

        Args:
            embeddings (HuggingFaceEmbeddings): La instancia del modelo de embeddings a usar.
        """
        if not isinstance(embeddings, HuggingFaceEmbeddings):
            raise TypeError("La instancia de 'embeddings' debe ser de tipo HuggingFaceEmbeddings.")
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
        logger.info("FAISSStore inicializado.")

    def create_from_documents(self, documents: List[Document], faiss_path: str = "faiss_index"):
        """
        Crea un VectorStore FAISS a partir de una lista de documentos y lo guarda localmente.

        Args:
            documents (List[Document]): Una lista de objetos Document de LangChain.
            faiss_path (str): La ruta donde se guardará el índice FAISS.
        """
        if not documents:
            logger.warning("No se proporcionaron documentos para crear el índice FAISS. No se creará el índice.")
            return

        logger.info(f"Creando índice FAISS a partir de {len(documents)} documentos...")
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(faiss_path, exist_ok=True) # Asegurarse de que la carpeta exista
            self.vectorstore.save_local(faiss_path)
            logger.info(f"Índice FAISS creado y guardado localmente en: {faiss_path}")
        except Exception as e:
            logger.error(f"Error al crear o guardar el índice FAISS: {e}")
            self.vectorstore = None # Asegurarse de que el vectorstore esté en None si falla

    def load_local(self, faiss_path: str = "faiss_index"):
        """
        Carga un VectorStore FAISS desde una ubicación local en disco.

        Args:
            faiss_path (str): La ruta desde donde se cargará el índice FAISS.

        Raises:
            FileNotFoundError: Si el índice FAISS no se encuentra en la ruta especificada.
            ValueError: Si hay un error al cargar el índice.
        """
        if not os.path.exists(faiss_path):
            logger.error(f"El índice FAISS no se encontró en la ruta: {faiss_path}")
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        logger.info(f"Cargando índice FAISS desde: {faiss_path}")
        try:
            # allow_dangerous_deserialization=True es necesario para cargar índices guardados
            # con versiones recientes de LangChain que usan pickle.
            self.vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Índice FAISS cargado exitosamente desde: {faiss_path}")
        except Exception as e:
            logger.error(f"Error al cargar el índice FAISS desde {faiss_path}: {e}")
            self.vectorstore = None
            raise ValueError(f"Error al cargar el índice FAISS: {e}")

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Retorna el VectorStore como un LangChain Retriever.
        Este retriever se puede usar directamente en las cadenas de LangChain.

        Args:
            search_kwargs (Optional[Dict[str, Any]]): Argumentos adicionales para la búsqueda del retriever.
                                                      Ej: {"k": 4} para recuperar los 4 documentos más relevantes.

        Returns:
            VectorStoreRetriever: Una instancia de LangChain VectorStoreRetriever.

        Raises:
            ValueError: Si el VectorStore no ha sido inicializado.
        """
        if self.vectorstore is None:
            logger.error("El VectorStore no ha sido inicializado. Llama a create_from_documents o load_local primero.")
            raise ValueError("VectorStore not initialized. Call create_from_documents or load_local first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4} # Valor por defecto si no se especifica

        logger.info(f"Retornando FAISS VectorStore como Retriever con search_kwargs: {search_kwargs}")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Realiza una búsqueda de similitud en el VectorStore para una consulta dada.

        Args:
            query (str): La cadena de consulta para buscar documentos similares.
            k (int): El número de documentos más similares a recuperar.

        Returns:
            List[Document]: Una lista de objetos Document que son los más similares a la consulta.

        Raises:
            ValueError: Si el VectorStore no ha sido inicializado.
        """
        if self.vectorstore is None:
            logger.error("El VectorStore no ha sido inicializado. Llama a create_from_documents o load_local primero.")
            raise ValueError("VectorStore not initialized.")
        
        if not query:
            logger.warning("La consulta está vacía. No se realizará la búsqueda de similitud.")
            return []

        logger.info(f"Realizando búsqueda de similitud para la consulta: '{query[:50]}...' (k={k})")
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Búsqueda de similitud completada. Se encontraron {len(results)} documentos.")
            return results
        except Exception as e:
            logger.error(f"Error durante la búsqueda de similitud: {e}")
            return []

if __name__ == "__main__":
    # Ejemplo de uso
    print("--- Probando FAISSStore ---")

    # 1. Inicializar el generador de embeddings (necesario para FAISSStore)
    try:
        embeddings_generator = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}, # o 'cuda' si tienes GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        faiss_store_instance = FAISSStore(embeddings=embeddings_generator)
    except Exception as e:
        print(f"Error al inicializar Embeddings o FAISSStore: {e}")
        exit()

    # 2. Documentos de prueba
    test_documents = [
        Document(page_content="LangChain es un framework para desarrollar aplicaciones con LLMs.", metadata={"source": "doc1"}),
        Document(page_content="FAISS es una librería de Facebook AI para búsqueda de similitud eficiente.", metadata={"source": "doc2"}),
        Document(page_content="Los embeddings son representaciones vectoriales de texto.", metadata={"source": "doc3"}),
        Document(page_content="Un sistema RAG mejora la generación de LLMs con recuperación de información.", metadata={"source": "doc4"}),
        Document(page_content="Python es un lenguaje de programación muy popular en ciencia de datos.", metadata={"source": "doc5"}),
    ]

    faiss_index_path = "temp_faiss_index"

    # 3. Crear y guardar el índice FAISS
    print("\nCreando y guardando el índice FAISS...")
    faiss_store_instance.create_from_documents(test_documents, faiss_index_path)

    # 4. Cargar el índice FAISS desde disco
    print("\nCargando el índice FAISS desde disco...")
    loaded_faiss_store_instance = FAISSStore(embeddings=embeddings_generator)
    try:
        loaded_faiss_store_instance.load_local(faiss_index_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"No se pudo cargar el índice: {e}")
        # Limpiar si se creó algo antes de fallar
        if os.path.exists(faiss_index_path):
            import shutil
            shutil.rmtree(faiss_index_path)
        exit()

    # 5. Realizar una búsqueda de similitud
    query = "¿Qué es FAISS y para qué sirve?"
    print(f"\nRealizando búsqueda de similitud para la consulta: '{query}'")
    retrieved_docs = loaded_faiss_store_instance.similarity_search(query, k=2)

    if retrieved_docs:
        print(f"\nDocumentos recuperados (k=2):")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Documento {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}) ---")
            print(doc.page_content[:200] + "...") # Mostrar solo una parte del contenido
    else:
        print("\nNo se recuperaron documentos.")

    # 6. Obtener el retriever de LangChain
    print("\nObteniendo el retriever de LangChain...")
    retriever = loaded_faiss_store_instance.as_retriever(search_kwargs={"k": 3})
    print(f"Tipo de retriever: {type(retriever)}")

    # Prueba con el retriever (simulando una llamada de LangChain)
    retrieved_by_langchain = retriever.invoke(query)
    print(f"\nDocumentos recuperados por el retriever de LangChain (k=3):")
    for i, doc in enumerate(retrieved_by_langchain):
        print(f"--- Documento {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}) ---")
        print(doc.page_content[:200] + "...")

    # 7. Limpiar el índice de prueba
    print(f"\nLimpiando el directorio del índice: {faiss_index_path}")
    import shutil
    if os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
        print("Directorio del índice FAISS temporal eliminado.")
    else:
        print("Directorio del índice FAISS temporal no encontrado para eliminar.")