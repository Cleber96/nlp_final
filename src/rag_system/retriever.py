import logging
from typing import Optional, Dict, Any

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.vector_store.faiss_store import FAISSStore # Importar FAISSStore
from src.rag_system.config import config # Importar la configuración

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CustomRetriever:
    """
    Clase para gestionar la recuperación de documentos utilizando un VectorStore FAISS.
    Carga un índice FAISS existente y lo expone como un LangChain Retriever.
    """
    def __init__(self, faiss_path: str = config.FAISS_INDEX_DIR, 
                 embeddings: Optional[HuggingFaceEmbeddings] = None, 
                 k: int = config.RETRIEVER_K):
        """
        Inicializa el CustomRetriever.

        Args:
            faiss_path (str): La ruta donde se encuentra el índice FAISS guardado.
            embeddings (Optional[HuggingFaceEmbeddings]): La instancia del modelo de embeddings.
                                                            Si no se proporciona, se intentará inicializar uno.
            k (int): El número de documentos más relevantes a recuperar por defecto.
        """
        self.faiss_path = faiss_path
        self.k = k
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
        self._load_vectorstore()
        logger.info(f"CustomRetriever inicializado con k={self.k} y ruta FAISS: {self.faiss_path}")

    def _load_vectorstore(self):
        """
        Carga el VectorStore FAISS desde la ruta especificada.
        Si no se proporciona una instancia de embeddings, intentará crear una.
        """
        if self.embeddings is None:
            logger.warning("No se proporcionó una instancia de embeddings. Intentando inicializar HuggingFaceEmbeddings.")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL_NAME,
                    model_kwargs=config.EMBEDDING_MODEL_KWARTS,
                    encode_kwargs=config.EMBEDDING_ENCODE_KWARTS
                )
                logger.info(f"Embeddings inicializados automáticamente para el retriever: {config.EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Error al inicializar embeddings automáticamente: {e}")
                raise ValueError("No se pudo inicializar el modelo de embeddings necesario para cargar FAISS.")

        try:
            # Reutilizamos la lógica de carga de FAISSStore
            faiss_handler = FAISSStore(embeddings=self.embeddings)
            faiss_handler.load_local(self.faiss_path)
            self.vectorstore = faiss_handler.vectorstore
            logger.info("VectorStore FAISS cargado exitosamente para el retriever.")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error al cargar el VectorStore FAISS: {e}")
            self.vectorstore = None
            raise ValueError(f"No se pudo cargar el índice FAISS desde {self.faiss_path}. "
                             "Asegúrate de que el índice haya sido creado previamente.")

    def get_langchain_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> VectorStoreRetriever:
        """
        Retorna un objeto LangChain VectorStoreRetriever configurado.

        Args:
            search_kwargs (Optional[Dict[str, Any]]): Argumentos para la búsqueda del retriever.
                                                      Si no se proporciona, usa el 'k' predeterminado.
                                                      Ej: {"k": 5}

        Returns:
            VectorStoreRetriever: Una instancia de LangChain VectorStoreRetriever.

        Raises:
            ValueError: Si el VectorStore no ha sido cargado/inicializado.
        """
        if self.vectorstore is None:
            logger.error("El VectorStore no está cargado. No se puede obtener el retriever.")
            raise ValueError("VectorStore not loaded. Call _load_vectorstore first or ensure index exists.")
        
        if search_kwargs is None:
            search_kwargs = {"k": self.k}
        
        logger.info(f"Retornando LangChain Retriever con search_kwargs: {search_kwargs}")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

if __name__ == "__main__":
    print("--- Probando CustomRetriever ---")
    
    # Para probar este módulo, necesitas un índice FAISS existente y un modelo de embeddings.
    # Asumimos que ya has ejecutado src/vector_store/faiss_store.py o src/main.py para crear el índice.
    
    # 1. Inicializar Embeddings (debe ser la misma instancia o configuración que se usó para crear el índice)
    try:
        from src.vector_store.embeddings_generator import EmbeddingsGenerator
        embed_gen = EmbeddingsGenerator(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs=config.EMBEDDING_MODEL_KWARTS,
            encode_kwargs=config.EMBEDDING_ENCODE_KWARTS
        )
        embeddings_instance = embed_gen.get_embeddings_model()
    except Exception as e:
        print(f"Error al inicializar EmbeddingsGenerator: {e}")
        print("Asegúrate de que el modelo de embeddings esté disponible.")
        exit()

    # 2. Intentar inicializar CustomRetriever
    try:
        # Asegúrate de que config.FAISS_INDEX_DIR apunte a un índice válido
        print(f"Intentando cargar retriever desde: {config.FAISS_INDEX_DIR}")
        custom_retriever = CustomRetriever(
            faiss_path=config.FAISS_INDEX_DIR,
            embeddings=embeddings_instance,
            k=3 # Prueba con un k diferente
        )
        langchain_retriever = custom_retriever.get_langchain_retriever()
        print(f"Retriever de LangChain obtenido: {type(langchain_retriever)}")

        # 3. Realizar una búsqueda de prueba con el retriever
        test_query = "¿Qué es LangChain?"
        print(f"\nRealizando búsqueda de prueba para: '{test_query}'")
        retrieved_docs = langchain_retriever.invoke(test_query)

        if retrieved_docs:
            print(f"Documentos recuperados ({len(retrieved_docs)}):")
            for i, doc in enumerate(retrieved_docs):
                print(f"--- Doc {i+1} (Fuente: {doc.metadata.get('source', 'N/A')}) ---")
                print(doc.page_content[:150] + "...") # Mostrar solo una parte
        else:
            print("No se recuperaron documentos. Asegúrate de que el índice FAISS exista y contenga datos relevantes.")

    except ValueError as e:
        print(f"\nError al inicializar o usar CustomRetriever: {e}")
        print("Asegúrate de haber creado el índice FAISS ejecutando src/main.py (o src/vector_store/faiss_store.py) primero.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado durante la prueba del CustomRetriever: {e}")