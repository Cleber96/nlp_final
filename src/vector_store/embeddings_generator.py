import logging
from typing import List, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EmbeddingsGenerator:
    """
    Clase para generar embeddings a partir de textos utilizando modelos de Hugging Face.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs: Dict[str, Any] = None, encode_kwargs: Dict[str, Any] = None):
        """
        Inicializa el generador de embeddings.

        Args:
            model_name (str): El nombre del modelo de Sentence-Transformers a utilizar.
                              Por defecto: "sentence-transformers/all-MiniLM-L6-v2".
            model_kwargs (Dict[str, Any], opcional): Argumentos adicionales para pasar al constructor del modelo.
            encode_kwargs (Dict[str, Any], opcional): Argumentos adicionales para pasar al método encode (ej. 'normalize_embeddings': True).
        """
        if model_kwargs is None:
            model_kwargs = {'device': 'cpu'} # Usar CPU por defecto, cambiar a 'cuda' si hay GPU disponible
        if encode_kwargs is None:
            encode_kwargs = {'normalize_embeddings': True} # Normalizar embeddings es una buena práctica para búsqueda de similitud

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"EmbeddingsGenerator inicializado con el modelo: {model_name}")
            logger.info(f"Model kwargs: {model_kwargs}, Encode kwargs: {encode_kwargs}")
        except Exception as e:
            logger.error(f"Error al inicializar HuggingFaceEmbeddings con el modelo {model_name}: {e}")
            raise

    def get_embeddings_model(self) -> HuggingFaceEmbeddings:
        """
        Retorna la instancia del modelo de embeddings.

        Returns:
            HuggingFaceEmbeddings: La instancia del modelo de embeddings.
        """
        return self.embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts (List[str]): Una lista de cadenas de texto.

        Returns:
            List[List[float]]: Una lista de embeddings, donde cada embedding es una lista de flotantes.
                               Retorna una lista vacía si no se proporcionan textos.
        """
        if not texts:
            logger.warning("No se proporcionaron textos para generar embeddings. Retornando lista vacía.")
            return [] # <-- Esta es la línea que evita la llamada a embed_documents si la lista está vacía.
        try:
            logger.info(f"Generando embeddings para {len(texts)} textos...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embeddings generados. Dimensión de los embeddings: {len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"Error al generar embeddings para los textos: {e}")
            return []

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Genera embeddings para el contenido de una lista de objetos Document de LangChain.
        Nota: Este método es más conceptual. En la práctica, FAISS.from_documents
        se encarga de extraer el texto y generar los embeddings internamente
        usando el modelo de embeddings proporcionado. Este método podría ser útil
        si necesitaras los embeddings antes de pasarlos a FAISS o para depuración.

        Args:
            documents (List[Document]): Una lista de objetos Document.

        Returns:
            List[Document]: La misma lista de documentos, posiblemente con metadata actualizada si se añaden los embeddings.
                            (Actualmente no se modifican los documentos, solo se muestra cómo se usaría).
        """
        if not documents:
            logger.warning("No se proporcionaron documentos para generar embeddings. Retornando lista vacía.")
            return []
        
        texts = [doc.page_content for doc in documents]
        # Aquí es donde realmente se generarían los embeddings,
        # pero para la integración con FAISS.from_documents,
        # solo necesitas pasar el objeto self.embeddings.
        # Los embeddings se calcularán internamente por FAISS.
        logger.info(f"Preparando para generar embeddings para {len(documents)} documentos. "
                    "FAISS se encargará de esto directamente.")
        
        # Si quisieras almacenar los embeddings directamente en la metadata de los documentos,
        # podrías hacer algo como esto (aunque no es el flujo estándar para FAISS.from_documents):
        # generated_embeddings = self.embed_texts(texts)
        # for i, doc in enumerate(documents):
        #     doc.metadata["embedding"] = generated_embeddings[i]
        
        return documents # Retorna los documentos sin modificar los embeddings aquí


if __name__ == "__main__":
    # Ejemplo de uso
    print("--- Probando EmbeddingsGenerator ---")
    
    # Textos de ejemplo
    sample_texts = [
        "El sistema RAG combina recuperación y generación.",
        "FAISS es una librería para búsqueda de similitud eficiente.",
        "LangChain facilita la construcción de aplicaciones LLM."
    ]

    try:
        # Inicializar el generador de embeddings (usará CPU por defecto)
        # Para usar GPU, cambia 'device': 'cuda' en model_kwargs
        embed_gen = EmbeddingsGenerator(model_kwargs={'device': 'cpu'}) 
        
        # Generar embeddings para los textos de ejemplo
        embeddings = embed_gen.embed_texts(sample_texts)

        if embeddings:
            print(f"\nEmbeddings generados para {len(embeddings)} textos.")
            print(f"Dimensión del primer embedding: {len(embeddings[0])}")
            print("Primeros 5 valores del primer embedding:")
            print(embeddings[0][:5])
            
            # Prueba con documentos (aunque FAISS los maneja internamente)
            sample_docs = [
                Document(page_content="Este es el contenido del documento uno."),
                Document(page_content="Y este es el contenido del documento dos.")
            ]
            embed_gen.embed_documents(sample_docs) # Esto solo registra un mensaje
            
        else:
            print("\nNo se pudieron generar embeddings.")

    except Exception as e:
        print(f"\nOcurrió un error durante la prueba del EmbeddingsGenerator: {e}")