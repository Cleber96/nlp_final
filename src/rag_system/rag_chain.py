import logging
from typing import Dict, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RAGChain:
    """
    Clase para construir y orquestar la cadena RAG (Retrieval-Augmented Generation)
    utilizando LangChain Expression Language (LCEL).
    """
    def __init__(self, retriever: VectorStoreRetriever, llm: BaseLLM, prompt: ChatPromptTemplate):
        """
        Inicializa la cadena RAG con un retriever, un LLM y una plantilla de prompt.

        Args:
            retriever (VectorStoreRetriever): La instancia del retriever de LangChain.
            llm (BaseLLM): La instancia del LLM de LangChain.
            prompt (ChatPromptTemplate): La plantilla de prompt de LangChain.
        """
        if not isinstance(retriever, VectorStoreRetriever):
            raise TypeError("El 'retriever' debe ser una instancia de VectorStoreRetriever.")
        if not isinstance(llm, BaseLLM):
            raise TypeError("El 'llm' debe ser una instancia de BaseLLM.")
        if not isinstance(prompt, ChatPromptTemplate):
            raise TypeError("El 'prompt' debe ser una instancia de ChatPromptTemplate.")

        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.chain = self._build_chain()
        logger.info("RAGChain inicializada y cadena construida.")

    def _build_chain(self):
        """
        Construye la cadena LangChain utilizando LangChain Expression Language (LCEL).
        La cadena sigue el flujo RAG:
        1. La pregunta del usuario pasa a RunnablePassthrough.
        2. El retriever recupera el contexto basado en la pregunta.
        3. El contexto y la pregunta se combinan en un diccionario.
        4. Este diccionario se pasa a la plantilla de prompt.
        5. El prompt formateado se envía al LLM.
        6. La salida del LLM se parsea a una cadena de texto.

        Returns:
            Runnable: La cadena LangChain construida.
        """
        # Define cómo se pasa el contexto y la pregunta al prompt
        # 'context' se obtiene del retriever, 'question' se obtiene directamente de la entrada
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("Cadena RAG construida con LCEL.")
        return rag_chain

    def invoke(self, question: str) -> str:
        """
        Invoca la cadena RAG con una pregunta del usuario.

        Args:
            question (str): La pregunta del usuario.

        Returns:
            str: La respuesta generada por el sistema RAG.
        """
        if not question:
            logger.warning("La pregunta está vacía. No se invocará la cadena RAG.")
            return "Por favor, proporciona una pregunta válida."
        
        logger.info(f"Invocando la cadena RAG con la pregunta: '{question[:50]}...'")
        try:
            response = self.chain.invoke(question)
            logger.info("Invocación de la cadena RAG completada.")
            return response
        except Exception as e:
            logger.error(f"Error al invocar la cadena RAG: {e}")
            return f"Lo siento, ocurrió un error al procesar tu pregunta: {e}"

    def invoke_with_context_and_question(self, input_dict: Dict[str, Any]) -> str:
        """
        Invoca la cadena RAG con un diccionario que contiene el contexto y la pregunta.
        Este método es útil para pruebas o escenarios donde el contexto ya ha sido
        pre-cargado o manipulado antes de pasar al LLM y al prompt.

        Args:
            input_dict (Dict[str, Any]): Un diccionario con las claves 'context' y 'question'.
                                        'context' debe ser una cadena o una lista de Documentos.
                                        'question' debe ser una cadena.

        Returns:
            str: La respuesta generada por el LLM.
        """
        if "context" not in input_dict or "question" not in input_dict:
            logger.error("El diccionario de entrada debe contener las claves 'context' y 'question'.")
            raise ValueError("Input dictionary must contain 'context' and 'question' keys.")
        
        logger.info(f"Invocando la cadena RAG con contexto y pregunta predefinidos. "
                    f"Pregunta: '{input_dict['question'][:50]}...'")
        try:
            # Aquí la cadena solo necesita el prompt, el LLM y el parser,
            # ya que el contexto y la pregunta ya están en el input_dict.
            # La parte del retriever se omite en este flujo.
            response = (self.prompt | self.llm | StrOutputParser()).invoke(input_dict)
            logger.info("Invocación de la cadena RAG con contexto/pregunta predefinidos completada.")
            return response
        except Exception as e:
            logger.error(f"Error al invocar la cadena RAG con contexto y pregunta: {e}")
            return f"Lo siento, ocurrió un error al procesar tu pregunta con el contexto dado: {e}"


if __name__ == "__main__":
    print("--- Probando RAGChain ---")
    
    # Para probar RAGChain, necesitamos instancias simuladas o reales de
    # Retriever, LLM y PromptTemplate.
    
    # 1. Simular o inicializar PromptTemplateManager
    from src.rag_system.prompt_template import PromptTemplateManager
    prompt_manager = PromptTemplateManager()
    qa_prompt = prompt_manager.get_qa_prompt()
    print("Prompt template listo.")

    # 2. Simular o inicializar LLMModel
    # NOTA: Asegúrate de que tu modelo GPT4All esté descargado y en la ruta correcta
    # (definida en src/rag_system/config.py) para que esto funcione.
    try:
        from src.rag_system.llm_model import LLMModel
        llm_manager = LLMModel()
        llm_instance = llm_manager.get_llm()
        print("LLM listo.")
    except Exception as e:
        print(f"Error al inicializar LLM: {e}")
        print("Por favor, revisa src/rag_system/llm_model.py y asegúrate de que el modelo esté disponible.")
        exit()

    # 3. Simular o inicializar CustomRetriever
    # NOTA: Necesitas un índice FAISS existente para que el retriever funcione.
    # Ejecuta src/main.py --ingest o src/vector_store/faiss_store.py para crearlo.
    try:
        from src.vector_store.embeddings_generator import EmbeddingsGenerator
        from src.rag_system.retriever import CustomRetriever
        
        embed_gen = EmbeddingsGenerator()
        embeddings_instance = embed_gen.get_embeddings_model()

        custom_retriever = CustomRetriever(
            faiss_path=config.FAISS_INDEX_DIR,
            embeddings=embeddings_instance,
            k=2 # Recuperar 2 documentos para la prueba
        )
        langchain_retriever = custom_retriever.get_langchain_retriever()
        print("Retriever listo.")
    except Exception as e:
        print(f"Error al inicializar Retriever: {e}")
        print("Asegúrate de que el índice FAISS exista en la ruta: {config.FAISS_INDEX_DIR}")
        print("Puedes crearlo ejecutando src/main.py --ingest.")
        exit()

    # 4. Inicializar RAGChain
    try:
        rag_chain_instance = RAGChain(
            retriever=langchain_retriever,
            llm=llm_instance,
            prompt=qa_prompt
        )
        print("\nRAGChain inicializada.")

        # 5. Invocar la cadena con una pregunta
        question_to_ask = "¿Qué es un sistema RAG?"
        print(f"\nPregunta al sistema RAG: '{question_to_ask}'")
        response = rag_chain_instance.invoke(question_to_ask)
        print(f"\nRespuesta del sistema RAG:\n{response.strip()}")

        # 6. Probar invoke_with_context_and_question
        print("\n--- Probando invoke_with_context_and_question ---")
        simulated_input = {
            "context": "Un sistema RAG (Retrieval-Augmented Generation) combina la recuperación de información con la generación de texto para mejorar la calidad de las respuestas de los LLMs. Utiliza un retriever para encontrar información relevante y un generador para formular la respuesta.",
            "question": "¿Cómo funciona RAG?"
        }
        print(f"Pregunta con contexto simulado: '{simulated_input['question']}'")
        response_simulated = rag_chain_instance.invoke_with_context_and_question(simulated_input)
        print(f"\nRespuesta con contexto simulado:\n{response_simulated.strip()}")

    except Exception as e:
        print(f"\nOcurrió un error inesperado durante la prueba de RAGChain: {e}")