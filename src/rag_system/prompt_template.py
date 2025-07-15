import logging
from langchain_core.prompts import ChatPromptTemplate

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PromptTemplateManager:
    """
    Gestiona las plantillas de prompt utilizadas en el sistema RAG.
    """
    def __init__(self):
        logger.info("PromptTemplateManager inicializado.")

    def get_qa_prompt(self) -> ChatPromptTemplate:
        """
        Retorna el template de prompt para el sistema de Preguntas y Respuestas (QA).
        Este prompt instruye al LLM sobre cómo usar el contexto para responder.

        Returns:
            ChatPromptTemplate: La plantilla de prompt configurada.
        """
        template = """Eres un asistente útil y experto en documentación técnica.
        Utiliza los siguientes fragmentos de contexto recuperados para responder a la pregunta.
        Si no sabes la respuesta basándote únicamente en el contexto proporcionado,
        di que no tienes suficiente información para responder y evita inventar datos.
        Mantén la respuesta concisa, profesional y directamente relacionada con la pregunta.

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta:
        """
        prompt = ChatPromptTemplate.from_template(template)
        logger.info("Plantilla de prompt QA cargada.")
        return prompt

    def get_refine_prompt(self) -> ChatPromptTemplate:
        """
        (Opcional) Retorna un template de prompt para refinar respuestas
        en un escenario de múltiples pasadas o para resumir.
        """
        template = """Dada la siguiente conversación y una nueva pregunta,
        genera una respuesta refinada. Si la nueva pregunta no se relaciona con la conversación anterior,
        responde la nueva pregunta directamente.

        Historial de Conversación:
        {chat_history}

        Contexto adicional:
        {context}

        Nueva Pregunta: {question}

        Respuesta Refinada:
        """
        prompt = ChatPromptTemplate.from_template(template)
        logger.info("Plantilla de prompt de refinamiento cargada (opcional).")
        return prompt

if __name__ == "__main__":
    print("--- Probando PromptTemplateManager ---")
    prompt_manager = PromptTemplateManager()

    qa_prompt = prompt_manager.get_qa_prompt()
    print("\nPlantilla de prompt QA:")
    print(qa_prompt.messages[0].prompt.template) # Acceder al template de la primera parte del mensaje

    # Ejemplo de cómo se vería el prompt con datos simulados
    simulated_context = "El LangChain Expression Language (LCEL) permite construir cadenas composables."
    simulated_question = "¿Qué es LCEL?"
    
    # Para ver el prompt formateado, necesitarías un LLM o usar .format_prompt().to_string()
    # Esto es solo una demostración de la estructura.
    print("\nEjemplo de prompt formateado (estructura):")
    print(f"Contexto: {simulated_context}")
    print(f"Pregunta: {simulated_question}")
    print("Respuesta: (generada por LLM)")

    # Prueba de la plantilla de refinamiento (si se implementó)
    refine_prompt = prompt_manager.get_refine_prompt()
    print("\nPlantilla de prompt de refinamiento (opcional):")
    print(refine_prompt.messages[0].prompt.template)