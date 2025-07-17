# tests/test_rag_chain.py
import pytest
import time
from unittest.mock import MagicMock, patch, ANY
from src.rag_system.rag_chain import RAGChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# Definir RETRIEVER_K. Ajusta este valor si tu retriever tiene una 'k' específica.
RETRIEVER_K = 2 # Por ejemplo, si tu retriever devuelve 2 documentos por defecto.

# Fixtures `mock_vectorstore_retriever`, `mock_llm`, `prompt_template_manager` se definen en conftest.py

@pytest.fixture
def sample_question():
    return "What is RAG?"

def test_rag_chain_initialization(rag_chain, mock_vectorstore_retriever, mock_llm, prompt_template_manager):
    """
    Verifica que la RAGChain se inicializa correctamente con sus componentes.
    """
    assert rag_chain.retriever is mock_vectorstore_retriever
    assert rag_chain.llm is mock_llm
    assert rag_chain.prompt == prompt_template_manager.get_qa_prompt()
    assert rag_chain.chain is not None # La cadena debe estar construida

def test_rag_chain_build_chain_structure(rag_chain):
    """
    Verifica que la cadena LangChain se construye con la estructura esperada
    (context, question, prompt, llm, parser).
    """
    # No podemos inspeccionar la estructura interna exacta de LCEL fácilmente,
    # pero podemos verificar que es un objeto de tipo Runnable y que los componentes
    # interactúan como se espera en la invocación.
    assert hasattr(rag_chain.chain, 'invoke')
    # Un test más detallado requeriría inspeccionar la composición de los Runnables,
    # pero eso puede ser frágil con los cambios internos de LangChain.
    # Confiamos en que la invocación exitosa y la llamada a los mocks son suficientes.

def test_rag_chain_invoke_success(rag_chain, sample_question, mock_vectorstore_retriever, mock_llm):
    """
    Prueba la invocación de la cadena RAG con una pregunta,
    asegurando que el retriever y el LLM son llamados.
    """
    # Resetear mocks para asegurar que solo se cuentan las llamadas de este test
    mock_vectorstore_retriever.invoke.reset_mock()
    mock_llm.invoke.reset_mock()

    response = rag_chain.invoke(sample_question)

    assert isinstance(response, str)
    assert response == "Respuesta simulada del LLM." # Basado en el mock_llm

    # Verificar que el retriever fue llamado con la pregunta.
    # Usamos assert_called_once() y luego verificamos el primer argumento posicional.
    mock_vectorstore_retriever.invoke.assert_called_once()
    assert mock_vectorstore_retriever.invoke.call_args[0][0] == sample_question

    # Verificar que el LLM fue llamado.
    mock_llm.invoke.assert_called_once()

def test_rag_chain_invoke_with_context_and_question(rag_chain, mock_llm, prompt_template_manager):
    """
    Prueba el método `invoke_with_context_and_question` que es útil para testing directo
    sin pasar por el retriever.
    """
    mock_llm.invoke.reset_mock() # Resetear para este test

    custom_context = "El RAG (Retrieval-Augmented Generation) combina la recuperación de información con la generación de texto."
    question = "Explica RAG."

    # Se asume que `invoke_with_context_and_question` en tu RAGChain
    # espera un diccionario como entrada, según los errores anteriores.
    input_for_method = {
        "context": custom_context,
        "question": question
    }
    response = rag_chain.invoke_with_context_and_question(input_for_method)

    assert isinstance(response, str)
    assert response == "Respuesta simulada del LLM." # Del mock_llm

    # Verificar que el LLM fue llamado.
    mock_llm.invoke.assert_called_once()

    # Opcional: Verificar el contenido del prompt que se le pasó al LLM
    # LangChain pasa un objeto al LLM (ej. ChatPromptValue) que contiene los mensajes.
    llm_input_object = mock_llm.invoke.call_args[0][0]

    # Verifica que el objeto tiene el atributo 'messages' (si es un ChatPromptValue)
    assert hasattr(llm_input_object, 'messages')
    assert len(llm_input_object.messages) > 0

    # Accede al contenido del primer mensaje (que debería ser el HumanMessage con el prompt completo)
    formatted_prompt_content = llm_input_object.messages[0].content

    assert custom_context in formatted_prompt_content
    assert question in formatted_prompt_content

def test_rag_chain_response_format_placeholder(rag_chain, sample_question, mock_llm):
    """
    Prueba que el formato de respuesta del LLM es manejado.
    Dado que el LLM está mockeado para devolver siempre "Respuesta simulada del LLM.",
    la prueba simplemente verifica que la cadena retorna este valor.
    Para una prueba real de formato ("No tengo suficiente información"),
    se necesitaría un LLM real o un mock más inteligente que pueda simular
    esas respuestas condicionalmente.
    """
    mock_llm.invoke.return_value = "No tengo suficiente información para responder con el contexto proporcionado."
    response = rag_chain.invoke(sample_question)
    assert "No tengo suficiente información" in response

def test_rag_chain_response_time_under_threshold(rag_chain, sample_question, mock_llm, mock_vectorstore_retriever):
    """
    Valida que el tiempo de respuesta para una invocación de la cadena RAG
    sea menor a un umbral (ej. 2 segundos para k=5).
    Esto requiere simular el retardo del LLM y retriever.
    """
    expected_max_time = 2.0 # segundos

    # Simular un retardo en el retriever y el LLM
    # Usamos *args, **kwargs para aceptar cualquier argumento posicional o de palabra clave.
    mock_vectorstore_retriever.invoke.side_effect = lambda *args, **kwargs: (
        time.sleep(0.1), [Document(page_content="Mocked context.")] * RETRIEVER_K
    )[1] # Retorna los documentos después del sleep

    mock_llm.invoke.side_effect = lambda *args, **kwargs: (time.sleep(0.5), "Simulated quick response.")[1]

    start_time = time.perf_counter()
    response = rag_chain.invoke(sample_question)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nTiempo de respuesta simulado para RAGChain: {elapsed_time:.4f} segundos")

    assert elapsed_time < expected_max_time
    assert response == "Simulated quick response."

    # Restaurar side_effects para otros tests (buena práctica)
    mock_vectorstore_retriever.invoke.side_effect = None
    mock_llm.invoke.side_effect = None