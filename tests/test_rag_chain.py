import pytest
import time
from src.rag_system.rag_chain import RAGChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from unittest.mock import MagicMock

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
    assert rag_chain.prompt is prompt_template_manager.get_qa_prompt()
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

    # Verificar que el retriever fue llamado con la pregunta
    mock_vectorstore_retriever.invoke.assert_called_once_with(sample_question)
    
    # Verificar que el LLM fue llamado. El input al LLM es el resultado del prompt.
    # Como el prompt tiene {context} y {question}, el input al LLM depende de estos.
    # El mock del retriever devuelve Documentos, que el prompt formatea.
    # No podemos verificar el contenido exacto del prompt sin más mocks,
    # pero sí podemos asegurar que invoke del LLM fue llamado.
    mock_llm.invoke.assert_called_once()
    # Podemos hacer una aserción más específica sobre el input del LLM si sabemos el formato del prompt
    # Por ejemplo, si el prompt formatea a un string, podríamos buscar ese string:
    # prompt_input_str = mock_llm.invoke.call_args[0][0]
    # assert "Contexto relevante 1." in prompt_input_str
    # assert "What is RAG?" in prompt_input_str

def test_rag_chain_invoke_with_context_and_question(rag_chain, mock_llm, prompt_template_manager):
    """
    Prueba el método `invoke_with_context_and_question` que es útil para testing directo
    sin pasar por el retriever.
    """
    mock_llm.invoke.reset_mock() # Resetear para este test
    
    custom_context = "El RAG (Retrieval-Augmented Generation) combina la recuperación de información con la generación de texto."
    input_dict = {
        "context": custom_context,
        "question": "Explica RAG."
    }

    # Recrear la cadena que `invoke_with_context_and_question` usaría para asegurar la llamada correcta
    # Este método de RAGChain no usa self.retriever. Simula una cadena más simple.
    temp_chain = rag_chain.prompt | rag_chain.llm | StrOutputParser()
    
    # Parchear el invoke de esta cadena temporal para verificar la llamada
    with patch.object(temp_chain, 'invoke', wraps=temp_chain.invoke) as mock_temp_chain_invoke:
        response = rag_chain.invoke_with_context_and_question(input_dict)

        assert isinstance(response, str)
        assert response == "Respuesta simulada del LLM." # Del mock_llm

        # Verificar que el LLM fue llamado con el input formateado por el prompt
        # Aquí podemos ser más específicos porque el `context` ya está dado como string
        mock_llm.invoke.assert_called_once()
        llm_input_content = mock_llm.invoke.call_args[0][0].messages[0].content # Acceder al contenido del HumanMessage
        assert "El RAG (Retrieval-Augmented Generation) combina la recuperación de información con la generación de texto." in llm_input_content
        assert "Explica RAG." in llm_input_content

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

def test_rag_chain_response_time_under_threshold(rag_chain, sample_question, mock_llm):
    """
    Valida que el tiempo de respuesta para una invocación de la cadena RAG
    sea menor a un umbral (ej. 2 segundos para k=5).
    Esto requiere simular el retardo del LLM y retriever.
    """
    expected_max_time = 2.0 # segundos
    
    # Simular un retardo en el retriever y el LLM
    mock_vectorstore_retriever.invoke.side_effect = lambda q: (
        time.sleep(0.1), [Document(page_content="Mocked context.")] * RETRIEVER_K
    )[1] # Retorna los documentos después del sleep
    mock_llm.invoke.side_effect = lambda p: (time.sleep(0.5), "Simulated quick response.")[1]

    start_time = time.perf_counter()
    response = rag_chain.invoke(sample_question)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    print(f"\nTiempo de respuesta simulado para RAGChain: {elapsed_time:.4f} segundos")
    
    assert elapsed_time < expected_max_time
    assert response == "Simulated quick response."
    
    # Restaurar side_effects para otros tests
    mock_vectorstore_retriever.invoke.side_effect = None
    mock_llm.invoke.side_effect = None