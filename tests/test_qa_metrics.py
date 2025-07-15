import pytest
from src.evaluation.qa_metrics import QAMetrics
from typing import List, Dict

# `qa_metrics` fixture se define en conftest.py

def test_qa_metrics_initialization(qa_metrics):
    """
    Verifica que QAMetrics se inicializa correctamente y carga la métrica ROUGE.
    """
    assert isinstance(qa_metrics, QAMetrics)
    assert qa_metrics.rouge is not None # Se espera que 'evaluate' haya cargado ROUGE

def test_calculate_rouge_success(qa_metrics):
    """
    Prueba el cálculo de métricas ROUGE con un ejemplo simple.
    """
    predictions = ["The cat was on the mat."]
    references = [["The cat is on the mat."]]
    
    results = qa_metrics.calculate_rouge(predictions, references)
    
    assert isinstance(results, dict)
    assert "rouge1" in results
    assert "rouge2" in results
    assert "rougeL" in results
    assert "rougeLsum" in results
    
    # Los valores exactos pueden variar ligeramente con las versiones de 'evaluate',
    # pero deberían ser números flotantes.
    assert 0.0 <= results["rouge1"] <= 1.0
    assert 0.0 <= results["rougeL"] <= 1.0

def test_calculate_rouge_multiple_references(qa_metrics):
    """
    Prueba ROUGE con múltiples referencias para una misma predicción.
    """
    predictions = ["This is a test."]
    references = [["This is a test.", "Here is a test."]] # Una predicción, dos referencias
    
    results = qa_metrics.calculate_rouge(predictions, references)
    assert isinstance(results, dict)
    assert "rouge1" in results

def test_calculate_rouge_empty_lists(qa_metrics):
    """
    Verifica que `calculate_rouge` maneja listas de entrada vacías.
    """
    results = qa_metrics.calculate_rouge([], [[]])
    assert results == {}

def test_calculate_rouge_mismatched_lengths(qa_metrics):
    """
    Verifica que `calculate_rouge` lanza ValueError si las longitudes no coinciden.
    """
    predictions = ["A"]
    references = [["B"], ["C"]] # Predicciones: 1, Referencias: 2
    with pytest.raises(ValueError, match="El número de predicciones y referencias debe ser el mismo."):
        qa_metrics.calculate_rouge(predictions, references)

def test_calculate_exact_match_success(qa_metrics):
    """
    Prueba el cálculo de Exact Match con varios casos.
    """
    predictions = [
        "The capital of France is Paris.",
        "2 + 2 = 5",
        "Python is a programming language.",
        "  HELLO WORLD  " # Con espacios extra
    ]
    references = [
        "The capital of France is Paris.",
        "2 + 2 = 4",
        "python is a programming language.", # Diferente capitalización
        "hello world" # Diferente capitalización y espacios
    ]
    
    em_score = qa_metrics.calculate_exact_match(predictions, references)
    
    # Se esperan 3 aciertos de 4: (Paris, python, hello world)
    # "2 + 2 = 5" != "2 + 2 = 4"
    assert em_score == 3 / 4.0 # 0.75

def test_calculate_exact_match_case_and_space_insensitivity(qa_metrics):
    """
    Verifica que Exact Match es insensible a mayúsculas/minúsculas y espacios extra.
    """
    predictions = ["  APPLE  ", "banana"]
    references = ["apple", "Banana  "]
    em_score = qa_metrics.calculate_exact_match(predictions, references)
    assert em_score == 1.0 # Ambos deberían coincidir después de normalización

def test_calculate_exact_match_empty_lists(qa_metrics):
    """
    Verifica que `calculate_exact_match` maneja listas de entrada vacías.
    """
    em_score = qa_metrics.calculate_exact_match([], [])
    assert em_score == 0.0

def test_calculate_exact_match_mismatched_lengths(qa_metrics):
    """
    Verifica que `calculate_exact_match` lanza ValueError si las longitudes no coinciden.
    """
    predictions = ["A"]
    references = ["B", "C"]
    with pytest.raises(ValueError, match="El número de predicciones y referencias debe ser el mismo para Exact Match."):
        qa_metrics.calculate_exact_match(predictions, references)

def test_calculate_f1_score_success(qa_metrics):
    """
    Prueba el cálculo de un F1-score basado en tokens con ejemplos simples.
    """
    predictions = ["the quick brown fox"]
    references = [["a quick brown fox jumps"]]

    results = qa_metrics.calculate_f1_score(predictions, references)
    assert isinstance(results, dict)
    assert "f1" in results
    assert results["f1"] > 0.0
    
    # Calculo manual para el primer caso:
    # pred_tokens = {"the", "quick", "brown", "fox"}
    # ref_tokens = {"a", "quick", "brown", "fox", "jumps"}
    # common = {"quick", "brown", "fox"} (3)
    # precision = 3/4 = 0.75
    # recall = 3/5 = 0.6
    # F1 = 2 * (0.75 * 0.6) / (0.75 + 0.6) = 2 * 0.45 / 1.35 = 0.9 / 1.35 = 0.666...
    assert results["f1"] == pytest.approx(0.666, abs=1e-3)

def test_calculate_f1_score_empty_prediction(qa_metrics):
    """
    Verifica que el F1-score es 0.0 si la predicción está vacía.
    """
    predictions = [""]
    references = [["some reference"]]
    results = qa_metrics.calculate_f1_score(predictions, references)
    assert results["f1"] == 0.0

def test_calculate_f1_score_no_overlap(qa_metrics):
    """
    Verifica que el F1-score es 0.0 si no hay superposición de tokens.
    """
    predictions = ["apple banana"]
    references = [["orange grape"]]
    results = qa_metrics.calculate_f1_score(predictions, references)
    assert results["f1"] == 0.0

def test_calculate_f1_score_multiple_references(qa_metrics):
    """
    Verifica que el F1-score elige el mejor F1 de múltiples referencias.
    """
    predictions = ["the cat sat"]
    references = [["the cat is on the mat", "cat sat there"]] # "cat sat there" debería dar mejor F1
    
    # pred_tokens = {"the", "cat", "sat"}
    # ref1_tokens = {"the", "cat", "is", "on", "the", "mat"} -> {"the", "cat", "is", "on", "mat"}
    #   common = {"the", "cat"} (2)
    #   precision = 2/3, recall = 2/5, F1 = 2 * (2/3 * 2/5) / (2/3 + 2/5) = 2 * (4/15) / (16/15) = 8/15 / 16/15 = 8/16 = 0.5
    
    # ref2_tokens = {"cat", "sat", "there"}
    #   common = {"cat", "sat"} (2)
    #   precision = 2/3, recall = 2/3, F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * (4/9) / (4/3) = 8/9 / 12/9 = 8/12 = 0.666...
    
    results = qa_metrics.calculate_f1_score(predictions, references)
    assert results["f1"] == pytest.approx(0.666, abs=1e-3)

def test_calculate_f1_score_empty_lists(qa_metrics):
    """
    Verifica que `calculate_f1_score` maneja listas de entrada vacías.
    """
    results = qa_metrics.calculate_f1_score([], [[]])
    assert results == {"f1": 0.0}

def test_calculate_f1_score_mismatched_lengths(qa_metrics):
    """
    Verifica que `calculate_f1_score` lanza ValueError si las longitudes no coinciden.
    """
    predictions = ["A"]
    references = [["B"], ["C"]]
    with pytest.raises(ValueError, match="El número de predicciones y referencias debe ser el mismo."):
        qa_metrics.calculate_f1_score(predictions, references)