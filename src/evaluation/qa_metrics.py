import logging
from typing import List, Dict, Union
import evaluate # Requiere 'pip install evaluate'

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class QAMetrics:
    """
    Clase para calcular métricas de evaluación de sistemas de Preguntas y Respuestas (QA).
    Incluye métricas como ROUGE y Exact Match.
    """
    def __init__(self):
        """
        Inicializa la clase cargando las métricas necesarias de la librería 'evaluate'.
        """
        try:
            self.rouge = evaluate.load("rouge")
            logger.info("Métrica ROUGE cargada exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar la métrica ROUGE: {e}")
            self.rouge = None

        # Nota: La métrica F1 Score del paquete `evaluate` es más comúnmente utilizada
        # para clasificación o para QA con respuestas basadas en extracción (span-based).
        # Para QA generativa, ROUGE y Exact Match son más directamente aplicables.
        # Si se necesitara un F1 más sofisticado para QA generativa, a menudo se implementa
        # una comparación basada en tokens o un F1 de "respuesta de alto nivel".
        logger.info("QAMetrics inicializado.")

    def calculate_rouge(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calcula las métricas ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
        ROUGE mide la superposición entre la respuesta generada y las respuestas de referencia.

        Args:
            predictions (List[str]): Lista de respuestas generadas por el modelo.
            references (List[List[str]]): Lista de listas de respuestas de referencia.
                                          Cada sublista puede contener una o más referencias válidas
                                          para la predicción correspondiente.

        Returns:
            Dict[str, float]: Un diccionario con los resultados de ROUGE (ej., 'rouge1', 'rouge2', 'rougeL', 'rougeLsum').
                              Los valores suelen estar entre 0 y 1.
        """
        if self.rouge is None:
            logger.warning("ROUGE no está disponible. No se pueden calcular las métricas ROUGE.")
            return {}
        
        if not predictions or not references:
            logger.warning("Listas de predicciones o referencias vacías para ROUGE. Retornando vacío.")
            return {}

        if len(predictions) != len(references):
            logger.error("El número de predicciones y referencias no coincide para ROUGE.")
            raise ValueError("El número de predicciones y referencias debe ser el mismo.")

        try:
            results = self.rouge.compute(predictions=predictions, references=references)
            logger.info(f"Métricas ROUGE calculadas: {results}")
            return results
        except Exception as e:
            logger.error(f"Error al calcular métricas ROUGE: {e}")
            return {}

    def calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        Calcula la métrica Exact Match (EM).
        Exact Match es 1 si la predicción coincide exactamente con una de las referencias (ignorando mayúsculas/minúsculas y espacios en blanco),
        y 0 en caso contrario.

        Args:
            predictions (List[str]): Lista de respuestas generadas por el modelo.
            references (List[str]): Lista de respuestas de referencia (una por cada predicción).
                                    Para Exact Match, generalmente se espera una única referencia por predicción.

        Returns:
            float: El porcentaje de coincidencias exactas (entre 0.0 y 1.0).
        """
        if not predictions or not references:
            logger.warning("Listas de predicciones o referencias vacías para Exact Match. Retornando 0.0.")
            return 0.0

        if len(predictions) != len(references):
            logger.error("El número de predicciones y referencias no coincide para Exact Match.")
            raise ValueError("El número de predicciones y referencias debe ser el mismo para Exact Match.")

        exact_matches = 0
        for pred, ref in zip(predictions, references):
            # Normalizar: eliminar espacios extra y convertir a minúsculas para una comparación robusta
            normalized_pred = ' '.join(pred.strip().lower().split())
            normalized_ref = ' '.join(ref.strip().lower().split())
            
            if normalized_pred == normalized_ref:
                exact_matches += 1
        
        em_score = exact_matches / len(predictions) if predictions else 0.0
        logger.info(f"Métrica Exact Match calculada: {em_score:.4f}")
        return em_score

    def calculate_f1_score(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calcula una versión simplificada del F1-Score basada en tokens, común en QA extractiva.
        Para QA generativa, el F1 a menudo se basa en la superposición de tokens entre
        la predicción y la(s) referencia(s). Esta implementación es una adaptación.

        Args:
            predictions (List[str]): Lista de respuestas generadas.
            references (List[List[str]]): Lista de listas de respuestas de referencia.

        Returns:
            Dict[str, float]: Un diccionario con el F1 score promedio.
        """
        # Nota: Para una implementación de F1 más rigurosa y estándar en QA,
        # se suele usar la lógica de SQuAD v1.1. Esta es una aproximación.
        # Si se usa el paquete `evaluate`, 'f1' es más para clasificación binaria/multiclase.

        def _calculate_f1_for_pair(prediction: str, references: List[str]) -> float:
            pred_tokens = set(prediction.strip().lower().split())
            
            if not pred_tokens: # Si la predicción está vacía, F1 es 0
                return 0.0
            
            best_f1 = 0.0
            for ref in references:
                ref_tokens = set(ref.strip().lower().split())

                common_tokens = len(pred_tokens.intersection(ref_tokens))
                
                if common_tokens == 0:
                    continue
                
                precision = common_tokens / len(pred_tokens)
                recall = common_tokens / len(ref_tokens)
                
                f1 = (2 * precision * recall) / (precision + recall)
                best_f1 = max(best_f1, f1)
            return best_f1

        if not predictions or not references:
            logger.warning("Listas de predicciones o referencias vacías para F1. Retornando vacío.")
            return {"f1": 0.0}

        if len(predictions) != len(references):
            logger.error("El número de predicciones y referencias no coincide para F1.")
            raise ValueError("El número de predicciones y referencias debe ser el mismo.")

        f1_scores = []
        for pred, refs in zip(predictions, references):
            f1_scores.append(_calculate_f1_for_pair(pred, refs))
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logger.info(f"Métrica F1 Score calculada (promedio): {avg_f1:.4f}")
        return {"f1": avg_f1}


if __name__ == "__main__":
    print("--- Probando QAMetrics ---")
    metrics_calculator = QAMetrics()

    # Ejemplos de datos de prueba
    predictions_rouge = [
        "El gato se sienta en la alfombra.",
        "El perro ladra en voz alta.",
        "Las manzanas son frutas saludables."
    ]
    references_rouge = [
        ["El gato está en la alfombra.", "Un gato se sienta en la alfombra."],
        ["El perro ladra muy fuerte.", "Un canino ladra fuerte."],
        ["Las manzanas son frutas nutritivas y saludables.", "Las manzanas son buenas frutas."]
    ]

    predictions_em = [
        "La capital de Francia es París.",
        "2 + 2 = 5",
        "Python es un lenguaje de programación."
    ]
    references_em = [
        "La capital de francia es paris.", # Coincidencia exacta (normalizada)
        "2 + 2 = 4",                       # No coincide
        "python es un lenguaje de programacion." # Coincidencia exacta (normalizada)
    ]

    predictions_f1 = [
        "El cielo es azul.",
        "Los pájaros vuelan alto."
    ]
    references_f1 = [
        ["El cielo es de color azul.", "Azul es el cielo."],
        ["Pájaros vuelan en el cielo.", "Los pájaros pueden volar muy alto."]
    ]

    # Calcular y mostrar métricas ROUGE
    print("\nCalculando ROUGE scores:")
    rouge_results = metrics_calculator.calculate_rouge(predictions_rouge, references_rouge)
    for key, value in rouge_results.items():
        print(f"  {key}: {value:.4f}")

    # Calcular y mostrar Exact Match
    print("\nCalculando Exact Match:")
    em_score = metrics_calculator.calculate_exact_match(predictions_em, references_em)
    print(f"  Exact Match Score: {em_score:.4f}")

    # Calcular y mostrar F1 Score
    print("\nCalculando F1 Score (aproximado):")
    f1_results = metrics_calculator.calculate_f1_score(predictions_f1, references_f1)
    print(f"  F1 Score: {f1_results.get('f1', 0.0):.4f}")

    # Pruebas con listas vacías o de longitud diferente
    print("\n--- Pruebas de errores/casos especiales ---")
    print("ROUGE con listas vacías:", metrics_calculator.calculate_rouge([], [[]]))
    try:
        metrics_calculator.calculate_rouge(["a"], [["b"], ["c"]])
    except ValueError as e:
        print(f"ROUGE con longitud diferente (esperado error): {e}")

    print("Exact Match con listas vacías:", metrics_calculator.calculate_exact_match([], []))
    try:
        metrics_calculator.calculate_exact_match(["a"], ["b", "c"])
    except ValueError as e:
        print(f"Exact Match con longitud diferente (esperado error): {e}")

    print("F1 con listas vacías:", metrics_calculator.calculate_f1_score([], [[]]))
    try:
        metrics_calculator.calculate_f1_score(["a"], [["b"], ["c"]])
    except ValueError as e:
        print(f"F1 con longitud diferente (esperado error): {e}")