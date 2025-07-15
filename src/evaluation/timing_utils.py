import time
import logging
from functools import wraps
from typing import Callable, Any, List, Dict

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def timing_decorator(func: Callable) -> Callable:
    """
    Un decorador que mide el tiempo de ejecución de una función.
    Imprime el tiempo de ejecución de la función decorada.

    Args:
        func (Callable): La función a la que se le medirá el tiempo.

    Returns:
        Callable: La función envuelta con la funcionalidad de medición de tiempo.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Función '{func.__name__}' ejecutada en {elapsed_time:.4f} segundos.")
        return result
    return wrapper

def measure_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Ejecuta una función y retorna su resultado junto con el tiempo de ejecución.

    Args:
        func (Callable): La función a ejecutar.
        *args: Argumentos posicionales para la función.
        **kwargs: Argumentos de palabra clave para la función.

    Returns:
        tuple[Any, float]: Una tupla que contiene el resultado de la función y
                           el tiempo de ejecución en segundos.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.debug(f"measure_time: Función '{func.__name__}' ejecutada en {elapsed_time:.4f} segundos.")
    return result, elapsed_time

def calculate_percentiles(data: List[float], percentiles: List[float] = None) -> Dict[str, float]:
    """
    Calcula los percentiles para una lista de datos numéricos.

    Args:
        data (List[float]): Una lista de números (ej. tiempos de respuesta).
        percentiles (List[float], optional): Una lista de percentiles a calcular (ej. [50, 95, 99]).
                                             Por defecto, calcula P50, P95, P99.

    Returns:
        Dict[str, float]: Un diccionario donde las claves son las etiquetas de los percentiles
                          y los valores son los tiempos correspondientes.
    """
    if not data:
        logger.warning("No hay datos para calcular percentiles. Retornando vacío.")
        return {}

    if percentiles is None:
        percentiles = [50, 95, 99]

    sorted_data = sorted(data)
    results = {}
    n = len(sorted_data)

    for p in percentiles:
        if not (0 <= p <= 100):
            logger.warning(f"Percentil inválido: {p}. Debe estar entre 0 y 100.")
            continue
        
        k = (n - 1) * p / 100
        f = int(k)
        c = k - f
        
        if f < 0:  # Handle edge case for p=0
            percentile_value = sorted_data[0]
        elif f >= n - 1: # Handle edge case for p=100
            percentile_value = sorted_data[n - 1]
        else:
            percentile_value = sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
        
        results[f"P{p}"] = percentile_value
    
    logger.info(f"Percentiles calculados para {len(data)} datos: {results}")
    return results

if __name__ == "__main__":
    print("--- Probando timing_utils ---")

    @timing_decorator
    def example_function(delay_sec: float):
        """Una función de ejemplo para probar el decorador de tiempo."""
        logger.info(f"Ejecutando función de ejemplo con retardo de {delay_sec} segundos...")
        time.sleep(delay_sec)
        return "Tarea completada"

    # Prueba del decorador
    result = example_function(0.5)
    print(f"Resultado de la función decorada: {result}")

    result_2 = example_function(0.1)
    print(f"Resultado de la función decorada: {result_2}")

    # Prueba de measure_time
    print("\n--- Probando measure_time ---")
    def another_example_function(iterations: int):
        total = 0
        for i in range(iterations):
            total += i
        return total

    func_result, elapsed = measure_time(another_example_function, iterations=1000000)
    print(f"Resultado de another_example_function: {func_result}, Tiempo: {elapsed:.4f} segundos")

    # Prueba de calculate_percentiles
    print("\n--- Probando calculate_percentiles ---")
    sample_times = [0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.12, 0.22, 0.08, 0.18]
    print(f"Tiempos de muestra: {sample_times}")
    
    percentiles_results = calculate_percentiles(sample_times)
    print(f"Percentiles estándar (P50, P95, P99): {percentiles_results}")

    custom_percentiles = calculate_percentiles(sample_times, percentiles=[10, 90])
    print(f"Percentiles personalizados (P10, P90): {custom_percentiles}")

    empty_percentiles = calculate_percentiles([])
    print(f"Percentiles con lista vacía: {empty_percentiles}")

    single_value_percentiles = calculate_percentiles([1.0], percentiles=[50, 100])
    print(f"Percentiles con un solo valor: {single_value_percentiles}")
    