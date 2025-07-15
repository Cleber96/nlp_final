import logging
import os
from typing import List, Tuple, Union

# Configurar el logging para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.

    Args:
        vec1 (List[float]): El primer vector.
        vec2 (List[float]): El segundo vector.

    Returns:
        float: La similitud coseno entre los dos vectores.
               Retorna 0.0 si alguno de los vectores está vacío o si la norma es cero.
    """
    if not vec1 or not vec2:
        logger.warning("Uno o ambos vectores están vacíos. La similitud coseno no se puede calcular.")
        return 0.0

    if len(vec1) != len(vec2):
        logger.error("Los vectores deben tener la misma dimensión para calcular la similitud coseno.")
        raise ValueError("Vectors must have the same dimension.")

    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm_vec1 = sum(v**2 for v in vec1)**0.5
    norm_vec2 = sum(v**2 for v in vec2)**0.5

    if norm_vec1 == 0 or norm_vec2 == 0:
        logger.warning("Uno o ambos vectores tienen norma cero. La similitud coseno es 0.")
        return 0.0

    similarity = dot_product / (norm_vec1 * norm_vec2)
    logger.debug(f"Similitud coseno calculada: {similarity}")
    return similarity

def get_vector_store_size(faiss_path: str) -> Union[int, None]:
    """
    Estima el tamaño en bytes de un índice FAISS guardado localmente.

    Args:
        faiss_path (str): La ruta donde se encuentra el índice FAISS.

    Returns:
        Union[int, None]: El tamaño total en bytes del directorio del índice, o None si no existe.
    """
    if not os.path.exists(faiss_path):
        logger.warning(f"El directorio del índice FAISS no existe en: {faiss_path}")
        return None
    
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(faiss_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp): # Evitar enlaces simbólicos para no contar dos veces
                    total_size += os.path.getsize(fp)
        logger.info(f"Tamaño del índice FAISS en '{faiss_path}': {total_size} bytes")
        return total_size
    except Exception as e:
        logger.error(f"Error al calcular el tamaño del índice FAISS en {faiss_path}: {e}")
        return None

if __name__ == "__main__":
    print("--- Probando VectorUtils ---")

    # Prueba de calculate_cosine_similarity
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.0, 2.0, 3.0]
    vec_c = [4.0, 5.0, 6.0]
    vec_d = [0.0, 0.0, 0.0]

    print(f"\nSimilitud coseno entre {vec_a} y {vec_b}: {calculate_cosine_similarity(vec_a, vec_b)}") # Debería ser 1.0
    print(f"Similitud coseno entre {vec_a} y {vec_c}: {calculate_cosine_similarity(vec_a, vec_c)}") # Debería ser > 0 y < 1
    print(f"Similitud coseno entre {vec_a} y {vec_d}: {calculate_cosine_similarity(vec_a, vec_d)}") # Debería ser 0.0

    try:
        calculate_cosine_similarity([1, 2], [1, 2, 3])
    except ValueError as e:
        print(f"Error esperado al comparar vectores de diferente dimensión: {e}")

    # Prueba de get_vector_store_size
    temp_dir = "temp_vector_utils_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
        f.write("a" * 100)
    with open(os.path.join(temp_dir, "file2.bin"), "wb") as f:
        f.write(os.urandom(200)) # 200 bytes aleatorios

    print(f"\nCalculando tamaño del directorio '{temp_dir}'...")
    size = get_vector_store_size(temp_dir)
    print(f"Tamaño reportado: {size} bytes (esperado: 300 bytes)")

    print(f"\nCalculando tamaño de un directorio inexistente: {get_vector_store_size('non_existent_dir')}")

    # Limpiar
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Directorio temporal '{temp_dir}' eliminado.")