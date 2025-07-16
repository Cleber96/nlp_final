# benckmarks/bench_qa_performance.py
import os
import json
import pandas as pd
import time
import numpy as np
import warnings
from datetime import datetime
import psutil # Para medir uso de RAM
import logging # Importar logging para usarlo

# Configurar el logging para este script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Ignorar advertencias de LangChain y otros módulos si son muy ruidosas durante el benchmark
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuración de rutas (relativas a la raíz del proyecto) ---
# Obtener la ruta absoluta de la raíz del proyecto
# Esto asume que el script bench_qa_performance.py está en benchmarks/
# y la raíz del proyecto es el directorio padre de 'benchmarks/'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DOCS_PATH = os.path.join(PROJECT_ROOT, "docs") # Carpeta con documentos PDF/Markdown para indexar
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index") # Donde se guardará/cargará el índice FAISS
QA_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "qa_test_set.json") # Dataset de preguntas y respuestas para evaluación
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results") # Directorio para guardar los resultados del benchmark

# --- CONFIGURACIÓN DEL MODELO LLM ---
# Nombre del modelo en Hugging Face Hub 
LLM_MODEL_NAME = "EleutherAI/gpt-neo-125M"
# Ruta LOCAL donde se descargará/cargará el modelo si lo tienes guardado.
# Si lo dejas como None, LLMModel buscará el modelo por LLM_MODEL_NAME directamente del Hub.
# Para EleutherAI/gpt-neo-125M, dejar None es lo común.
LLM_MODEL_LOCAL_PATH = None
# Dispositivo a usar para la inferencia del LLM: "auto" (GPU si disponible), "cuda" (forzar GPU), "cpu" (forzar CPU)
# Para EleutherAI/gpt-neo-125M, "cpu" es muy eficiente. "auto" también funcionará.
LLM_DEVICE = "cpu" # Forzado a CPU para mayor estabilidad en demos sin GPU potente

# Rangos de K para evaluar el retriever
K_VALUES = [1, 3, 5, 7, 10]

# --- Importar componentes de nuestro sistema RAG ---
# Se importan desde la raíz del proyecto debido a PYTHONPATH en run_bench.sh
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
from src.rag_system.llm_model import LLMModel # Importar LLMModel
from src.rag_system.retriever import CustomRetriever
# Asegúrate de que esta clase exista en src/rag_system/rag_chain.py
from src.rag_system.rag_chain import RAGChain 

# --- Importar utilidades de evaluación ---
# Asegúrate de que estos archivos existan en src/evaluation/
from src.evaluation.qa_metrics import QAMetrics # Necesita HuggingFace Evaluate
from src.evaluation.timing_utils import measure_time, calculate_percentiles
# CAMBIO CLAVE AQUÍ: Importar las funciones correctas de memory_utils
from src.evaluation.memory_utils import get_current_process_memory_usage, get_gpu_memory_usage


# --- Funciones de Utilidad ---
def setup_rag_system(k_value: int, load_faiss_if_exists: bool = True):
    """
    Inicializa y configura el sistema RAG con un valor 'k' específico.
    Si load_faiss_if_exists es True, intenta cargar un índice FAISS existente.
    Si no existe o load_faiss_if_exists es False, lo crea desde los documentos en DOCS_PATH.
    """
    print(f"Configurando sistema RAG para k={k_value}...")

    # 1. Embeddings y FAISS
    embeddings_generator = EmbeddingsGenerator()
    embeddings = embeddings_generator.embeddings # Obtener la instancia de embeddings de HuggingFace
    faiss_store = FAISSStore(embeddings=embeddings)

    # Lógica para cargar o crear el índice FAISS
    faiss_index_file = f"{FAISS_INDEX_PATH}.faiss"
    faiss_pkl_file = f"{FAISS_INDEX_PATH}.pkl"

    if load_faiss_if_exists and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
        try:
            faiss_store.load_local(FAISS_INDEX_PATH)
            print(f"Índice FAISS cargado desde {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Error al cargar índice FAISS desde {FAISS_INDEX_PATH}: {e}. Procediendo a crearlo.")
            load_faiss_if_exists = False # Forzar creación
    
    if not load_faiss_if_exists or not (os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file)):
        print(f"Creando índice FAISS desde documentos en {DOCS_PATH}...")
        document_loader = DocumentLoader()
        text_splitter = TextSplitter()
        
        all_documents = []
        if not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(f"La carpeta de documentos '{DOCS_PATH}' no existe. Por favor, crea la carpeta y añade documentos o ajusta DOCS_PATH.")
        
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(".pdf"):
                    all_documents.extend(document_loader.load_pdf(file_path))
                elif file.lower().endswith(".md"):
                    all_documents.extend(document_loader.load_markdown(file_path))
        
        if not all_documents:
            # Crear un documento dummy si la carpeta está vacía y es para el benchmark.
            # Esto evita un error, pero para un benchmark real, la calidad de la respuesta será nula.
            dummy_doc_path = os.path.join(DOCS_PATH, "dummy_doc_for_benchmark.md")
            os.makedirs(DOCS_PATH, exist_ok=True)
            with open(dummy_doc_path, "w", encoding="utf-8") as f:
                f.write("Este es un documento de prueba para el benchmark de un sistema RAG. Contiene información básica sobre LangChain, embeddings y almacenes vectoriales como FAISS. Un sistema RAG (Retrieval Augmented Generation) mejora los LLMs.")
            print(f"Advertencia: No se encontraron documentos reales. Se ha creado un documento dummy en {dummy_doc_path}. Se recomienda añadir documentos reales para un benchmark significativo.")
            all_documents.extend(document_loader.load_markdown(dummy_doc_path))


        chunks = text_splitter.split_documents(all_documents)
        print(f"Documentos cargados y divididos en {len(chunks)} fragmentos.")
        
        faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH)
        print("Índice FAISS creado y guardado.")

    # 2. Retriever
    # Se pasa faiss_store.index directamente para asegurar que se usa el índice cargado/creado
    retriever = CustomRetriever(faiss_path=FAISS_INDEX_PATH, embeddings=embeddings, k=k_value)
    langchain_retriever = retriever.get_langchain_retriever()
    print(f"Retriever configurado con k={k_value}")

    # 3. LLM (Modelo ligero)
    llm_model = LLMModel(model_name=LLM_MODEL_NAME, model_path=LLM_MODEL_LOCAL_PATH, device=LLM_DEVICE)
    llm = llm_model.get_llm()
    print(f"LLM inicializado: {LLM_MODEL_NAME}.")

    # 4. Prompt Template
    prompt_manager = PromptTemplateManager()
    qa_prompt = prompt_manager.get_qa_prompt()
    print("Prompt template cargado.")

    # 5. RAG Chain
    rag_chain = RAGChain(retriever=langchain_retriever, llm=llm, prompt=qa_prompt)
    print("Cadena RAG construida.")
    
    return rag_chain

def load_qa_dataset(filepath: str):
    """Carga el dataset de preguntas y respuestas desde un archivo JSON."""
    if not os.path.exists(filepath):
        # Crear un dataset QA dummy si no existe para que el benchmark no falle
        print(f"Advertencia: Dataset QA no encontrado en: {filepath}. Creando un dataset dummy.")
        dummy_data = [
            {"question": "¿Qué es RAG?", "answer": "RAG significa Retrieval Augmented Generation.", "references": []},
            {"question": "¿Para qué sirve FAISS?", "answer": "FAISS se usa para búsqueda eficiente de similitud vectorial.", "references": []}
        ]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, indent=2)
        print("Dataset QA dummy creado. Se recomienda crear un dataset real para métricas significativas.")
        return dummy_data
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Dataset QA cargado: {len(data)} ejemplos.")
    return data

def run_benchmark():
    """
    Ejecuta el proceso de benchmarking completo.
    """
    print("Iniciando benchmark de rendimiento del sistema RAG...")
    
    # Crear el directorio de resultados si no existe
    os.makedirs(RESULTS_DIR, exist_ok=True)

    qa_dataset = load_qa_dataset(QA_DATASET_PATH)
    qa_metrics_calculator = QAMetrics()

    all_k_results = []
    
    # Listas para almacenar tiempos y usos de memoria de cada pregunta
    all_response_times_by_k = {}
    all_memory_usages_by_k = {}
    all_gpu_memory_usages_by_k = {}

    for k_value in K_VALUES:
        print(f"\n--- Ejecutando benchmark para k={k_value} ---")
        
        # Setup RAG system for the current k
        try:
            # Asegurarse de que el índice se cargue/cree correctamente
            rag_system = setup_rag_system(k_value=k_value, load_faiss_if_exists=True)
        except Exception as e:
            print(f"Error fatal al configurar el sistema RAG para k={k_value}: {e}. Saltando este valor de k.")
            continue

        predictions = []
        references = []
        current_k_response_times = []
        current_k_memory_usages = []
        current_k_gpu_memory_usages = []
        
        for i, entry in enumerate(qa_dataset):
            question = entry["question"]
            # Las referencias pueden ser una lista de respuestas válidas
            reference_context = [entry["answer"]] + entry.get("references", [])

            # --- Medir memoria antes de la invocación ---
            mem_before_dict = get_current_process_memory_usage()
            mem_before_mb = mem_before_dict.get('rss_mb', 0.0) # Uso de RAM (RSS) del proceso

            gpu_mem_before_dict = get_gpu_memory_usage()
            # Asumiendo que quieres el uso de VRAM de la primera GPU (gpu_0)
            gpu_mem_before_mb = gpu_mem_before_dict.get('gpu_0', {}).get('used_vram_mb', 0.0)
            if "error" in gpu_mem_before_dict:
                gpu_mem_before_mb = 0.0 
                logger.debug(f"No se pudo obtener VRAM antes: {gpu_mem_before_dict['error']}")

            # Usar la función measure_time de timing_utils.py
            try:
                # CAMBIO CLAVE 1: Pasa solo la cadena de texto de la pregunta a rag_system.invoke()
                generated_answer_raw, elapsed_time_sec = measure_time(rag_system.invoke, question)

                # Añade este print para ver la salida cruda, incluso si es None o un objeto inesperado
                print(f"DEBUG: Salida cruda de rag_system.invoke para '{question[:30]}...': {generated_answer_raw} (Tipo: {type(generated_answer_raw)})")

                # Ahora, procesa la salida
                if isinstance(generated_answer_raw, str): # Si directamente es un string (lo esperado ahora)
                    generated_answer = generated_answer_raw
                # Esto es un fallback, si por alguna razón rag_chain.py devolviera un dict, aunque no debería
                elif isinstance(generated_answer_raw, dict) and "answer" in generated_answer_raw:
                    generated_answer = generated_answer_raw["answer"]
                else:
                    # Si no es un string ni un dict con "answer", lo convertimos a string para el log
                    generated_answer = f"UNEXPECTED_OUTPUT_TYPE: {str(generated_answer_raw)}"
                    logger.warning(f"Advertencia: La RAGChain devolvió un tipo inesperado para '{question[:50]}...': {type(generated_answer_raw)}")

            except Exception as e:
                # Aquí es donde vamos a obtener la información crucial
                logger.error(f"Error CRÍTICO al invocar RAGChain para pregunta '{question[:50]}...': {type(e).__name__}: {e}")
                generated_answer = f"ERROR_GENERATION: {e}" # Incluir el error para debugging
                elapsed_time_sec = 0.0 
            
            response_time_ms = elapsed_time_sec * 1000 # Convertir a milisegundos

            # --- Medir memoria después de la invocación ---
            mem_after_dict = get_current_process_memory_usage()
            mem_after_mb = mem_after_dict.get('rss_mb', 0.0)
            
            # El pico de RAM es el máximo entre antes y después para el proceso.
            peak_memory_mb = max(mem_before_mb, mem_after_mb)
            
            gpu_mem_after_dict = get_gpu_memory_usage()
            # Asumiendo que quieres el uso de VRAM de la primera GPU (gpu_0)
            gpu_mem_after_mb = gpu_mem_after_dict.get('gpu_0', {}).get('used_vram_mb', 0.0)
            if "error" in gpu_mem_after_dict:
                gpu_mem_after_mb = 0.0
                logger.debug(f"No se pudo obtener VRAM después: {gpu_mem_after_dict['error']}")

            # El pico de VRAM es el máximo entre antes y después para la primera GPU.
            peak_gpu_memory_mb = max(gpu_mem_before_mb, gpu_mem_after_mb)


            predictions.append(generated_answer)
            references.append(reference_context)

            current_k_response_times.append(response_time_ms)
            current_k_memory_usages.append(peak_memory_mb)
            current_k_gpu_memory_usages.append(peak_gpu_memory_mb)

            print(f"  Q: {question[:70]}...")
            print(f"  A (Generada): {generated_answer[:70].replace('\n', ' ')}...") # Limpiar nueva línea para log
            print(f"  Tiempo: {response_time_ms:.2f} ms, Memoria: {peak_memory_mb:.2f} MB, GPU Mem: {peak_gpu_memory_mb:.2f} MB")

        # Almacenar los tiempos y usos de memoria para este k
        all_response_times_by_k[k_value] = current_k_response_times
        all_memory_usages_by_k[k_value] = current_k_memory_usages
        all_gpu_memory_usages_by_k[k_value] = current_k_gpu_memory_usages

        # Calcular métricas de exactitud para el lote actual de k
        current_k_summary = {"k": k_value}
        
        # Filtrar predicciones y referencias para asegurar que no haya errores de generación
        # Las referencias deben ser listas de strings
        valid_predictions = [p for p in predictions if p != "ERROR_GENERATION" and not p.startswith("ERROR_GENERATION:") and isinstance(p, str)]
        
        # Solo incluye las referencias si la predicción correspondiente es válida
        valid_references_final = []
        for idx, p in enumerate(predictions):
            if p in valid_predictions: # Si la predicción es válida
                refs_for_pred = [ref for ref in references[idx] if isinstance(ref, str)]
                if refs_for_pred: # Asegura que hay al menos una referencia válida
                    valid_references_final.append(refs_for_pred)


        if valid_predictions and valid_references_final and len(valid_predictions) == len(valid_references_final):
            rouge_results = qa_metrics_calculator.calculate_rouge(valid_predictions, valid_references_final)
            # Para Exact Match, se suele comparar con una sola referencia (la principal)
            exact_match_results = qa_metrics_calculator.calculate_exact_match(valid_predictions, [ref[0] for ref in valid_references_final])
            f1_results = qa_metrics_calculator.calculate_f1_score(valid_predictions, valid_references_final)
            
            current_k_summary.update({
                "exact_match": exact_match_results,
                # CAMBIO CLAVE 2: Acceder directamente a los valores flotantes de ROUGE
                "rouge1_fmeasure": float(rouge_results.get("rouge1", 0.0)),
                "rouge2_fmeasure": float(rouge_results.get("rouge2", 0.0)),
                "rougeL_fmeasure": float(rouge_results.get("rougeL", 0.0)),
                "f1_score": float(f1_results.get("f1", 0.0))
            })
            print(f"  Métricas para k={k_value}: EM={exact_match_results:.4f}, ROUGE-L={float(rouge_results.get('rougeL', 0.0)):.4f}, F1={float(f1_results.get('f1', 0.0)):.4f}")
        else:
            print(f"  Advertencia: No hay suficientes datos válidos para calcular métricas de exactitud para k={k_value}. Asegúrate de que las respuestas en qa_test_set.json no estén vacías y que las invocaciones no fallen.")
            current_k_summary.update({
                "exact_match": 0.0, "rouge1_fmeasure": 0.0, "rouge2_fmeasure": 0.0, "rougeL_fmeasure": 0.0, "f1_score": 0.0
            })

        # Calcular percentiles de tiempo y memoria para el k actual
        current_k_response_times_arr = np.array(current_k_response_times)
        current_k_memory_usages_arr = np.array(current_k_memory_usages)
        current_k_gpu_memory_usages_arr = np.array(current_k_gpu_memory_usages)

        time_percentiles = calculate_percentiles(current_k_response_times_arr.tolist())
        mem_percentiles = calculate_percentiles(current_k_memory_usages_arr.tolist())
        gpu_mem_percentiles = calculate_percentiles(current_k_gpu_memory_usages_arr.tolist())

        current_k_summary["avg_response_time_ms"] = np.mean(current_k_response_times_arr) if current_k_response_times_arr.size > 0 else 0.0
        current_k_summary["p50_response_time_ms"] = time_percentiles.get("P50", 0.0)
        current_k_summary["p95_response_time_ms"] = time_percentiles.get("P95", 0.0)
        current_k_summary["p99_response_time_ms"] = time_percentiles.get("P99", 0.0)
        
        current_k_summary["avg_memory_mb"] = np.mean(current_k_memory_usages_arr) if current_k_memory_usages_arr.size > 0 else 0.0
        current_k_summary["p50_memory_mb"] = mem_percentiles.get("P50", 0.0)
        current_k_summary["p95_memory_mb"] = mem_percentiles.get("P95", 0.0)
        current_k_summary["p99_memory_mb"] = mem_percentiles.get("P99", 0.0)

        current_k_summary["avg_gpu_memory_mb"] = np.mean(current_k_gpu_memory_usages_arr) if current_k_gpu_memory_usages_arr.size > 0 else 0.0
        current_k_summary["p50_gpu_memory_mb"] = gpu_mem_percentiles.get("P50", 0.0)
        current_k_summary["p95_gpu_memory_mb"] = gpu_mem_percentiles.get("P95", 0.0)
        current_k_summary["p99_gpu_memory_mb"] = gpu_mem_percentiles.get("P99", 0.0)

        all_k_results.append(current_k_summary)

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resumen de resultados por K (exactitud, promedios y percentiles de tiempo/memoria)
    results_summary_df = pd.DataFrame(all_k_results)
    summary_file = os.path.join(RESULTS_DIR, f"benchmark_summary_{timestamp}.csv")
    results_summary_df.to_csv(summary_file, index=False)
    print(f"\nResumen de resultados de benchmark guardado en: {summary_file}")

    # Detalles de tiempo de respuesta por pregunta (para gráficos más granulares si se desea)
    detailed_response_times = []
    for k, times in all_response_times_by_k.items():
        for t in times:
            detailed_response_times.append({"k": k, "time_ms": t})
    response_time_df = pd.DataFrame(detailed_response_times)
    response_time_file = os.path.join(RESULTS_DIR, f"response_time_details_{timestamp}.csv")
    response_time_df.to_csv(response_time_file, index=False)
    print(f"Detalles de tiempo de respuesta guardados en: {response_time_file}")

    detailed_memory_usages = []
    for k, mems in all_memory_usages_by_k.items():
        for i, m in enumerate(mems):
            detailed_memory_usages.append({
                "k": k, 
                "memory_mb": m, 
                "gpu_memory_mb": all_gpu_memory_usages_by_k[k][i] 
            })
    memory_usage_df = pd.DataFrame(detailed_memory_usages)
    memory_usage_file = os.path.join(RESULTS_DIR, f"memory_usage_details_{timestamp}.csv")
    memory_usage_df.to_csv(memory_usage_file, index=False)
    print(f"Detalles de uso de memoria guardados en: {memory_usage_file}")

    print("\nBenchmark de rendimiento completado.")

if __name__ == "__main__":
    run_benchmark()