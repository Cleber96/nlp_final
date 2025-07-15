import os
import json
import pandas as pd
import time
import numpy as np
import warnings
from datetime import datetime
import psutil # Para medir uso de RAM
# from pynvml.pynvml import * # Para medir uso de VRAM (requiere NVIDIA GPU)

# Ignorar advertencias de LangChain y otros módulos si son muy ruidosas durante el benchmark
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Importar componentes de nuestro sistema RAG
from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.text_splitter import TextSplitter
from src.vector_store.embeddings_generator import EmbeddingsGenerator
from src.vector_store.faiss_store import FAISSStore
from src.rag_system.prompt_template import PromptTemplateManager
from src.rag_system.llm_model import LLMModel # Importar LLMModel
from src.rag_system.retriever import CustomRetriever
from src.rag_system.rag_chain import RAGChain
from src.evaluation.qa_metrics import QAMetrics
from src.evaluation.timing_utils import measure_time, calculate_percentiles
from src.evaluation.memory_utils import get_memory_usage_mb, get_gpu_memory_usage_mb

# --- Configuración de rutas y parámetros ---
DOCS_PATH = "docs/" # Carpeta con documentos PDF/Markdown para indexar
FAISS_INDEX_PATH = "faiss_index" # Donde se guardará/cargará el índice FAISS
QA_DATASET_PATH = "data/qa_test_set.json" # Dataset de preguntas y respuestas para evaluación
RESULTS_DIR = "benchmarks/results/"

# --- CONFIGURACIÓN DEL MODELO LLM (GPT-NeoX) ---
# Nombre del modelo GPT-NeoX en Hugging Face Hub
LLM_MODEL_NAME = "EleutherAI/gpt-neox-20b"
# Ruta LOCAL donde se descargará/cargará el modelo GPT-NeoX.
# Asegúrate de que esta carpeta exista y tenga espacio suficiente (aprox. 40GB para FP16).
# Ejemplo: si el modelo se descarga en `proyecto8_rag/models/EleutherAI/gpt-neox-20b`
LLM_MODEL_LOCAL_PATH = "models/EleutherAI/gpt-neox-20b"
# Dispositivo a usar para la inferencia del LLM: "auto" (GPU si disponible), "cuda" (forzar GPU), "cpu" (forzar CPU)
# "auto" es la mejor opción para que transformers decida.
LLM_DEVICE = "auto"

# Rangos de K para evaluar el retriever
K_VALUES = [1, 3, 5, 7, 10]

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

    if load_faiss_if_exists and os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_store.load_local(FAISS_INDEX_PATH)
            print(f"Índice FAISS cargado desde {FAISS_INDEX_PATH}")
        except FileNotFoundError:
            print(f"Índice FAISS no encontrado en {FAISS_INDEX_PATH}, procediendo a crearlo.")
            # Fallback a creación si hay error inesperado al cargar
            load_faiss_if_exists = False # Para forzar la creación
    
    if not load_faiss_if_exists or not os.path.exists(FAISS_INDEX_PATH):
        print(f"Creando índice FAISS desde documentos en {DOCS_PATH}...")
        document_loader = DocumentLoader()
        text_splitter = TextSplitter()
        
        all_documents = []
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    all_documents.extend(document_loader.load_pdf(file_path))
                elif file.endswith(".md"):
                    all_documents.extend(document_loader.load_markdown(file_path))
        
        if not all_documents:
            raise ValueError(f"No se encontraron documentos en {DOCS_PATH}. Asegúrate de tener archivos PDF o Markdown.")

        chunks = text_splitter.split_documents(all_documents)
        print(f"Documentos cargados y divididos en {len(chunks)} fragmentos.")
        
        faiss_store.create_from_documents(chunks, FAISS_INDEX_PATH)
        print("Índice FAISS creado y guardado.")

    # 2. Retriever
    retriever = CustomRetriever(faiss_path=FAISS_INDEX_PATH, embeddings=embeddings, k=k_value)
    langchain_retriever = retriever.get_langchain_retriever()
    print(f"Retriever configurado con k={k_value}")

    # 3. LLM (GPT-NeoX)
    # Se crea el LLMModel usando el nombre y la ruta local para GPT-NeoX
    # El LLMModel se encargará de cargar el modelo usando transformers y HuggingFacePipeline
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
        raise FileNotFoundError(f"Dataset QA no encontrado en: {filepath}")
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
            rag_system = setup_rag_system(k_value=k_value, load_faiss_if_exists=True)
        except Exception as e:
            print(f"Error al configurar el sistema RAG para k={k_value}: {e}. Saltando este valor de k.")
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

            # Medir tiempo y memoria antes de la invocación
            process_info_before = psutil.Process(os.getpid())
            mem_before_mb = get_memory_usage_mb(process_info_before)
            gpu_mem_before_mb = get_gpu_memory_usage_mb()

            # Usar la función measure_time de timing_utils.py
            try:
                generated_answer, elapsed_time_sec = measure_time(rag_system.invoke, question)
            except Exception as e:
                print(f"Error al invocar RAGChain para pregunta '{question[:50]}...': {e}")
                generated_answer = "ERROR_GENERATION" # Marcar como error
                elapsed_time_sec = 0.0 # O np.nan si prefieres
            
            response_time_ms = elapsed_time_sec * 1000 # Convertir a milisegundos

            # Medir uso de memoria después de la invocación
            process_info_after = psutil.Process(os.getpid())
            # Tomamos el máximo para capturar el pico de uso durante la operación
            peak_memory_mb = max(mem_before_mb, get_memory_usage_mb(process_info_after))
            peak_gpu_memory_mb = max(gpu_mem_before_mb, get_gpu_memory_usage_mb()) # Si no hay GPU, esta retornará 0.0

            predictions.append(generated_answer)
            references.append(reference_context)

            current_k_response_times.append(response_time_ms)
            current_k_memory_usages.append(peak_memory_mb)
            current_k_gpu_memory_usages.append(peak_gpu_memory_mb)

            print(f"  Q: {question[:70]}...")
            print(f"  A (Generada): {generated_answer[:70]}...")
            print(f"  Tiempo: {response_time_ms:.2f} ms, Memoria: {peak_memory_mb:.2f} MB, GPU Mem: {peak_gpu_memory_mb:.2f} MB")

        # Almacenar los tiempos y usos de memoria para este k
        all_response_times_by_k[k_value] = current_k_response_times
        all_memory_usages_by_k[k_value] = current_k_memory_usages
        all_gpu_memory_usages_by_k[k_value] = current_k_gpu_memory_usages

        # Calcular métricas de exactitud para el lote actual de k
        current_k_summary = {"k": k_value}
        
        # Filtrar predicciones y referencias para asegurar que no haya errores de generación
        valid_predictions = [p for p in predictions if p != "ERROR_GENERATION"]
        valid_references = [r for r, p in zip(references, predictions) if p != "ERROR_GENERATION"]

        if valid_predictions and all(ref for ref_list in valid_references for ref in ref_list):
            rouge_results = qa_metrics_calculator.calculate_rouge(valid_predictions, valid_references)
            # Para Exact Match, se suele comparar con una sola referencia (la principal)
            exact_match_results = qa_metrics_calculator.calculate_exact_match(valid_predictions, [ref[0] for ref in valid_references])
            f1_results = qa_metrics_calculator.calculate_f1_score(valid_predictions, valid_references)
            
            current_k_summary.update({
                "exact_match": exact_match_results,
                "rouge1_fmeasure": rouge_results.get("rouge1", {}).get("fmeasure", 0.0),
                "rouge2_fmeasure": rouge_results.get("rouge2", {}).get("fmeasure", 0.0),
                "rougeL_fmeasure": rouge_results.get("rougeL", {}).get("fmeasure", 0.0),
                "f1_score": f1_results.get("f1", 0.0)
            })
            print(f"  Métricas para k={k_value}: EM={exact_match_results:.4f}, ROUGE-L={rouge_results.get('rougeL', {}).get('fmeasure', 0.0):.4f}, F1={f1_results.get('f1', 0.0):.4f}")
        else:
            print(f"  Advertencia: No hay suficientes datos válidos para calcular métricas de exactitud para k={k_value}.")
            current_k_summary.update({
                "exact_match": 0.0, "rouge1_fmeasure": 0.0, "rouge2_fmeasure": 0.0, "rougeL_fmeasure": 0.0, "f1_score": 0.0
            })

        # Calcular percentiles de tiempo y memoria para el k actual
        # Asegurarse de que las listas no estén vacías para evitar errores de np.mean o calculate_percentiles
        current_k_response_times_arr = np.array(current_k_response_times)
        current_k_memory_usages_arr = np.array(current_k_memory_usages)
        current_k_gpu_memory_usages_arr = np.array(current_k_gpu_memory_usages)

        time_percentiles = calculate_percentiles(current_k_response_times)
        mem_percentiles = calculate_percentiles(current_k_memory_usages)
        gpu_mem_percentiles = calculate_percentiles(current_k_gpu_memory_usages)

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
                "gpu_memory_mb": all_gpu_memory_usages_by_k[k][i] if k in all_gpu_memory_usages_by_k and len(all_gpu_memory_usages_by_k[k]) > i else 0.0
            })
    memory_usage_df = pd.DataFrame(detailed_memory_usages)
    memory_usage_file = os.path.join(RESULTS_DIR, f"memory_usage_details_{timestamp}.csv")
    memory_usage_df.to_csv(memory_usage_file, index=False)
    print(f"Detalles de uso de memoria guardados en: {memory_usage_file}")

    print("\nBenchmark de rendimiento completado.")
    print("Recuerda visualizar los resultados en el cuaderno exposicion.ipynb.")

if __name__ == "__main__":
    run_benchmark()