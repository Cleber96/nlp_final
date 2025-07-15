# Proyecto 8: RAG con LangChain + VectorStore
## Descripción del Proyecto

Este repositorio contiene la implementación de un sistema de Preguntas-Respuestas (Question-Answering) basado en la arquitectura **Retrieval Augmented Generation (RAG)**. El objetivo principal es construir una solución robusta y modular para interactuar con documentación técnica diversa (PDF, Markdown), utilizando LangChain para orquestar los componentes, FAISS como almacén vectorial eficiente y un Modelo de Lenguaje Grande (LLM) para la generación de respuestas contextualizadas.

El sistema está diseñado para superar las limitaciones de los LLMs puros, como las "alucinaciones" o la falta de conocimiento específico sobre datos privados o muy recientes, al "aumentar" la capacidad de generación del LLM con información relevante recuperada de una base de conocimiento especializada.

## Estructura del Repositorio

```plaintext
# Proyecto 8: Sistema RAG para Preguntas-Respuestas sobre Documentación Técnica

## Descripción del Proyecto

Este repositorio contiene la implementación de un sistema de Preguntas-Respuestas (Question-Answering) basado en la arquitectura **Retrieval Augmented Generation (RAG)**. El objetivo principal es construir una solución robusta y modular para interactuar con documentación técnica diversa (PDF, Markdown), utilizando LangChain para orquestar los componentes, FAISS como almacén vectorial eficiente y un Modelo de Lenguaje Grande (LLM) para la generación de respuestas contextualizadas.

El sistema está diseñado para superar las limitaciones de los LLMs puros, como las "alucinaciones" o la falta de conocimiento específico sobre datos privados o muy recientes, al "aumentar" la capacidad de generación del LLM con información relevante recuperada de una base de conocimiento especializada.

## Estructura del Repositorio

```plaintext
proyecto/
├── src/                      <- Implementación principal de los módulos del sistema RAG
│   ├── __init__.py
│   ├── data_ingestion/       <- Módulos para carga y preprocesamiento de documentos
│   │   ├── document_loader.py
│   │   └── text_splitter.py
│   ├── vector_store/         <- Módulos para generación de embeddings y gestión del almacén vectorial
│   │   ├── embeddings_generator.py
│   │   └── faiss_store.py
│   ├── rag_system/           <- Módulos que orquestan la cadena RAG
│   │   ├── prompt_template.py
│   │   ├── llm_model.py
│   │   └── retriever.py
│   ├── evaluation/           <- Módulos para métricas de evaluación y benchmarking
│   │   └── qa_metrics.py
│   └── utils.py              <- Funciones auxiliares (ej. logging)
│
├── tests/                    <- Pruebas unitarias y funcionales (pytest)
│   ├── test_embeddings_generator.py
│   ├── test_document_loader.py
│   ├── test_text_splitter.py
│   ├── test_faiss_store.py
│   └── test_retriever.py
│   └── test_llm_model.py # Si hay tests específicos para el LLM
│
├── benchmarks/               <- Scripts y resultados de benchmarking
│   ├── run_bench.sh          <- Script bash para ejecutar benchmarks
│   ├── benchmark_accuracy_latency.py <- Código para medir tiempo de respuesta y exactitud
│   └── results/              <- Carpeta para almacenar los resultados de los benchmarks
│       └── benchmark_results.csv
│
├── models/                   <- Carpeta opcional para modelos LLM descargados localmente (e.g., .gguf)
│   └── modelo.gguf 
│
├── data/                     <- Carpeta opcional para documentos de ejemplo
│   └── prueba.md
│
├── exposicion.ipynb          <- Cuaderno único con explicación, demos y visualizaciones
├── requirements.txt          <- Dependencias exactas del proyecto
├── README.md                 <- Este archivo
└── .gitignore                <- Archivos y directorios a ignorar por Git
````

## Requisitos de Hardware

  * **CPU:** Procesador multinúcleo moderno (Intel i5/Ryzen 5 o superior recomendado).
  * **RAM:** Mínimo 8 GB (16 GB o más recomendado para el procesamiento de documentos y modelos LLM ligeros).
  * **GPU (Opcional pero recomendado para modelos ligeramente más grandes):** Una GPU NVIDIA con al menos 4 GB de VRAM puede acelerar la inferencia del LLM si se utiliza cuantificación (requiere `bitsandbytes` y `accelerate`). El proyecto está configurado para funcionar en CPU por defecto si no se detecta una GPU compatible.
  * **Espacio en Disco:** Al menos 10 GB de espacio libre para dependencias, cachés de modelos de Hugging Face y el índice FAISS.

## Cómo Ejecutar el Proyecto

Siga los siguientes pasos para configurar y ejecutar el proyecto:

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/Cleber96/nlp_final.git
    cd nlp_final
    ```

2.  **Configurar el entorno virtual e instalar dependencias:**
    Se recomienda usar un entorno virtual para gestionar las dependencias.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate   # En Windows

    pip install -r requirements.txt
    ```

    *Si tienes GPU y deseas utilizar la cuantificación de 4 bits para modelos más grandes (ej. Phi-2) en `src/rag_system/llm_model.py`, asegúrate de instalar `bitsandbytes` y `accelerate`:*

    ```bash
    pip install bitsandbytes accelerate
    ```

3.  **Preparar los documentos (opcional, si no usas los de ejemplo):**
    Coloca tus documentos `.pdf` o `.md` en la carpeta `data/`. El cuaderno `exposicion.ipynb` espera encontrar documentos en esta ruta por defecto para la ingesta.

4.  **Generar el índice FAISS:**
    El sistema requiere un índice FAISS previamente generado para la recuperación. Puedes generarlo ejecutando las celdas iniciales del cuaderno `exposicion.ipynb`. Este proceso cargará los documentos, los dividirá, generará embeddings y guardará el índice FAISS.

5.  **Ejecutar pruebas unitarias y funcionales:**
    Se han implementado pruebas exhaustivas para asegurar la robustez de los componentes.

    ```bash
    PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing
    ```

    *La cobertura mínima esperada es del 70%. Este comando también mostrará un informe detallado de cobertura.*

6.  **Correr benchmarks:**
    Los scripts de benchmarking están en la carpeta `benchmarks/`. Estos scripts miden la exactitud del sistema RAG frente al tamaño del contexto (`k`) y el tiempo de respuesta.

    ```bash
    bash benchmarks/run_bench.sh
    ```

    *Los resultados se guardarán en `benchmarks/results/benchmark_results.csv` y se visualizarán en el `exposicion.ipynb`.*

7.  **Ejecutar el modelo compleot**

    ```bash
    python3 demo_script_.py
    ```