proyecto8_rag/                         <- Raíz del repositorio
├── src/                               <- Implementación principal
│   ├── __init__.py
│   ├── data_ingestion/                <- Módulo para carga y preprocesamiento de datos
│   │   ├── document_loader.py         <- Carga de PDF/Markdown
│   │   ├── text_splitter.py           <- División de texto en chunks
│   │   └── data_processor.py          <- Orquestación de carga y división
│   ├── vector_store/                  <- Módulo para VectorStore y embeddings
│   │   ├── embeddings_generator.py    <- Generación de embeddings (ej. usando Hugging Face Embeddings)
│   │   ├── faiss_store.py             <- Interacción con FAISS (creación, carga, búsqueda)
│   │   └── vector_utils.py            <- Funciones auxiliares para el VectorStore
│   ├── rag_system/                    <- Módulo principal del sistema RAG
│   │   ├── prompt_template.py         <- Definición de los templates de prompt
│   │   ├── llm_model.py               <- Interacción con el LLM (GPT-NeoX)
│   │   ├── retriever.py               <- Clase para la recuperación de documentos (LangChain Retriever)
│   │   ├── rag_chain.py               <- Construcción de la cadena LangChain (Retriever + LLM + Prompt)
│   │   └── config.py                  <- Parámetros de configuración del sistema (rutas, modelos, etc.)
│   ├── evaluation/                    <- Módulo para métricas y evaluación
│   │   ├── qa_metrics.py              <- Métricas de exactitud (ej. ROUGE, F1, Exact Match)
│   │   ├── timing_utils.py            <- Utilidades para medir tiempo de respuesta
│   │   └── memory_utils.py            <- Utilidades para medir uso de memoria
│   └── main.py                        <- Script principal para ejecutar el sistema QA
│
├── tests/                             <- Pruebas (pytest)
│   ├── conftest.py
│   ├── test_document_loader.py
│   ├── test_text_splitter.py
│   ├── test_embeddings_generator.py
│   ├── test_faiss_store.py
│   ├── test_retriever.py
│   ├── test_rag_chain.py
│   └── test_qa_metrics.py
│
├── benchmarks/                        <- Scripts o notebooks de benchmarking
│   ├── run_bench.sh                   <- Script bash "todo en uno"
│   ├── bench_qa_performance.py        <- Script para medir exactitud, tiempo y memoria
│   ├── results/
│   │   ├── accuracy_vs_context.csv    <- Resultados de exactitud vs tamaño del contexto
│   │   ├── response_time.csv          <- Resultados de tiempo de respuesta
│   │   └── memory_usage.csv           <- Resultados de uso de memoria
│
├── docs/                              <- Carpeta para la documentación técnica (PDF/Markdown)
│   └── technical_document.pdf
│   └── another_doc.md
│
├── data/                              <- Carpeta para datos auxiliares (opcional)
│   └── qa_test_set.json               <- Dataset de preguntas y respuestas para evaluación
│
├── exposicion.ipynb                   <- Único cuaderno evaluable
├── requirements.txt                   <- Dependencias exactas (pip freeze)
├── README.md                          <- Instrucciones de ejecución y links al video
└── .gitignore