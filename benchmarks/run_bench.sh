#!/bin/bash

# run_bench.sh
# Script para ejecutar los benchmarks del sistema RAG.

echo "Iniciando proceso de benchmarking..."

# Activar el entorno virtual (asumiendo que se llama 'venv' o 'env')
# Si tu entorno virtual tiene otro nombre o ubicación, ajústalo aquí.
if [ -d "venv" ]; then
    echo "Activando entorno virtual 'venv'..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "Activando entorno virtual 'env'..."
    source env/bin/activate
else
    echo "Advertencia: No se encontró un entorno virtual 'venv' o 'env'. Asegúrate de que las dependencias estén instaladas."
fi

# Crear el directorio de resultados si no existe
mkdir -p benchmarks/results

# Ejecutar el script de benchmarking
echo "Ejecutando bench_qa_performance.py..."
python benchmarks/bench_qa_performance.py

# Desactivar el entorno virtual
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Desactivando entorno virtual."
    deactivate
fi

echo "Proceso de benchmarking completado. Resultados guardados en benchmarks/results/."