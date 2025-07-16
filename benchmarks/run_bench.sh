#!/bin/bash
# benchmarks/run_bench.sh
# Script para ejecutar los benchmarks del sistema RAG.

echo "Iniciando proceso de benchmarking..."

# Ruta a la raíz del proyecto. Asume que este script está en 'benchmarks/'
PROJECT_ROOT=$(dirname "$0")/..

# Activar el entorno virtual (asumiendo que se llama 'venv' o 'env' en la raíz del proyecto)
# Si tu entorno virtual tiene otro nombre o ubicación, ajústalo aquí.
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activando entorno virtual 'venv'..."
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$PROJECT_ROOT/env" ]; then
    echo "Activando entorno virtual 'env'..."
    source "$PROJECT_ROOT/env/bin/activate"
else
    echo "Advertencia: No se encontró un entorno virtual 'venv' o 'env' en la raíz del proyecto ($PROJECT_ROOT). Asegúrate de que las dependencias estén instaladas en tu entorno activo."
    # Opcional: podrías salir aquí si el entorno es mandatorio
    # exit 1 
fi

# Crear el directorio de resultados si no existe (ruta relativa a la ejecución actual)
mkdir -p results

# Ejecutar el script de benchmarking.
# Establecemos PYTHONPATH para que Python pueda encontrar los módulos en 'src'
# La ruta del script Python ahora es relativa a la ubicación actual del shell ('benchmarks/')
echo "Ejecutando bench_qa_performance.py..."
PYTHONPATH="$PROJECT_ROOT" python3 bench_qa_performance.py # <--- CAMBIO AQUÍ (sin 'benchmarks/' y con PYTHONPATH)

# Desactivar el entorno virtual
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Desactivando entorno virtual."
    deactivate
fi

echo "Proceso de benchmarking completado. Resultados guardados en benchmarks/results/."