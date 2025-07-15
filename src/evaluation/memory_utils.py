import psutil # Requiere 'pip install psutil'
import os
import logging
import platform
from typing import Dict, Any, Optional

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Importar para VRAM si PyTorch y CUDA están disponibles
try:
    import torch
    if torch.cuda.is_available():
        from pynvml import * # Requiere 'pip install pynvml'
        _NVML_AVAILABLE = True
        nvmlInit() # Inicializa la biblioteca NVML
        logger.info("pynvml (NVML) cargado para monitoreo de VRAM.")
    else:
        _NVML_AVAILABLE = False
        logger.info("CUDA no está disponible, el monitoreo de VRAM con pynvml no se utilizará.")
except (ImportError, NVMLError) as e:
    _NVML_AVAILABLE = False
    logger.warning(f"pynvml no está disponible o falló al inicializar ({e}). El monitoreo de VRAM no funcionará.")
except Exception as e:
    _NVML_AVAILABLE = False
    logger.warning(f"Error inesperado al importar o inicializar pynvml: {e}. El monitoreo de VRAM no funcionará.")


def get_current_process_memory_usage() -> Dict[str, float]:
    """
    Obtiene el uso de memoria RAM del proceso actual.

    Returns:
        Dict[str, float]: Un diccionario con el uso de memoria en MB.
                          'rss': Resident Set Size (memoria física usada).
                          'vms': Virtual Memory Size (memoria virtual total).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    rss_mb = mem_info.rss / (1024 * 1024) # en MB
    vms_mb = mem_info.vms / (1024 * 1024) # en MB
    
    logger.debug(f"Uso de RAM actual - RSS: {rss_mb:.2f} MB, VMS: {vms_mb:.2f} MB")
    return {"rss_mb": rss_mb, "vms_mb": vms_mb}

def get_system_memory_info() -> Dict[str, float]:
    """
    Obtiene información general sobre el uso de memoria RAM del sistema.

    Returns:
        Dict[str, float]: Un diccionario con información de memoria del sistema en MB.
                          'total_mb': Memoria RAM total.
                          'available_mb': Memoria RAM disponible.
                          'used_mb': Memoria RAM usada.
                          'percent_used': Porcentaje de RAM usada.
    """
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    available_mb = mem.available / (1024 * 1024)
    used_mb = mem.used / (1024 * 1024)
    percent_used = mem.percent
    
    logger.debug(f"Memoria del sistema - Total: {total_mb:.2f} MB, Disponible: {available_mb:.2f} MB, Usada: {used_mb:.2f} MB ({percent_used:.1f}%)")
    return {
        "total_mb": total_mb,
        "available_mb": available_mb,
        "used_mb": used_mb,
        "percent_used": percent_used
    }

def get_gpu_memory_usage() -> Dict[str, Union[float, str]]:
    """
    Obtiene el uso de memoria VRAM de las GPUs NVIDIA.
    Requiere que pynvml esté instalado y que haya GPUs NVIDIA disponibles.

    Returns:
        Dict[str, Union[float, str]]: Un diccionario con el uso de VRAM en MB por GPU.
                                      Si pynvml no está disponible, retorna un mensaje.
    """
    if not _NVML_AVAILABLE:
        logger.warning("NVML (pynvml) no está disponible o no se pudo inicializar. No se puede obtener el uso de VRAM.")
        return {"error": "pynvml no está disponible o no se pudo inicializar."}

    gpu_mem_info = {}
    try:
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            
            total_vram_mb = mem_info.total / (1024 * 1024)
            used_vram_mb = mem_info.used / (1024 * 1024)
            free_vram_mb = mem_info.free / (1024 * 1024)
            
            gpu_mem_info[f"gpu_{i}"] = {
                "total_vram_mb": total_vram_mb,
                "used_vram_mb": used_vram_mb,
                "free_vram_mb": free_vram_mb,
                "percent_used": (used_vram_mb / total_vram_mb) * 100 if total_vram_mb > 0 else 0.0
            }
            logger.debug(f"GPU {i} VRAM - Usada: {used_vram_mb:.2f} MB / Total: {total_vram_mb:.2f} MB")
    except NVMLError as e:
        logger.error(f"Error NVML al obtener el uso de VRAM: {e}")
        gpu_mem_info["error"] = f"Error NVML: {e}"
    except Exception as e:
        logger.error(f"Error inesperado al obtener el uso de VRAM: {e}")
        gpu_mem_info["error"] = f"Error inesperado: {e}"
    finally:
        # No se debe llamar nvmlShutdown() aquí si se espera seguir monitoreando.
        # Es mejor llamarlo una vez al final del script principal o benchmark.
        pass
        
    return gpu_mem_info

if __name__ == "__main__":
    print("--- Probando memory_utils ---")

    # Prueba de uso de memoria del proceso
    print("\n--- Uso de memoria del proceso actual (RAM) ---")
    current_mem_usage = get_current_process_memory_usage()
    print(f"Memoria RSS (física): {current_mem_usage.get('rss_mb', 0):.2f} MB")
    print(f"Memoria VMS (virtual): {current_mem_usage.get('vms_mb', 0):.2f} MB")

    # Prueba de información de memoria del sistema
    print("\n--- Información de memoria del sistema (RAM) ---")
    system_mem_info = get_system_memory_info()
    print(f"RAM Total: {system_mem_info.get('total_mb', 0):.2f} MB")
    print(f"RAM Disponible: {system_mem_info.get('available_mb', 0):.2f} MB")
    print(f"RAM Usada: {system_mem_info.get('used_mb', 0):.2f} MB ({system_mem_info.get('percent_used', 0):.1f}%)")

    # Prueba de uso de memoria GPU (si aplica)
    print("\n--- Uso de memoria GPU (VRAM) ---")
    gpu_mem_usage = get_gpu_memory_usage()
    if "error" in gpu_mem_usage:
        print(f"No se pudo obtener el uso de VRAM: {gpu_mem_usage['error']}")
        print("Asegúrate de tener GPUs NVIDIA y que `pynvml` esté instalado y funcionando.")
    else:
        for gpu_id, info in gpu_mem_usage.items():
            print(f"  {gpu_id.upper()}:")
            print(f"    VRAM Usada: {info.get('used_vram_mb', 0):.2f} MB")
            print(f"    VRAM Total: {info.get('total_vram_mb', 0):.2f} MB")
            print(f"    VRAM % Usada: {info.get('percent_used', 0):.1f}%")

    # Ejemplo de cómo se mediría el pico de memoria en una función (simplificado)
    print("\n--- Ejemplo de medición de pico de memoria (conceptual) ---")
    # En un escenario real, se tomarían instantáneas de memoria antes y después
    # de una operación o se usaría una herramienta de profiling.
    # psutil.Process().memory_info().rss o .uss (si es en Linux) son útiles para picos.

    def simulate_heavy_task(data_size_mb: int):
        logger.info(f"Simulando tarea pesada que consume {data_size_mb} MB...")
        # Simular consumo de memoria
        _ = [0] * (data_size_mb * 1024 * 1024 // 8) # Crear una lista de ints
        logger.info("Tarea simulada completada.")
        del _ # Liberar memoria (Python GC puede ser perezoso)

    initial_mem = get_current_process_memory_usage().get('rss_mb', 0)
    print(f"Memoria RSS inicial: {initial_mem:.2f} MB")
    
    # Este es un ejemplo simple, para un pico real se necesitaría un enfoque más avanzado
    # como un subproceso que monitoree o un profiler de memoria.
    simulate_heavy_task(100) # Simula consumir 100 MB
    
    final_mem = get_current_process_memory_usage().get('rss_mb', 0)
    print(f"Memoria RSS final: {final_mem:.2f} MB (cambio: {final_mem - initial_mem:.2f} MB)")

    # Es importante llamar a nvmlShutdown() si se inicializó nvmlInit() y ya no se usará.
    if _NVML_AVAILABLE:
        try:
            nvmlShutdown()
            logger.info("NVML desinicializado.")
        except NVMLError as e:
            logger.error(f"Error al desinicializar NVML: {e}")