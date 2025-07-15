import os
import logging
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Opcional: para usar BitsAndBytesConfig para cuantificación 4-bit/8-bit
from transformers import BitsAndBytesConfig # Asegúrate de que bitsandbytes esté instalado si usas GPU

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class LLMModel:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M", model_path: str = None, device: str = "auto"):
        """
        Inicializa el modelo LLM.
        Se configura para usar un modelo de Hugging Face (como GPT-Neo o Phi-2) a través de HuggingFacePipeline.

        Args:
            model_name (str): El nombre del modelo en Hugging Face Hub.
                              Por defecto, "EleutherAI/gpt-neo-125M" para una carga rápida.
            model_path (str, optional): Ruta local específica donde el modelo ya está descargado o se guardará.
                                        Debe apuntar a la carpeta del modelo descargado.
                                        Por defecto, None.
            device (str): Dispositivo a usar para la inferencia ("auto", "cpu", "cuda").
                          "auto" intentará usar GPU si está disponible.
                          Para una demo ligera en CPU, puedes forzar "cpu".
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.llm = None
        self._load_llm()

    def _load_llm(self):
        logger.info(f"Cargando modelo LLM: {self.model_name}")
        
        # Decide la fuente del modelo: si model_path existe, úsalo, de lo contrario usa model_name del Hub
        model_source_for_loading = self.model_path if self.model_path and os.path.exists(self.model_path) else self.model_name
        
        if not os.path.exists(model_source_for_loading) and model_source_for_loading == self.model_path:
            logger.warning(f"La ruta local del modelo '{self.model_path}' no existe. Intentando descargar desde Hugging Face Hub con el nombre '{self.model_name}'.")
            # Si el model_path no existe, forzamos la descarga por el nombre del modelo
            model_source_for_loading = self.model_name 
            
            # Crea la carpeta si no existe, para la descarga si es necesario.
            # Transformers gestionará su propia caché, pero si queremos descargar a una ruta específica,
            # podríamos intentar usar el método `snapshot_download` de `huggingface_hub` aquí
            # para asegurarnos de que el modelo esté en LLM_MODEL_LOCAL_PATH.
            # Para este ejemplo, dejaremos que transformers maneje su caché por defecto,
            # o se asume que el usuario lo ha descargado previamente a LLM_MODEL_LOCAL_PATH.
            if self.model_path and not os.path.exists(self.model_path):
                os.makedirs(self.model_path, exist_ok=True)


        try:
            # Cargar el tokenizador
            tokenizer = AutoTokenizer.from_pretrained(model_source_for_loading)

            # Ajustes para eficiencia y uso de memoria:
            model_loading_kwargs = {
                # Usa bfloat16 si la GPU lo soporta (más reciente), sino float16 para menor VRAM.
                # Si estás en CPU, esto no tiene un impacto significativo, ya que torch.float32 será el predeterminado.
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                "device_map": self.device, # "auto" para que transformers gestione el volcado a CPU si no cabe en GPU.
            }

            # Si quieres usar cuantificación (solo si hay GPU y bitsandbytes instalado), descomenta estas líneas.
            # Para 'EleutherAI/gpt-neo-125M', la cuantificación no es estrictamente necesaria por su tamaño,
            # pero la lógica sigue siendo válida.
            if torch.cuda.is_available() and self.device != "cpu":
                logger.info("GPU disponible y device no es 'cpu'. Intentando cargar con cuantificación de 4 bits para ahorrar VRAM.")
                # Asegúrate de importar BitsAndBytesConfig desde transformers
                # from transformers import BitsAndBytesConfig 
                
                model_loading_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.float16, 
                    bnb_4bit_use_double_quant=True,
                )
                # Forzar device_map a "auto" si se usa cuantificación para que bitsandbytes lo gestione
                model_loading_kwargs["device_map"] = "auto" 
            

            model = AutoModelForCausalLM.from_pretrained(
                model_source_for_loading,
                **model_loading_kwargs
            )
            
            # Crear un pipeline de generación de texto
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256, # Reducido para modelos más pequeños, puedes ajustar
                temperature=0.1,    # Baja temperatura para respuestas más concisas y fácticas en RAG
                pad_token_id=tokenizer.eos_token_id # Para evitar warnings con padding en algunos modelos
            )

            # Envolver el pipeline en una instancia de HuggingFacePipeline de LangChain
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Modelo {self.model_name} cargado y configurado con éxito.")

        except Exception as e:
            logger.error(f"Fallo al cargar el modelo LLM '{self.model_name}' desde '{model_source_for_loading}': {e}")
            raise

    def get_llm(self):
        return self.llm