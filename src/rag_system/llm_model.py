# src/rag_system/llm_model.py
import os
import logging
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class LLMModel:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M", model_path: str = None, device: str = "auto"):
        self.model_name = model_name
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.llm = None
        self._load_llm()

    def _resolve_device(self, device_setting: str) -> str:
        if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()):
            if torch.cuda.is_available():
                logger.info("Device set to use cuda")
                return "cuda"
            else:
                logger.warning("CUDA no está disponible, se usará CPU.")
                return "cpu"
        else:
            logger.info("Device set to use cpu")
            return "cpu"

    def _load_llm(self):
        logger.info(f"Cargando modelo LLM: {self.model_name}")
        
        model_source_for_loading = self.model_path if self.model_path and os.path.exists(self.model_path) else self.model_name
        
        if not os.path.exists(model_source_for_loading) and model_source_for_loading == self.model_path:
            logger.warning(f"La ruta local del modelo '{self.model_path}' no existe. Intentando descargar desde Hugging Face Hub con el nombre '{self.model_name}'.")
            model_source_for_loading = self.model_name
            
            if self.model_path and not os.path.exists(self.model_path):
                os.makedirs(self.model_path, exist_ok=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_source_for_loading)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Se configuró pad_token = eos_token ({tokenizer.eos_token}) para el tokenizer.")
            
            model_loading_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            }

            if torch.cuda.is_available() and self.device != "cpu":
                logger.info("GPU disponible y device no es 'cpu'. Intentando cargar con cuantificación de 4 bits para ahorrar VRAM.")
                model_loading_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.float16, 
                    bnb_4bit_use_double_quant=True,
                )
                model_loading_kwargs["device_map"] = "auto" # Force auto when using quantization
            else:
                # When running on CPU, explicitly set device_map to "cpu"
                model_loading_kwargs["device_map"] = "cpu"
                model_loading_kwargs["torch_dtype"] = torch.float32 # Use float32 for CPU for better compatibility

            model = AutoModelForCausalLM.from_pretrained(
                model_source_for_loading,
                **model_loading_kwargs
            )
            
            # These are the *default* generation arguments for the pipeline
            # We will pass stop_sequences to HuggingFacePipeline directly
            pipe_kwargs = {
                "max_new_tokens": 512, 
                "temperature": 0.7,    
                "do_sample": True,
                "top_p": 0.95,         
                "repetition_penalty": 1.1, 
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                # No 'stop_sequences' here for now in the pipeline creation
                # Pass directly to HuggingFacePipeline if this resolves the error
            }

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # If device_map is "auto" or "cpu", the pipeline might infer it or handle it.
                # Do not pass 'device' here as device_map already handles it.
                **pipe_kwargs 
            )

            # Pass stop_sequences directly to HuggingFacePipeline
            # This is a common way to handle stop sequences with LangChain wrappers
            self.llm = HuggingFacePipeline(
                pipeline=pipe, 
                model_kwargs={"stop_sequences": ["Human:", "Pregunta:", "Contexto:", "Respuesta:"]}
            )
            logger.info(f"Modelo {self.model_name} cargado y configurado con éxito.")

        except Exception as e:
            logger.error(f"Fallo al cargar el modelo LLM '{self.model_name}' desde '{model_source_for_loading}': {e}", exc_info=True)
            raise

    def get_llm(self):
        return self.llm