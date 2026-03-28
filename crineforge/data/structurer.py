import os
import torch
import gc
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ..utils.diffcheck import validate_token_length
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Singleton Instance
_structurer_instance = None

class Structurer:
    """
    Uses local DeepSeek Chat model to format raw data into Instruct-Response JSON pairs.
    Loaded lazily and cached via Singleton to prevent OOM.
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.max_new_tokens = 2048
        self.temperature = 0.2
        # Removed premature _load_model() execution to enforce explicit laziness
        
    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
            
        logger.info(f"[Structurer] Downloading/Loading model: {self.model_id} (this may take time on first run)")
        
        cache_dir = os.path.expanduser("~/.crineforge/models")
        os.makedirs(cache_dir, exist_ok=True)
        
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info("[HF] Authenticated model access enabled for structurer")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir, token=hf_token)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                cache_dir=cache_dir,
                device_map="auto",
                quantization_config=bnb_config,
                token=hf_token
            )
            logger.info("[Structurer] Model loaded successfully in 4-bit precision.")
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as e:
            logger.warning(f"[Structurer] BitsAndBytes/Triton initialization failed ({type(e).__name__}): {str(e)}")
            logger.info("[Structurer] Cascading Fallback -> Attempting native torch.float16 on GPU.")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    cache_dir=cache_dir,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    token=hf_token
                )
                logger.info("[Structurer] Model loaded successfully via float16 fallback.")
            except torch.cuda.OutOfMemoryError as oom:
                logger.error(f"[Structurer] CUDA OOM Error during float16 fallback: {str(oom)}")
                logger.warning("[Structurer] Last Resort Fallback -> Attempting CPU float32 (Very Slow or OOM).")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, 
                        cache_dir=cache_dir,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        token=hf_token
                    )
                    logger.info("[Structurer] Model loaded successfully via CPU ultimate fallback.")
                except Exception as e2:
                    self._handle_fallback_error(e2)
            except Exception as e2:
                self._handle_fallback_error(e2)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[Structurer] CUDA OOM Error on 4-bit load: {str(e)}")
            logger.warning("[Structurer] Falling back to CPU / float32. This will be very slow or OOM.")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    cache_dir=cache_dir,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    token=hf_token
                )
            except Exception as e2:
                self._handle_fallback_error(e2)

    def _handle_fallback_error(self, e2):
        if "401" in str(e2) or "unauthorized" in str(e2).lower():
            logger.error(f"[HF] Unauthorized to access '{self.model_id}'. Ensure HF_TOKEN is correctly set in environment if this is a gated model, and you have accepted the model license.")
        else:
            logger.error(f"[Structurer] Fallback loading failed completely: {str(e2)}.")
        raise e2

    def generate_pairs(self, text_chunk: str) -> str:
        """Converts raw text chunk into JSON format mapping {instruction: response} pairs."""
        if not self.model or not self.tokenizer:
            self._load_model()
            
        if not self.model or not self.tokenizer:
            raise RuntimeError("Structurer model not loaded.")
            
        base_prompt = (
            "You are a strict data formatter. Convert the following text into a JSON array of {instruction, response} pairs. "
            "DO NOT summarize. Retain all technical terms. ONLY output valid JSON. "
            f"\n\nText:\n{text_chunk}\n\nJSON:\n"
        )
        
        r_text = ""
        for attempt in range(2):
            prompt = base_prompt if attempt == 0 else base_prompt + "\n\nError: Previous output was not valid JSON. Ensure you ONLY output a valid JSON array starting with [ and ending with ]."
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if attempt == 0 else 0.01,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            r_text = response.strip()
            
            # 1. Enforce strict JSON validation
            try:
                json.loads(r_text)
                break
            except json.JSONDecodeError as e:
                logger.warning(f"[Structurer] Attempt {attempt+1} failed JSON validation.")
                if attempt == 1:
                    raise ValueError(f"Structurer failed to produce valid JSON after retries. Error: {str(e)}\nOutput was: {r_text}")
        
        # 2. Prevent silent data corruption (Validate output length >= 80% of input length)
        is_valid = validate_token_length(self.tokenizer, text_chunk, "\n".join(r_text.splitlines()), threshold=0.8)
        if not is_valid:
            logger.warning("[Structurer] Output length < 80% of input length. Potential data corruption or summarization detected.")
            
        return r_text

def get_structurer(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Structurer:
    """Returns the globally cached instance of the Structurer, instantiating it lazily if needed."""
    global _structurer_instance
    if _structurer_instance is None:
        _structurer_instance = Structurer(model_id=model_id)
    elif getattr(_structurer_instance, 'model_id', None) != model_id:
        free_structurer()
        _structurer_instance = Structurer(model_id=model_id)
    return _structurer_instance

def free_structurer():
    """Frees the structurer model from memory entirely and cleans the CUDA cache."""
    global _structurer_instance
    if _structurer_instance is not None:
        logger.info("[GPU] Freeing Structurer memory footprint...")
        if hasattr(_structurer_instance, 'model'):
            del _structurer_instance.model
        if hasattr(_structurer_instance, 'tokenizer'):
            del _structurer_instance.tokenizer
        del _structurer_instance
        _structurer_instance = None
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("[GPU] Structurer cache definitively cleared.")
