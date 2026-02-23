import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    
    def __init__(self, model_id: str = "deepseek-ai/deepseek-llm-7b-chat"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.max_new_tokens = 2048
        self.temperature = 0.2
        self._load_model()
        
    def _load_model(self):
        logger.info(f"Lazy loading structurer model: {self.model_id} (this may take time on first run)")
        
        cache_dir = os.path.expanduser("~/.crineforge/models")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                cache_dir=cache_dir,
                device_map="auto",
                load_in_4bit=True
            )
            logger.info("Structurer model loaded successfully in 4-bit precision.")
        except Exception as e:
            logger.error(f"Failed to load structurer model: {str(e)}")
            logger.warning("Falling back to CPU / float16. This will be very slow or OOM.")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    cache_dir=cache_dir,
                    device_map="cpu",
                    torch_dtype=torch.float16
                )
            except Exception as e2:
                logger.error(f"Fallback loading failed: {str(e2)}. Structuring will be unavailable.")

    def generate_pairs(self, text_chunk: str) -> str:
        """Converts raw text chunk into JSON format mapping {instruction: response} pairs."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Structurer model not loaded.")
            
        prompt = (
            "You are a strict data formatter. Convert the following text into JSON array of {instruction, response} pairs. "
            "DO NOT summarize. Retain all technical terms. ONLY output valid JSON. "
            f"\n\nText:\n{text_chunk}\n\nJSON:\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        r_text = response.strip()
        
        # Validate Length Preservation
        is_valid = validate_token_length(self.tokenizer, text_chunk, "\n".join(r_text.splitlines()), threshold=0.9)
        if not is_valid:
            logger.warning("Generation discarded due to length validation failure. Retrying with lower temperature...")
            # Fallback retry logic
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=0.01, # almost greedy
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            r_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
        return r_text

def get_structurer() -> Structurer:
    """Returns the globally cached instance of the Structurer, instantiating it lazily if needed."""
    global _structurer_instance
    if _structurer_instance is None:
        _structurer_instance = Structurer()
    return _structurer_instance

def free_structurer():
    """Frees the structurer model from memory when training phase begins."""
    global _structurer_instance
    if _structurer_instance is not None:
        logger.info("Freeing Structurer memory footprint...")
        del _structurer_instance.model
        del _structurer_instance.tokenizer
        _structurer_instance = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Structurer cache cleared.")
