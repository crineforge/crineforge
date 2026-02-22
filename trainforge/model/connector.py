from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from .gpu import GPUSensitive
from ..utils.logger import get_logger
import torch

logger = get_logger(__name__)

class ModelConnector:
    """Connects to HF models or local paths and prepares them based on GPU capabilities."""
    
    @staticmethod
    def load(model_id: str):
        strategy = GPUSensitive.get_strategy()
        logger.info(f"Loading target model '{model_id}' with strategy: {strategy}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model_kwargs = {
            "device_map": "auto" if strategy["device"] == "cuda" else "cpu",
            "trust_remote_code": True
        }
        
        try:
            if strategy["precision"] == "4bit":
                model_kwargs["load_in_4bit"] = True
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            elif strategy["precision"] == "float16":
                model_kwargs["torch_dtype"] = torch.float16
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                
            logger.info("Target model loaded successfully.")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load target model: {str(e)}")
            raise e
            
    @staticmethod
    def prepare_lora(model, r=8, alpha=16, dropout=0.05):
        """Attaches LoRA to the actual target model for training."""
        logger.info("Preparing model for LoRA fine-tuning...")
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"], # Safe default for most LLMs
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, config)
        peft_model.print_trainable_parameters()
        return peft_model
