import torch
import gc
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GPUSensitive:
    """Detects CUDA availability and provides VRAM-based precision strategies to avoid OOM."""
    
    @staticmethod
    def log_vram_usage(step_name: str):
        """Logs allocated and reserved VRAM safely, tracking pipeline overhead."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            
            logger.info(
                f"[VRAM] {step_name} | "
                f"Allocated: {allocated:.2f} GB | "
                f"Reserved: {reserved:.2f} GB"
            )
        else:
            logger.info(f"[VRAM] {step_name} | CPU mode")
    
    @staticmethod
    def get_strategy() -> dict:
        if not torch.cuda.is_available():
            logger.warning("[GPU] CUDA not available. Falling back to CPU precision.")
            return {"device": "cpu", "precision": "float32"}
            
        try:
            device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
            logger.info(f"[GPU] Detected GPU: {torch.cuda.get_device_name(device)} with {total_vram:.2f} GB VRAM")
            
            if total_vram >= 16.0:
                is_bf16 = torch.cuda.is_bf16_supported()
                if is_bf16:
                    logger.info("[GPU] VRAM >= 16GB with BF16 support. Recommending bf16 precision.")
                    return {"device": "cuda", "precision": "bf16"}
                else:
                    logger.info("[GPU] VRAM >= 16GB. Recommending fp16 precision.")
                    return {"device": "cuda", "precision": "float16"}
            elif total_vram >= 8.0:
                logger.info("[GPU] VRAM >= 8GB. Recommending 4-bit quantization.")
                return {"device": "cuda", "precision": "4bit"}
            else:
                logger.warning("[GPU] VRAM < 8GB. Falling back to CPU to prevent OOM. (Training will be slow)")
                return {"device": "cpu", "precision": "float32"}
        except Exception as e:
            logger.error(f"[GPU] Error reading GPU properties: {str(e)}")
            return {"device": "cpu", "precision": "float32"}
            
    @staticmethod
    def empty_cache():
        """Safely empties CUDA cache and collects garbage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("[GPU] CUDA memory cache cleared safely.")
