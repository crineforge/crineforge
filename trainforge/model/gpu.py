import torch
import gc
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GPUSensitive:
    """Detects CUDA availability and provides VRAM-based precision strategies to avoid OOM."""
    
    @staticmethod
    def get_strategy() -> dict:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU precision.")
            return {"device": "cpu", "precision": "float32"}
            
        try:
            device = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
            logger.info(f"Detected GPU: {torch.cuda.get_device_name(device)} with {total_vram:.2f} GB VRAM")
            
            if total_vram >= 16.0:
                logger.info("VRAM >= 16GB. Recommending fp16 precision.")
                return {"device": "cuda", "precision": "float16"}
            elif total_vram >= 8.0:
                logger.info("VRAM >= 8GB. Recommending 4-bit quantization.")
                return {"device": "cuda", "precision": "4bit"}
            else:
                logger.warning("VRAM < 8GB. Falling back to CPU to prevent OOM. (Training will be slow)")
                return {"device": "cpu", "precision": "float32"}
        except Exception as e:
            logger.error(f"Error reading GPU properties: {str(e)}")
            return {"device": "cpu", "precision": "float32"}
            
    @staticmethod
    def empty_cache():
        """Safely empties CUDA cache and collects garbage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("CUDA memory cache cleared safely.")
