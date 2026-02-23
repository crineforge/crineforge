import os
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Saver:
    """Handles saving trained models to disk."""

    @staticmethod
    def save_lora(model, output_dir: str):
        """Saves only the LoRA adapter weights, keeping size small and safe."""
        logger.info(f"Saving LoRA adapter to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        try:
            model.save_pretrained(output_dir)
            logger.info("LoRA adapter successfully saved.")
        except Exception as e:
            logger.error(f"Failed to save LoRA adapter: {str(e)}")
            raise e
