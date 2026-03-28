import json
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Validator:
    """Validates the structured outputs for proper JSON sizing and safety."""
    
    @staticmethod
    def validate_dataset_size(dataset: list, debug_mode: bool = False) -> bool:
        size = len(dataset)
        logger.info(f"Dataset generated with {size} examples.")
        
        if size < 20 and not debug_mode:
            raise ValueError(
                f"Dataset size check failed.\n"
                f"Reason: Found {size} examples, but need at least 20 instances to prevent poor fine-tuning results.\n"
                f"Suggested fix: Add more raw data or set debug_mode=True in Trainer to bypass during testing."
            )
            
        return True

    @staticmethod
    def parse_valid_json(text: str) -> list[dict]:
        """Strict JSON parsing validation."""
        try:
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            data = json.loads(text)
            if not isinstance(data, list):
                logger.warning("Evaluated JSON is not a list. Attempting to wrap in list.")
                return [data]
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Structurer failed to output valid JSON: {str(e)}")
            return []
