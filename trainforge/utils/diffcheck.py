from transformers import PreTrainedTokenizer
from .logger import get_logger

logger = get_logger(__name__)

def validate_token_length(tokenizer: PreTrainedTokenizer, original: str, structured: str, threshold: float = 0.9) -> bool:
    """
    Validates that the output token length is >= a certain percentage of the original.
    This helps prevent the structurer from summarizing or dropping critical information.
    """
    original_tokens = len(tokenizer.encode(original))
    structured_tokens = len(tokenizer.encode(structured))
    
    if original_tokens == 0:
        return True # Avoid div by zero for empty chunks
        
    ratio = structured_tokens / original_tokens
    logger.debug(f"Token length check: original={original_tokens}, structured={structured_tokens}, ratio={ratio:.2f}")
    
    if ratio < threshold:
        logger.warning(f"Structurer output rejected: Length ({structured_tokens}) is < {threshold*100}% of original ({original_tokens}).")
        return False
        
    return True
