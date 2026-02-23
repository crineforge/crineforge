from ..utils.logger import get_logger

logger = get_logger(__name__)

class Chunker:
    """Splits large texts into manageable chunks for the structurer."""
    
    @staticmethod
    def split(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
            
        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks
