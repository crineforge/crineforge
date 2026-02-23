from ..utils.logger import get_logger

logger = get_logger(__name__)

class InternetEnrichment:
    """Simulated web retrieval module. Disabled by default."""
    
    @staticmethod
    def enrich(data_pairs: list[dict]) -> list[dict]:
        """Appends mocked internet enrichment tags without replacing original context."""
        logger.info("Simulating internet enrichment...")
        
        enriched_data = []
        for pair in data_pairs:
            new_pair = dict(pair)
            # Simulated logic: if instruction asks 'what is', sometimes internet data adds context
            if "what" in str(new_pair.get("instruction", "")).lower():
                new_pair["response"] = f"{new_pair.get('response', '')} [ENRICHED: Cross-reference verified]"
            enriched_data.append(new_pair)
            
        return enriched_data
