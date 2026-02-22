from ..model.gpu import GPUSensitive
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AutoConfig:
    """Heuristic engine to determine safe hyperparameters based on hardware and dataset constraints."""
    
    @staticmethod
    def get_safe_params(dataset_len: int) -> dict:
        strategy = GPUSensitive.get_strategy()
        
        params = {
            "batch_size": 2,
            "learning_rate": 2e-4,
            "epochs": 3,
            "gradient_accumulation_steps": 4
        }
        
        if dataset_len > 1000:
            params["epochs"] = 1
        elif dataset_len > 500:
            params["epochs"] = 2
            
        if strategy["precision"] == "4bit":
            params["batch_size"] = 1
            params["gradient_accumulation_steps"] = 8
        elif strategy["precision"] == "float16":
            params["batch_size"] = 2
            params["gradient_accumulation_steps"] = 4
        elif strategy["device"] == "cpu":
            params["batch_size"] = 1
            params["epochs"] = 1 
            params["learning_rate"] = 5e-5
            logger.warning("CPU training is exceptionally slow. Capped epochs at 1 for safety.")
            
        logger.info(f"AutoConfig recommended hyperparameters: {params}")
        return params
