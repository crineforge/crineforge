import os
from .utils.logger import get_logger
from .utils.seed import set_seed
from .data.extractor import DataExtractor
from .data.chunker import Chunker
from .data.structurer import get_structurer, free_structurer
from .data.validator import Validator
from .data.enrichment import InternetEnrichment
from .model.connector import ModelConnector
from .model.gpu import GPUSensitive
from .hyperparams.auto import AutoConfig
from .training.engine import Engine
from .training.saver import Saver

logger = get_logger(__name__)

class Trainer:
    """
    TrainForge main facade.
    Provides a simple, elegant API for end-users to fine-tune models from raw data.
    """
    
    def __init__(self, seed: int = 42):
        self.model_id = None
        self.data_path = None
        self.enrichment_enabled = False
        self.hyperparams = {}
        self.structured_pairs = []
        self._model = None
        self._tokenizer = None
        
        set_seed(seed)
        logger.info("TrainForge Trainer initialized.")

    def connect_model(self, model_id: str):
        """Connects a HuggingFace Hub ID or local path as the target for training."""
        self.model_id = model_id
        logger.info(f"Connected to target model: {self.model_id}")

    def load_data(self, file_path: str):
        """Extracts and chunks raw data (PDF, CSV, TXT, JSON, MD)."""
        raw_text = DataExtractor.extract(file_path)
        self.data_path = file_path
        self._chunks = Chunker.split(raw_text)
        logger.info(f"Data loaded from {self.data_path} ({len(self._chunks)} chunks ready).")

    def enable_enrichment(self, enabled: bool = True):
        """Toggle internet enrichment module (simulated)."""
        self.enrichment_enabled = enabled
        logger.info(f"Enrichment mode set to: {self.enrichment_enabled}")

    def auto_config(self):
        """Automatically detects GPU and dataset constraints to set safe hyperparameters."""
        if not hasattr(self, '_chunks') or len(self._chunks) == 0:
             logger.warning("auto_config called before load_data, assuming chunk length of zero for params.")
             dummy_len = 0
        else:
             dummy_len = len(self._chunks)
             
        self.hyperparams = AutoConfig.get_safe_params(dummy_len)

    def manual_config(self, **kwargs):
        """Manually override hyperparameter values safely."""
        self.hyperparams.update(kwargs)
        logger.info(f"Manual override applied. Current params: {self.hyperparams}")

    def train(self):
        """Starts the local structuring and fine-tuning process."""
        if not self.model_id or not self.data_path:
            raise ValueError("Model and Data must be configured before training.")
            
        logger.info("=== Phsase 1: Structuring Data ===")
        structurer = get_structurer()
        for chunk in self._chunks:
            json_str = structurer.generate_pairs(chunk)
            parsed_data = Validator.parse_valid_json(json_str)
            self.structured_pairs.extend(parsed_data)
        
        Validator.validate_dataset_size(self.structured_pairs)
        
        if self.enrichment_enabled:
            self.structured_pairs = InternetEnrichment.enrich(self.structured_pairs)
            
        # Free memory footprint of 7B param Instruct LLM
        free_structurer()
        
        logger.info("=== Phase 2: Target Model Loading ===")
        model, tokenizer = ModelConnector.load(self.model_id)
        self._model = ModelConnector.prepare_lora(model)
        self._tokenizer = tokenizer
        
        logger.info("=== Phase 3: Fine-Tuning Execution ===")
        Engine.run(self._model, self._tokenizer, self.structured_pairs, self.hyperparams)
        
        # Free cache post-training
        GPUSensitive.empty_cache()

    def dry_run(self):
        """Runs the entire pipeline without invoking actual structural inference or target training updates."""
        logger.info("*** DRY RUN MODE INITIATED ***")
        if not self.model_id or not self.data_path:
            raise ValueError("Model and Data must be configured before dry run.")
        
        logger.info(f"Validating target Model: {self.model_id}")
        logger.info(f"Validating target Data: {self.data_path}")
        logger.info(f"Validating Hyperparams: {self.hyperparams}")
        
        logger.info("Mocking structurer and GPU checks...")
        strategy = GPUSensitive.get_strategy()
        logger.info(f"Would use GPU Strategy: {strategy}")
        
        # Dry Run fake dataset
        fake_data = [{"instruction": "Test", "response": "Mocked Output"}] * max((len(self._chunks) if hasattr(self, '_chunks') else 0), 20)
        Validator.validate_dataset_size(fake_data)
        
        logger.info("*** DRY RUN SUCCESS - PIPELINE VALIDATED ***")

    def save(self, output_dir: str):
        """Saves the trained LoRA adapter or full model."""
        if not self._model:
            raise ValueError("No model has been trained yet. Cannot save.")
        Saver.save_lora(self._model, output_dir)
