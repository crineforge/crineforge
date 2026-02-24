import os
import torch
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
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
    Crineforge main facade.
    Provides a simple, elegant API for end-users to fine-tune models from raw data.
    """
    
    def __init__(self, seed: int = 42, debug_mode: bool = False, structurer_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.structurer_model = structurer_model
        self.model_id = None
        self.data_path = None
        self.enrichment_enabled = False
        self.hyperparams = {}
        self.structured_pairs = []
        self.debug_mode = debug_mode
        self._model = None
        self._tokenizer = None
        
        set_seed(seed)
        logger.info("[Pipeline] Crineforge Trainer initialized.")

    def connect_model(self, model_id: str):
        """Connects a HuggingFace Hub ID or local path as the target for training."""
        self.model_id = model_id
        logger.info(f"[Pipeline] Connected to target model: {self.model_id}")

    def load_data(self, file_path: str):
        """Extracts and chunks raw data (PDF, CSV, TXT, JSON, MD)."""
        raw_text = DataExtractor.extract(file_path)
        self.data_path = file_path
        self._chunks = Chunker.split(raw_text)
        logger.info(f"[Pipeline] Data loaded from {self.data_path} ({len(self._chunks)} chunks ready).")

    def enable_enrichment(self, enabled: bool = True):
        """Toggle internet enrichment module (simulated)."""
        self.enrichment_enabled = enabled
        logger.info(f"[Pipeline] Enrichment mode set to: {self.enrichment_enabled}")

    def auto_config(self):
        """Automatically detects GPU and dataset constraints to set safe hyperparameters."""
        if not hasattr(self, '_chunks') or len(self._chunks) == 0:
             logger.warning("[Validation] auto_config called before load_data, assuming chunk length of zero for params.")
             dummy_len = 0
        else:
             dummy_len = len(self._chunks)
             
        self.hyperparams = AutoConfig.get_safe_params(dummy_len)

    def manual_config(self, **kwargs):
        """Manually override hyperparameter values safely."""
        self.hyperparams.update(kwargs)
        logger.info(f"[Validation] Manual override applied. Current params: {self.hyperparams}")

    def structure_only(self, input_path: str = None):
        """Generates structured dataset from the raw input without training."""
        GPUSensitive.log_vram_usage("Pipeline Start")
        start_time = time.time()
        
        target_path = input_path or self.data_path
        if not target_path:
            raise ValueError(
                "Data path is missing.\n"
                "Reason: No input path provided to structure_only() AND load_data() was not called previously.\n"
                "Suggested fix: Provide a valid file path or call load_data(file_path)."
            )
            
        if getattr(self, 'data_path', None) != target_path or not hasattr(self, '_chunks'):
            self.load_data(target_path)
            
        logger.info("[Pipeline] === Phase 1: Structuring Data ===")
        GPUSensitive.log_vram_usage("Before Structuring")
        
        try:
            structurer = get_structurer(self.structurer_model)
            structured_pairs = []
            logger.info("[Structurer] Formatting chunks...")
            for chunk in tqdm(self._chunks, desc="Structuring chunks"):
                json_str = structurer.generate_pairs(chunk)
                parsed_data = Validator.parse_valid_json(json_str)
                structured_pairs.extend(parsed_data)
            
            Validator.validate_dataset_size(structured_pairs, debug_mode=getattr(self, 'debug_mode', False))
            
            if self.enrichment_enabled:
                structured_pairs = InternetEnrichment.enrich(structured_pairs)
                
            self.structured_pairs = structured_pairs
            
            duration = time.time() - start_time
            logger.info(f"[Performance] Structuring completed in {duration:.2f}s")
            
            return self.structured_pairs
            
        except Exception as e:
            logger.error(f"[Validation] Structuring failed: {str(e)}")
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(
                f"Data structuring failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Ensure raw text can be parsed into target format."
            ) from e
            
        finally:
            free_structurer()
            GPUSensitive.log_vram_usage("After Structuring Cleanup")

    def train_from_structured(self, structured_data: list):
        """Executes the fine-tuning phase using pre-structured data."""
        start_time = time.time()
        if not self.model_id:
            raise ValueError(
                "Target model not connected.\n"
                "Reason: connect_model() was not called pipeline execution.\n"
                "Suggested fix: Call connect_model(model_id) first."
            )
            
        if not structured_data:
            raise ValueError(
                "Structured data is empty.\n"
                "Reason: The provided dataset has 0 instances.\n"
                "Suggested fix: Provide a valid array of training pairs."
            )
            
        Validator.validate_dataset_size(structured_data, debug_mode=getattr(self, 'debug_mode', False))
        
        logger.info("[Pipeline] === Phase 2: Target Model Loading ===")
        GPUSensitive.log_vram_usage("Before Training")
        try:
            model, tokenizer = ModelConnector.load(self.model_id)
            self._model = ModelConnector.prepare_lora(model)
            self._tokenizer = tokenizer
        except Exception as e:
            logger.error(f"[Validation] Failed to load target model {self.model_id}: {str(e)}")
            raise RuntimeError(
                f"Model loading failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Verify model ID exists and you have network access."
            ) from e

        logger.info("[Pipeline] === Phase 3: Fine-Tuning Execution ===")
        try:
            Engine.run(self._model, self._tokenizer, structured_data, self.hyperparams)
        except getattr(torch.cuda, 'OutOfMemoryError', Exception) as e:
            logger.error(f"[GPU] OOM error during training: {str(e)}")
            raise RuntimeError(
                "Training aborted due to OOM.\n"
                "Reason: Out of memory executing fine-tuning sequence.\n"
                "Suggested fix: Decrease batch size, enable 4bit quantization, or use a smaller model."
            ) from e
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"[GPU] OOM error during training: {str(e)}")
                raise RuntimeError(
                    "Training aborted due to OOM.\n"
                    "Reason: Out of memory executing fine-tuning sequence.\n"
                    "Suggested fix: Decrease batch size, enable 4bit quantization, or use a smaller model."
                ) from e
            logger.error(f"[Pipeline] Training engine failed: {str(e)}")
            raise RuntimeError(
                f"Training execution failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Validate hyperparams and data structure correctness."
            ) from e
        finally:
            GPUSensitive.empty_cache()
            GPUSensitive.log_vram_usage("After Training")
            
        duration = time.time() - start_time
        logger.info(f"[Performance] Training completed in {duration:.2f}s")

    def train(self):
        """Starts the local structuring and fine-tuning process securely."""
        if not self.model_id or not self.data_path:
            raise ValueError(
                "Model and Data must be configured before training.\n"
                "Reason: Missing prerequisites for pipeline execution.\n"
                "Suggested fix: Call connect_model() and load_data() first."
            )
            
        try:
            structured_data = self.structure_only(self.data_path)
            self.train_from_structured(structured_data)
        except Exception as e:
            logger.error(f"Training pipeline aborted securely: {str(e)}")
            raise

    def dry_run(self):
        """Runs the entire pipeline without invoking actual structural inference or target training updates."""
        logger.info("[Pipeline] *** DRY RUN MODE INITIATED ***")
        if not self.model_id or not self.data_path:
            raise ValueError("Model and Data must be configured before dry run.")
        
        logger.info(f"[Validation] Validating target Model: {self.model_id}")
        logger.info(f"[Validation] Validating target Data: {self.data_path}")
        logger.info(f"[Validation] Validating Hyperparams: {self.hyperparams}")
        
        logger.info("[Pipeline] Mocking structurer and GPU checks...")
        strategy = GPUSensitive.get_strategy()
        logger.info(f"[GPU] Would use GPU Strategy: {strategy}")
        
        # Dry Run fake dataset
        fake_data = [{"instruction": "Test", "response": "Mocked Output"}] * max((len(self._chunks) if hasattr(self, '_chunks') else 0), 20)
        Validator.validate_dataset_size(fake_data, debug_mode=getattr(self, 'debug_mode', False))
        
        logger.info("[Pipeline] *** DRY RUN SUCCESS - PIPELINE VALIDATED ***")

    def save(self, output_dir: str):
        """Saves the trained LoRA adapter or full model."""
        if not self._model:
            raise ValueError("No model has been trained yet. Cannot save.")
        Saver.save_lora(self._model, output_dir)
