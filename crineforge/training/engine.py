import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from ..model.gpu import GPUSensitive
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Engine:
    """Executes the fine-tuning loop securely and deterministically."""

    @staticmethod
    def construct_dataset(pairs: list[dict], tokenizer) -> Dataset:
        logger.info("Constructing HF Dataset from pairs...")
        
        # Pre-format texts for SFTTrainer
        formatted_pairs = []
        for p in pairs:
            # handle cases where instruction/response might be lists
            instructions = p.get('instruction', '')
            responses = p.get('response', '')
            
            if isinstance(instructions, str):
                instructions = [instructions]
                responses = [responses]
            
            for inst, resp in zip(instructions, responses):
                formatted_pairs.append({'text': f"Instruction: {inst}\nResponse: {resp}"})
                
        dataset = Dataset.from_list(formatted_pairs)
        return dataset

    @staticmethod
    def run(model, tokenizer, pairs: list[dict], hyperparams: dict) -> None:
        """Runs the trl SFTTrainer using the established safe hyperparams."""
        logger.info("Initializing Training Engine...")
        
        try:
            dataset = Engine.construct_dataset(pairs, tokenizer)
                
            training_args = TrainingArguments(
                output_dir="./crineforge_tmp_outputs",
                per_device_train_batch_size=hyperparams.get("batch_size", 1),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
                learning_rate=hyperparams.get("learning_rate", 2e-4),
                num_train_epochs=hyperparams.get("epochs", 1),
                logging_steps=10,
                optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
                save_strategy="no",
                fp16=True if GPUSensitive.get_strategy()["precision"] == "float16" else False,
                bf16=True if GPUSensitive.get_strategy()["precision"] == "bf16" else False,
                report_to="none" # Disable external logging for privacy
            )
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=hyperparams.get("max_seq_length", 512),
                args=training_args,
            )
            
            logger.info("Starting safe SFT fine-tuning. This will take time.")
            trainer.train()
            logger.info("Fine-tuning completed successfully.")
            
        except Exception as e:
            logger.error(f"Training failed securely: {str(e)}")
            raise e
