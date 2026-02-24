import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from ..model.gpu import GPUSensitive
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Engine:
    """Executes the fine-tuning loop securely and deterministically."""

    @staticmethod
    def construct_dataset(pairs: list[dict], tokenizer) -> Dataset:
        logger.info("Constructing HF Dataset from pairs...")
        dataset = Dataset.from_list(pairs)
        return dataset

    @staticmethod
    def run(model, tokenizer, pairs: list[dict], hyperparams: dict) -> None:
        """Runs the trl SFTTrainer using the established safe hyperparams."""
        logger.info("Initializing Training Engine...")
        
        try:
            dataset = Engine.construct_dataset(pairs, tokenizer)
            
            def formatting_prompts_func(example):
                output_texts = []
                instructions = example.get('instruction', [])
                responses = example.get('response', [])
                
                if isinstance(instructions, str):
                    instructions = [instructions]
                    responses = [responses]
                    
                for i in range(len(instructions)):
                    text = f"Instruction: {instructions[i]}\nResponse: {responses[i]}"
                    output_texts.append(text)
                return output_texts
                
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
                report_to="none" # Disable external logging for privacy
            )
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                formatting_func=formatting_prompts_func,
                processing_class=tokenizer,
                max_seq_length=hyperparams.get("max_seq_length", 512),
                args=training_args,
            )
            
            logger.info("Starting safe SFT fine-tuning. This will take time.")
            trainer.train()
            logger.info("Fine-tuning completed successfully.")
            
        except Exception as e:
            logger.error(f"Training failed securely: {str(e)}")
            raise e
