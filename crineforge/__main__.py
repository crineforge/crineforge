import argparse
import sys
import logging
from .core import Trainer

def main():
    parser = argparse.ArgumentParser(description="Crineforge - Define and run AI training jobs.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using local data")
    train_parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    train_parser.add_argument("--data", type=str, required=True, help="Path to the dataset (PDF, CSV, TXT, JSON, MD)")
    train_parser.add_argument("--output", type=str, default="output_model", help="Output directory")
    train_parser.add_argument("--dry-run", action="store_true", help="Run the pipeline without training")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("crineforge.cli")
    
    if args.command == "train":
        try:
            logger.info(f"Starting Crineforge with model={args.model} and data={args.data}")
            trainer = Trainer()
            trainer.connect_model(args.model)
            trainer.load_data(args.data)
            trainer.auto_config()
            
            if args.dry_run:
                logger.info("Executing dry run...")
                trainer.dry_run()
            else:
                logger.info("Starting training...")
                trainer.train()
                trainer.save(args.output)
                logger.info(f"Training complete. Model saved to {args.output}")
                
        except ValueError as e:
            logger.error(f"[User Error] {e}")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"[Runtime Error] {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[Unexpected Error] {e}")
            raise

if __name__ == "__main__":
    main()
