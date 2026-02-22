import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainforge import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    
    # Example flow
    trainer.connect_model("gpt2")
    
    # Load test data
    trainer.load_data("sample.txt")
    
    trainer.auto_config()
    
    # Test the pipeline without executing the training loops
    trainer.dry_run()
    
    print("TrainForge dry run executed successfully!")
