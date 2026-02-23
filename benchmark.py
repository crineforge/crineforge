import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crineforge import Trainer
from crineforge.utils.logger import default_logger as logger

def build_dummy_file():
    with open("dummy_extract.txt", "w", encoding="utf-8") as file:
        file.write("Build memory profiler testing text block\n")
        
def run_benchmark():
    logger.info("Starting memory benchmark test.")
    build_dummy_file()
    
    trainer = Trainer(debug_mode=True)
    trainer.connect_model("gpt2")
    trainer.load_data("dummy_extract.txt")
    
    # We will engage the benchmark dry run which mocks the generation natively to check memory lifecycle tracking correctly
    trainer.dry_run()

if __name__ == "__main__":
    run_benchmark()
