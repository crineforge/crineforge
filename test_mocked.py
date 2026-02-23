import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['bitsandbytes'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['fitz'] = MagicMock() # PyMuPDF
sys.modules['datasets'] = MagicMock() 

import logging
logging.basicConfig(level=logging.INFO)

from crineforge import Trainer

def test_dry_run():
    trainer = Trainer(seed=42)
    trainer.connect_model("gpt2")
    trainer.load_data("sample.txt")
    trainer.auto_config()
    trainer.dry_run()
    from crineforge.utils.logger import default_logger as logger
    logger.info("MOCKED: Crineforge dry run executed successfully!")

if __name__ == "__main__":
    test_dry_run()
