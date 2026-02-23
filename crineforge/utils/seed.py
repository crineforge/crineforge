import random
import os
import torch
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    """Sets deterministic seed for Python, NumPy, and PyTorch cleanly."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring determinism in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Deterministic seed set to {seed}")
