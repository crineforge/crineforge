import logging

def get_logger(name: str = "crineforge") -> logging.Logger:
    """Returns a pre-configured logger for Crineforge."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

default_logger = get_logger("crineforge")
