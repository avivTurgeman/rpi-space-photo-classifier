import logging
import sys

def get_logger(name):
    """Get a logger that prints INFO level messages to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # ensure no duplicate handlers if called multiple times
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)  # format: timestamp - module - level - message:contentReference[oaicite:36]{index=36}
        logger.addHandler(handler)
    return logger
