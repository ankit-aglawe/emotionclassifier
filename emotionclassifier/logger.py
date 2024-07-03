# emotionclassifier/logger.py
# Author: Ankit Aglawe

import logging
import warnings


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)

        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
    return logger


def set_logging_level(level):
    """Set the logging level for the package.

    Args:
        level (str): The logging level to set. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    level = level.upper()
    if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)

        logging.info(f"Logging level set to {level}")
        logging.info(f"Logging level set to {level}")
    else:
        raise ValueError(
            f"Invalid logging level: {level}. Use 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'."
        )
