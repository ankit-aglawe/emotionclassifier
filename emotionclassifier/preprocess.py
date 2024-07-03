# emotionclassifier/preprocess.py
# Author: Ankit Aglawe

import re

from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


def default_clean_text(text):
    """Clean the input text using default rules.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    try:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.error(f"Error in default_clean_text: {e}")
        raise


class Preprocessor:
    def __init__(self, clean_function=default_clean_text):
        """Initialize the Preprocessor with a cleaning function.

        Args:
            clean_function (callable): The function to use for cleaning text.
        """
        try:
            self.clean_function = clean_function
            logger.info("Preprocessor initialized with custom clean function.")
        except Exception as e:
            logger.error(f"Error in initializing Preprocessor: {e}")
            raise

    def clean(self, text):
        """Clean the input text.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        try:
            return self.clean_function(text)
        except Exception as e:
            logger.error(f"Error in cleaning text: {e}")
            raise
