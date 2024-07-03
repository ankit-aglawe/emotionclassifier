# emotionclassifier/model_loader.py
# Author: Ankit Aglawe

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotionclassifier.logger import get_logger
from emotionclassifier.suppress_tqdm import suppress_tqdm

logger = get_logger(__name__)


def load_model_and_tokenizer(model_name, suppress_output=True):
    """Load the model and tokenizer from the given model name.

    Args:
        model_name (str): The name of the model to load.
        suppress_output (bool): Whether to suppress output during model loading.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    try:
        with suppress_tqdm(suppress_output):
            logger.info(f"Downloading model and tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                f"AnkitAI/{model_name}-base-emotions-classifier"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                f"AnkitAI/{model_name}-base-emotions-classifier"
            )
        logger.info("Download complete.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error in loading model and tokenizer: {e}")
        raise
