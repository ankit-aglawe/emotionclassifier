# emotionclassifier/utils.py
# Author: Ankit Aglawe

from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


def get_label_with_threshold(probabilities, labels, threshold=0.5):
    """Get the label based on a probability threshold.

    Args:
        probabilities (Tensor): The probabilities for each label.
        labels (dict): The mapping from label ids to label names.
        threshold (float): The probability threshold.

    Returns:
        str: The label if the highest probability exceeds the threshold, else 'Uncertain'.
    """
    try:
        max_prob = probabilities.max().item()
        if max_prob > threshold:
            return labels[probabilities.argmax().item()]
        else:
            return "Uncertain"
    except Exception as e:
        logger.error(f"Error in get_label_with_threshold: {e}")
        raise
