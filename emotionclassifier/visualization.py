# emotionclassifier/visualization.py
# Author: Ankit Aglawe

import matplotlib.pyplot as plt

from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


def plot_emotion_distribution(emotion_probs, labels):
    """Plot the distribution of emotion probabilities.

    Args:
        emotion_probs (list): A list of emotion probabilities.
        labels (list): A list of emotion labels.
    """
    try:
        if isinstance(emotion_probs[0], list):
            emotion_probs = emotion_probs[0]
        labels = list(labels)
        emotion_probs = [float(prob) for prob in emotion_probs]
        plt.bar(labels, emotion_probs)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        plt.title("Emotion Distribution")
        plt.show()
        logger.info("Emotion distribution plotted successfully.")
    except Exception as e:
        logger.error(f"Error in plotting emotion distribution: {e}")
        raise
