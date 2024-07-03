# emotionclassifier/trends.py
# Author: Ankit Aglawe

import matplotlib.pyplot as plt

from emotionclassifier.classifier import EmotionClassifier
from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


class EmotionTrends:
    def __init__(self, model_name="deberta-v3-small"):
        """Initialize the EmotionTrends with a model.

        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.classifier = EmotionClassifier(model_name=model_name)
            logger.info(f"EmotionTrends initialized with model {model_name}.")
        except Exception as e:
            logger.error(f"Failed to initialize EmotionTrends: {e}")
            raise

    def analyze_trends(self, texts):
        """Analyze emotion trends for a list of texts.

        Args:
            texts (list): A list of texts to analyze.

        Returns:
            list: A list of emotion labels.
        """
        try:
            emotions = [self.classifier.predict(text)["label"] for text in texts]
            logger.info("Emotion trends analysis complete.")
            return emotions
        except Exception as e:
            logger.error(f"Error in analyzing trends: {e}")
            raise

    def plot_trends(self, emotions):
        """Plot emotion trends over time.

        Args:
            emotions (list): A list of emotion labels.
        """
        try:
            plt.plot(range(len(emotions)), emotions)
            plt.xlabel("Text Index")
            plt.ylabel("Emotion")
            plt.title("Emotion Trends Over Time")
            plt.xticks(rotation="vertical")
            plt.show()
            logger.info("Emotion trends plotted successfully.")
        except Exception as e:
            logger.error(f"Error in plotting trends: {e}")
            raise
