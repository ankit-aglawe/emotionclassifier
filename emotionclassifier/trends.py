# emotionclassifier/trends.py
# Author: Ankit Aglawe

import matplotlib.pyplot as plt

from emotionclassifier.classifier import EmotionClassifier


class EmotionTrends:
    def __init__(self, model_name="deberta-v3-small"):
        self.classifier = EmotionClassifier(model_name=model_name)

    def analyze_trends(self, texts):
        emotions = [self.classifier.predict(text)["label"] for text in texts]
        return emotions

    def plot_trends(self, emotions):
        plt.plot(range(len(emotions)), emotions)
        plt.xlabel("Text Index")
        plt.ylabel("Emotion")
        plt.title("Emotion Trends Over Time")
        plt.xticks(rotation="vertical")
        plt.show()
