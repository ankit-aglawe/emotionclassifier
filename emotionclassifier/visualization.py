# emotionclassifier/visualization.py
# Author: Ankit Aglawe

import matplotlib.pyplot as plt


def plot_emotion_distribution(emotion_probs, labels):
    if isinstance(emotion_probs[0], list):
        emotion_probs = emotion_probs[0]
    labels = list(labels)
    emotion_probs = [float(prob) for prob in emotion_probs]
    plt.bar(labels, emotion_probs)
    plt.xlabel("Emotions")
    plt.ylabel("Probability")
    plt.title("Emotion Distribution")
    plt.show()
