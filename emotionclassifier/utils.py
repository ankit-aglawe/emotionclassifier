# emotionclassifier/utils.py
# Author: Ankit Aglawe


def get_label_with_threshold(probabilities, labels, threshold=0.5):
    max_prob = probabilities.max().item()
    if max_prob > threshold:
        return labels[probabilities.argmax().item()]
    else:
        return "Uncertain"
