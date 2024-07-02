# examples/example.py
# Author: Ankit Aglawe

from emotionclassifier.classifier import EmotionClassifier
from emotionclassifier.integration import DataFrameEmotionClassifier
from emotionclassifier.visualization import plot_emotion_distribution

# Initialize the classifier with default settings
classifier = EmotionClassifier()

# Classify a single text
text = "I am very happy today!"
result = classifier.predict(text)
print("Emotion:", result["label"])
print("Confidence:", result["confidence"])

# Batch processing
texts = ["I am very happy today!", "I am so sad."]
results = classifier.predict_batch(texts)
print("Batch processing results:", results)

# DataFrame Integration
import pandas as pd

df = pd.DataFrame({"text": ["I am very happy today!", "I am so sad."]})

df_classifier = DataFrameEmotionClassifier()
df = df_classifier.classify_dataframe(df, "text")
print(df)


# Classify a single text
text = "I am very happy today!"
result = classifier.predict(text)
print("Emotion:", result["label"])
print("Confidence:", result["confidence"])

# Plot emotion distribution
plot_emotion_distribution(result["probabilities"], classifier.labels.values())
