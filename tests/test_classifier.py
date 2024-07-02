# tests/test_classifier.py
# Author: Ankit Aglawe

import unittest

import pandas as pd

from emotionclassifier.classifier import EmotionClassifier
from emotionclassifier.integration import DataFrameEmotionClassifier


class TestEmotionClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = EmotionClassifier()

    def test_single_prediction(self):
        text = "I am very happy today!"
        result = self.classifier.predict(text)
        self.assertIn(result["label"], self.classifier.labels.values())
        self.assertTrue(0 <= result["confidence"] <= 1)

    def test_batch_prediction(self):
        texts = ["I am very happy today!", "I am so sad."]
        results = self.classifier.predict_batch(texts)
        for result in results:
            self.assertIn(result["label"], self.classifier.labels.values())
            self.assertTrue(0 <= result["confidence"] <= 1)

    def test_dataframe_integration(self):
        df = pd.DataFrame({"text": ["I am very happy today!", "I am so sad."]})
        df_classifier = DataFrameEmotionClassifier()
        df = df_classifier.classify_dataframe(df, "text")
        for emotion in df["emotion"]:
            self.assertIn(emotion, self.classifier.labels.values())


if __name__ == "__main__":
    unittest.main()
