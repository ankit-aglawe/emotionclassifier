# emotionclassifier/integration.py
# Author: Ankit Aglawe

from .classifier import EmotionClassifier


class DataFrameEmotionClassifier:
    def __init__(self, model_name="deberta-v3-small"):
        self.classifier = EmotionClassifier(model_name=model_name)

    def classify_dataframe(self, df, text_column):
        df["emotion"] = df[text_column].apply(
            lambda x: self.classifier.predict(x)["label"]
        )
        return df
