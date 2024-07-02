# emotionclassifier/classifier.py
# Author: Ankit Aglawe

import torch

from emotionclassifier.model_loader import load_model_and_tokenizer
from emotionclassifier.preprocess import Preprocessor


class EmotionClassifier:
    def __init__(self, model_name="deberta-v3-small", suppress_output=True):
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name, suppress_output
        )
        self.preprocessor = Preprocessor()
        self.labels = self.model.config.id2label

    def preprocess(self, text):
        return self.preprocessor.clean(text)

    def predict(self, text):
        text = self.preprocess(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        label = self.labels[probabilities.argmax().item()]
        confidence = probabilities.max().item()
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
        }

    def predict_batch(self, texts):
        texts = [self.preprocess(text) for text in texts]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        results = [
            {
                "label": self.labels[prob.argmax().item()],
                "confidence": prob.max().item(),
            }
            for prob in probabilities
        ]
        return results
