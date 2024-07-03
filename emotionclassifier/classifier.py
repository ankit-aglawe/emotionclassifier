# emotionclassifier/classifier.py
# Author: Ankit Aglawe

import torch

from emotionclassifier.logger import get_logger
from emotionclassifier.model_loader import load_model_and_tokenizer
from emotionclassifier.preprocess import Preprocessor

logger = get_logger(__name__)


class EmotionClassifier:
    def __init__(self, model_name="deberta-v3-small", suppress_output=True):
        """Initialize the EmotionClassifier with a model and tokenizer.

        Args:
            model_name (str): The name of the model to use.
            suppress_output (bool): Whether to suppress output during model loading.
        """
        try:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name, suppress_output
            )
            self.preprocessor = Preprocessor()
            self.labels = self.model.config.id2label
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EmotionClassifier: {e}")
            raise

    def preprocess(self, text):
        """Clean the input text.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The cleaned text.
        """
        try:
            return self.preprocessor.clean(text)
        except Exception as e:
            logger.error(f"Error in preprocessing text: {e}")
            raise

    def predict(self, text):
        """Predict the emotion of a given text.

        Args:
            text (str): The text to classify.

        Returns:
            dict: A dictionary containing the label, confidence, and probabilities.
        """
        try:
            text = self.preprocess(text)
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            label = self.labels[probabilities.argmax().item()]
            confidence = probabilities.max().item()
            logger.info(f"Prediction successful for text: {text}")
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }
        except Exception as e:
            logger.error(f"Error in predicting emotion: {e}")
            raise

    def predict_batch(self, texts):
        """Predict the emotions of a batch of texts.

        Args:
            texts (list): A list of texts to classify.

        Returns:
            list: A list of dictionaries containing labels and confidences for each text.
        """
        try:
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
            logger.info(f"Batch prediction successful for {len(texts)} texts.")
            return results
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
