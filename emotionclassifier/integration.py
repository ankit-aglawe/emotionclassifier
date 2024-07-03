# emotionclassifier/integration.py
# Author: Ankit Aglawe

from emotionclassifier.classifier import EmotionClassifier
from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


class DataFrameEmotionClassifier:
    def __init__(self, model_name="deberta-v3-small"):
        """Initialize the DataFrameEmotionClassifier with a model.

        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.classifier = EmotionClassifier(model_name=model_name)
            logger.info(
                f"DataFrameEmotionClassifier initialized with model {model_name}."
            )
        except Exception as e:
            logger.error(f"Failed to initialize DataFrameEmotionClassifier: {e}")
            raise

    def classify_dataframe(self, df, text_column):
        """Classify emotions for each text in a DataFrame column.

        Args:
            df (pd.DataFrame): The DataFrame containing texts to classify.
            text_column (str): The name of the column with texts.

        Returns:
            pd.DataFrame: The DataFrame with an added 'emotion' column.
        """
        try:
            df["emotion"] = df[text_column].apply(
                lambda x: self.classifier.predict(x)["label"]
            )
            logger.info(f"DataFrame classified successfully. {len(df)} rows processed.")
            return df
        except Exception as e:
            logger.error(f"Error in classifying DataFrame: {e}")
            raise
