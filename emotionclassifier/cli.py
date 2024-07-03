# emotionclassifier/cli.py
# Author: Ankit Aglawe

import click

from emotionclassifier.classifier import EmotionClassifier
from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--model", default="deberta-v3-small", help="Model name")
@click.option(
    "--text", prompt="Text to classify", help="Text for emotion classification"
)
def classify_text(model, text):
    """Classify the emotion of a given text using a specified model.

    Args:
        model (str): The name of the model to use.
        text (str): The text to classify.
    """
    try:
        classifier = EmotionClassifier(model_name=model)
        result = classifier.predict(text)
        click.echo(f"Emotion: {result['label']}")
        click.echo(f"Confidence: {result['confidence']}")
        logger.info(f"Text classified successfully: {text}")
    except Exception as e:
        logger.error(f"Error in classify_text command: {e}")
        click.echo("Error in classifying text. Check logs for details.")


if __name__ == "__main__":
    classify_text()
