# emotionclassifier/cli.py
# Author: Ankit Aglawe

import click

from emotionclassifier.classifier import EmotionClassifier


@click.command()
@click.option("--model", default="deberta-v3-small", help="Model name")
@click.option(
    "--text", prompt="Text to classify", help="Text for emotion classification"
)
def classify_text(model, text):
    classifier = EmotionClassifier(model_name=model)
    result = classifier.predict(text)
    click.echo(f"Emotion: {result['label']}")
    click.echo(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    classify_text()
