# emotionclassifier/model_loader.py
# Author: Ankit Aglawe

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotionclassifier.suppress_tqdm import suppress_tqdm


def load_model_and_tokenizer(model_name, suppress_output=True):
    with suppress_tqdm(suppress_output):
        print(f"Downloading model and tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            f"AnkitAI/{model_name}-base-emotions-classifier"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            f"AnkitAI/{model_name}-base-emotions-classifier"
        )
    print("Download complete.")
    return model, tokenizer
