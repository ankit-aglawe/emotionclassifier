# emotionclassifier/preprocess.py
# Author: Ankit Aglawe

import re


def default_clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Preprocessor:
    def __init__(self, clean_function=default_clean_text):
        self.clean_function = clean_function

    def clean(self, text):
        return self.clean_function(text)
