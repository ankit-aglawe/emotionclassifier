# emotionclassifier/suppress_tqdm.py
# Author: Ankit Aglawe

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_tqdm(enable=True):
    if enable:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        if enable:
            sys.stdout = original_stdout
