# setup.py
# Author: Ankit Aglawe

from setuptools import find_packages, setup

setup(
    name="emotionclassifier",
    version="0.1.0",
    description="A flexible emotion classifier with support for multiple models",
    author="Ankit Aglawe",
    author_email="aglawe.ankit@example.com",
    url="https://github.com/ankit-aglawe/emotionclassifier",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "matplotlib",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "emotionclassifier=emotion_classifier.cli:classify_text",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
