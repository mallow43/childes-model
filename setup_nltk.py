#!/usr/bin/env python3
"""
Download required NLTK data for the CHILDES model.
Run this after installing requirements: python setup_nltk.py
"""

import nltk

REQUIRED_DATA = [
    'universal_tagset',           # For universal POS tag mapping
    'averaged_perceptron_tagger', # POS tagger model
    'averaged_perceptron_tagger_eng',  # English-specific tagger
]


def main():
    print("Downloading required NLTK data...")
    for resource in REQUIRED_DATA:
        print(f"  - {resource}")
        nltk.download(resource, quiet=True)
    print("Done! NLTK data is ready.")


if __name__ == "__main__":
    main()
