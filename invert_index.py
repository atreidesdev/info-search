import os
import json
import re
from collections import defaultdict

def build_inverted_index(base_path="pages"):
    inverted_index = defaultdict(set)
    for page_folder in os.listdir(base_path):
        lemmatized_path = os.path.join(base_path, page_folder, "lemmatized_tokens.txt")
        if not os.path.exists(lemmatized_path):
            continue
        with open(lemmatized_path, "r", encoding="utf-8") as f:
            tokens = f.read().split()
            for token in tokens:
                inverted_index[token].add(page_folder)

    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in inverted_index.items()}, f, ensure_ascii=False, indent=2)

    return inverted_index

def load_inverted_index(path="inverted_index.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


if __name__ == "__main__":
    index = build_inverted_index()
