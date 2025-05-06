import os
from collections import defaultdict
import math

PAGES_DIR = "pages"

import json

def save_tf_idf_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tokens = file.read().splitlines()
    return tokens


def read_lemmatized(file_path):
    lemmatized = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            lemma = parts[0]
            forms = set(parts[1:])
            lemmatized[lemma] = forms
    return lemmatized


def calculate_tf(tokens):
    term_counts = defaultdict(int)
    total_terms = len(tokens)
    for token in tokens:
        term_counts[token] += 1
    tf = {term: count / total_terms for term, count in term_counts.items()}
    return tf


def calculate_idf(pages_dir, terms):
    doc_count = defaultdict(int)
    total_docs = 0
    for folder_name in os.listdir(pages_dir):
        folder_path = os.path.join(pages_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        total_docs += 1
        tokens_file = os.path.join(folder_path, "tokens.txt")
        tokens = set(read_tokens(tokens_file))
        for term in terms:
            if term in tokens:
                doc_count[term] += 1
    idf = {term: math.log(total_docs / (doc_count[term] + 1)) for term in terms}
    return idf


def save_results(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for term, (idf, tf_idf) in sorted(data.items()):
            file.write(f"{term} {idf} {tf_idf}\n")


def process_page(page_dir):
    tokens_file = os.path.join(page_dir, "tokens.txt")
    lemmatized_file = os.path.join(page_dir, "lemmatized_tokens.txt")

    tokens = read_tokens(tokens_file)
    lemmatized = read_lemmatized(lemmatized_file)

    term_tf = calculate_tf(tokens)

    all_terms = set(term_tf.keys())
    term_idf = calculate_idf(PAGES_DIR, all_terms)

    term_tf_idf = {term: (term_idf[term], term_tf[term] * term_idf[term]) for term in all_terms}

    terms_result_file = os.path.join(page_dir, "terms_tf_idf.txt")
    save_results(terms_result_file, term_tf_idf)

    lemma_tf = {}
    for lemma, forms in lemmatized.items():
        total_forms_tf = sum(term_tf.get(form, 0) for form in forms)
        lemma_tf[lemma] = total_forms_tf

    all_lemmas = set(lemma_tf.keys())
    lemma_idf = calculate_idf(PAGES_DIR, all_lemmas)

    lemma_tf_idf = {lemma: (lemma_idf[lemma], lemma_tf[lemma] * lemma_idf[lemma]) for lemma in all_lemmas}

    lemmas_result_file = os.path.join(page_dir, "lemmas_tf_idf.txt")
    save_results(lemmas_result_file, lemma_tf_idf)
    save_tf_idf_json(os.path.join(page_dir, "terms_tf_idf.json"),
                     {term: {"idf": idf, "tfidf": tfidf} for term, (idf, tfidf) in term_tf_idf.items()})
    save_tf_idf_json(os.path.join(page_dir, "lemmas_tf_idf.json"),
                     {lemma: {"idf": idf, "tfidf": tfidf} for lemma, (idf, tfidf) in lemma_tf_idf.items()})

def main():
    if not os.path.exists(PAGES_DIR):
        print(f"Папка {PAGES_DIR} не существует.")
        return

    for folder_name in os.listdir(PAGES_DIR):
        folder_path = os.path.join(PAGES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        process_page(folder_path)
        print(f"Обработано: {folder_path}")


if __name__ == "__main__":
    main()