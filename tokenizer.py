import os
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('snowball_data')
nltk.download('perluniprops')
nltk.download('universal_tagset')
nltk.download('nonbreaking_prefixes')
nltk.download('wordnet')
nltk.download('punkt_tab')

PAGES_DIR = "pages"

stop_words = set(stopwords.words('russian'))
morph = MorphAnalyzer()

def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def clean_token(token):
    if not token.isalpha():
        return False
    if re.fullmatch(r'[а-яА-Я]+', token):
        return True
    if re.fullmatch(r'[a-zA-Z]+', token):
        return False
    return False

def tokenize_text(text):
    tokens = word_tokenize(text.lower(), language='russian')
    cleaned_tokens = set()
    for token in tokens:
        if clean_token(token) and token not in stop_words:
            cleaned_tokens.add(token)
    return sorted(cleaned_tokens)

def lemmatize_tokens(tokens):
    lemmatized = {}
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma == token:
            continue
        if lemma not in lemmatized:
            lemmatized[lemma] = set()
        lemmatized[lemma].add(token)
    return lemmatized

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            clean_text = clean_html(text)
            tokens = tokenize_text(clean_text)
            lemmatized_tokens = lemmatize_tokens(tokens)

            tokens_file_path = os.path.join(folder_path, "tokens.txt")
            with open(tokens_file_path, 'w', encoding='utf-8') as file:
                file.writelines([f"{token}\n" for token in tokens])

            lemmatized_file_path = os.path.join(folder_path, "lemmatized_tokens.txt")
            with open(lemmatized_file_path, 'w', encoding='utf-8') as file:
                for lemma, tokens_set in sorted(lemmatized_tokens.items()):
                    file.write(f"{lemma} {' '.join(sorted(tokens_set))}\n")

            print(f"Обработано: {folder_path}")

def main():
    if not os.path.exists(PAGES_DIR):
        print(f"Папка {PAGES_DIR} не существует.")
        return

    for folder_name in os.listdir(PAGES_DIR):
        folder_path = os.path.join(PAGES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        process_folder(folder_path)

if __name__ == "__main__":
    main()