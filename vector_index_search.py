import json
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VectorSearchEngine:
    def __init__(self, index_path="inverted_index.json"):
        self.index_path = index_path
        self.doc_to_tokens = {}
        self.vectorizer = TfidfVectorizer()
        self.doc_names = []

    def load_inverted_index(self):
        with open(self.index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_doc_token_map(self, inverted_index):
        doc_to_tokens = defaultdict(list)
        for term, docs in inverted_index.items():
            for doc in docs:
                doc_to_tokens[doc].append(term)
        return doc_to_tokens

    def build_index(self):
        raw_index = self.load_inverted_index()
        self.doc_to_tokens = self.build_doc_token_map(raw_index)
        self.doc_names = sorted(self.doc_to_tokens.keys())
        docs_as_strings = [" ".join(self.doc_to_tokens[doc]) for doc in self.doc_names]
        self.tfidf_matrix = self.vectorizer.fit_transform(docs_as_strings)

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_names[i], similarities[i]) for i in top_indices if similarities[i] > 0]


if __name__ == "__main__":
    engine = VectorSearchEngine()
    engine.build_index()

    print("Векторный поиск по построенному индексу. Введите запрос (пустая строка — выход):")
    while True:
        query = input("Запрос: ").strip()
        if not query:
            break
        results = engine.search(query)
        if results:
            print("Результаты:")
            for doc, score in results:
                print(f"{doc}: {score:.4f}")
        else:
            print("Совпадений не найдено.")
