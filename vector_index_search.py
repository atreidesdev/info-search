import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion


class VectorSearchEngine:
    def __init__(self, pages_dir="pages"):
        self.pages_dir = pages_dir
        self.doc_names = []
        self.doc_texts = []
        self.doc_to_tokens = {}
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)))
        ])
        self.tfidf_matrix = None

    def load_terms_from_pages(self):
        doc_to_tokens = {}
        for folder_name in os.listdir(self.pages_dir):
            folder_path = os.path.join(self.pages_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            tfidf_file = os.path.join(folder_path, "terms_tf_idf.json")
            if not os.path.exists(tfidf_file):
                continue

            with open(tfidf_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tokens = list(data.keys())
                doc_to_tokens[folder_name] = tokens

        return doc_to_tokens

    def build_index(self):
        self.doc_to_tokens = self.load_terms_from_pages()
        self.doc_names = sorted(self.doc_to_tokens.keys())
        self.doc_texts = [" ".join(self.doc_to_tokens[doc]) for doc in self.doc_names]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def get_snippet(self, doc_tokens, query_terms):
        tokens = doc_tokens.split()
        for tok in tokens:
            if any(qt.lower() in tok.lower() for qt in query_terms):
                return tok
        return ""

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        query_terms = query.lower().split()
        for i in top_idx:
            score = float(sims[i])
            if score > 0:
                doc = self.doc_names[i]
                snippet = self.get_snippet(self.doc_texts[i], query_terms)
                if snippet:
                    results.append((doc, score, snippet))
        return results



if __name__ == "__main__":
    engine = VectorSearchEngine()
    engine.build_index()

    print("Векторный поиск. Введите запрос (пустая строка — выход):")
    while True:
        query = input("Запрос: ").strip()
        if not query:
            break
        results = engine.search(query)
        if results:
            print("Результаты:")
            for doc, score, snippet in results:
                print(f"  {doc}: {score:.4f} | Фрагмент: ...{snippet}...")
        else:
            print("Совпадений не найдено.")
