from flask import Flask, render_template, request
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion
import numpy as np

app = Flask(__name__)

class VectorSearchEngine:
    def __init__(self, index_path="inverted_index.json"):
        self.index_path = index_path
        self.doc_to_tokens = {}
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)))
        ])
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

    def get_snippet(self, doc_tokens, query_terms):
        tokens = doc_tokens.split()
        for tok in tokens:
            if any(q_term.lower() in tok.lower() for q_term in query_terms):
                return tok
        return ''

    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        query_terms = query.split()
        for i in top_idx:
            score = float(sims[i])
            if score > 0:
                doc = self.doc_names[i]
                tokens_str = " ".join(self.doc_to_tokens[doc])
                snippet = self.get_snippet(tokens_str, query_terms)
                results.append((doc, score, snippet))
        return results

engine = VectorSearchEngine()
engine.build_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ''
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results = engine.search(query)
    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)