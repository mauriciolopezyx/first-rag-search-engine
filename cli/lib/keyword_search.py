from .keyword_search_utils import (
    load_movies,
    load_stop_words,
    MAX_RESULTS,
    CACHE_DIR,
    BM25_K1,
    BM25_B
)
from nltk.stem import PorterStemmer # for stemming
from collections import defaultdict, Counter
import string
import pickle
import math
import os

def search(query: str):
    movies = load_movies()
    results = []
    tokenized_query = tokenize_text(query)

    for movie in movies:
        tokenized_title = tokenize_text(movie["title"])
        if has_matching_token(tokenized_query, tokenized_title):
            results.append(movie)
            if len(results) >= MAX_RESULTS:
                break
    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    for title_token in title_tokens:
        for query_token in query_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str):
    translator = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(translator)
    return text

def tokenize_text(text: str):
    cleaned_text = preprocess_text(text)
    tokens = cleaned_text.split()
    stop_words = load_stop_words()
    valid_tokens = []
    stemmer = PorterStemmer()
    for token in tokens:
        if not token or token in stop_words:
            continue
        valid_tokens.append(stemmer.stem(token))
    return valid_tokens

######

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
    
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total = sum(self.doc_lengths.values())
        return total / len(self.doc_lengths)

    def get_tf(self, doc_id, term) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Expected only one token, got != 1")
        return self.term_frequencies[int(doc_id)][tokens[0]]
    
    def get_idf(self, term) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Expected only one token, got != 1")
        term_doc_count = len(self.index[tokens[0]])
        doc_count = len(self.docmap)
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Expected only one token, got != 1")
        term_doc_count = len(self.index[tokens[0]])
        doc_count = len(self.docmap)
        return math.log( ((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5)) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25_search(self, term: str, k1: float, b: float, limit: int=5) -> float:
        results: dict[int, int] = {}
        tokens = tokenize_text(term)
        for token in tokens:
            bm25_idf = self.get_bm25_idf(token)
            for doc_id in self.docmap.keys():
                bm25_tf = self.get_bm25_tf(doc_id, token, k1, b)
                bm25 = bm25_idf * bm25_tf
                results[doc_id] = results.get(doc_id, 0) + bm25
        
        sorted_results = sorted(results.items(), key=lambda item:item[1], reverse=True)[:limit]
        formatted_results = []
        for doc_id, score in sorted_results:
            doc = self.docmap[doc_id]
            formatted_result = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"],
                "score": round(score, 3)
            }
            formatted_results.append(formatted_result)
        return formatted_results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int=MAX_RESULTS) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term) * idx.get_idf(term)

# better version of idf called bm25-idf (no negatives, no NaN, no excessively large #s)
def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

# tf, but with additional occurrences of term resulting in diminishing returns (tunable parameter = k1)
# we can also normalize document lengths (tunable parameter = b, i.e how much we care about document length)
def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_command(term: str, k1: float=BM25_K1, b: float=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(term, k1, b)