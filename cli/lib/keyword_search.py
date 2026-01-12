from .keyword_search_utils import (
    load_movies,
    load_stop_words,
    MAX_RESULTS,
    CACHE_DIR
)
from nltk.stem import PorterStemmer # for stemming
from collections import defaultdict
import string
import pickle
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
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

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

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
    
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
    
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