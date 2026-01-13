from .semantic_search_utils import (
    CACHE_DIR,
    cosine_similarity
)
from .keyword_search_utils import (
    load_movies
)
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.documents = None
        self.document_map = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Error: Must pass a non-empty text string")
        results = self.model.encode([text])
        return results[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_list = []
        for d in documents:
            self.document_map[d["id"]] = d
            doc_list.append(f"{d["title"]}: {d["description"]}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save(file=self.embeddings_path, allow_pickle=True, arr=self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for d in documents:
            self.document_map[d["id"]] = d
        if os.path.isfile(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int=5):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        results = []
        for i, document_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, document_embedding)
            results.append((similarity_score, self.documents[i])) # since the order is the same
        sorted_results = sorted(results, key=lambda item:item[0], reverse=True)[:limit]
        formatted_results = []
        for score, doc in sorted_results:
            formatted_result = {
                "title": doc["title"],
                "description": doc["description"],
                "score": round(score, 3)
            }
            formatted_results.append(formatted_result)
        return formatted_results
        
        
def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_query(query: str) -> float:
    search = SemanticSearch()
    return search.generate_embedding(query)

def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    return search.load_or_create_embeddings(movies)

def search(query: str, limit: int=5):
    search = SemanticSearch()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    return search.search(query, limit)