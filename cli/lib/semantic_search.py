from .semantic_search_utils import (
    CACHE_DIR,
    cosine_similarity
)
from .keyword_search_utils import (
    load_movies
)
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
                "document": doc["description"][:100],
                "score": round(score, 3)
            }
            formatted_results.append(formatted_result)
        return formatted_results
        
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = self.chunk_metadata = None

        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = []
        all_chunks_metadata: list[dict[int, list[str]]] = []
        for d_idx, d in enumerate(documents):
            self.document_map[d["id"]] = d
            if not d["description"].strip():
                continue
            d_chunks: list[list[str]] = chunk_semantic_query(d["description"], 4, 1)
            for d_chunk_idx, d_chunk in enumerate(d_chunks):
                all_chunks.append(d_chunk)

                # 3 ways to later get the corresponding chunk of a document
                all_chunks_metadata.append({
                    "movie_idx": d_idx,
                    "chunk_idx": d_chunk_idx,
                    "total_chunks": len(all_chunks)
                })

        self.chunk_metadata = all_chunks_metadata

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        np.save(file=self.chunk_embeddings_path, allow_pickle=True, arr=self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for d in documents:
            self.document_map[d["id"]] = d
        if os.path.isfile(self.chunk_embeddings_path) and os.path.isfile(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int=5):
        if len(self.chunk_embeddings) == 0:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        results = []

        for chunk_metadata in self.chunk_metadata:
            chunk_embedding = self.chunk_embeddings[chunk_metadata["total_chunks"] - 1]
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            results.append({
                "chunk_idx": chunk_metadata["chunk_idx"],
                "movie_idx": chunk_metadata["movie_idx"],
                "score": similarity_score
            })

        # because a document has chunks and some chunks might score higher than others, we find the highest chunk per document

        scores_dict: dict[int, int] = {}
        for res in results:
            scores_dict[res["movie_idx"]] = max(scores_dict.get(res["movie_idx"], 0), res["score"])
        
        sorted_results = sorted(scores_dict.items(), key=lambda item:item[1], reverse=True)[:limit]
        formatted_results = []
        for movie_idx, score in sorted_results:
            # recall that movie_idx is the INDEX from ENUMERATION, NOT the movie's attribute id
            # we don't care about returning the individual chunk specifically but rather the entire movie (obviously)
            
            movie = self.documents[movie_idx]
            formatted_result = {
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
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

def chunk_query(query: str, chunk_size: int=200, overlap=0):
    words = query.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range (0, len(words), chunk_size)]
    if not overlap or len(chunks) == 1:
        return chunks
    res = [chunks[0]]
    if overlap == chunk_size and len(chunks) > 1:
        res = []
    for i in range(1, len(chunks)):
        fixed_overlap = min(overlap, len(chunks[i-1]))
        prev_chunk = chunks[i-1][-fixed_overlap:]
        res.append(prev_chunk + chunks[i])
    return res

def chunk_semantic_query(query: str, chunk_size: int=4, overlap=0):
    query = query.strip()
    if not query:
        return []
    # matches any space after punctuation and uses it to split
    sentences = re.split(r"(?<=[.!?])\s+", query)
    chunks = [" ".join(sentences[i:i+chunk_size]).strip()
              for i in range (0, len(sentences), chunk_size)
              if "".join(sentences[i:i+chunk_size]).strip()]
    if not overlap or len(chunks) == 1:
        return chunks
    res = [chunks[0]]
    if overlap == chunk_size and len(chunks) > 1:
        res = []
    for i in range(1, len(chunks)):
        fixed_overlap = min(overlap, len(chunks[i-1]))
        prev_chunk = chunks[i-1][-fixed_overlap:]
        res.append(prev_chunk + chunks[i])
    return res


##### Chunks #####

def embed_chunks():
    chunk_search = ChunkedSemanticSearch()
    movies = load_movies()
    return chunk_search.load_or_create_chunk_embeddings(movies)

def search_chunked(query: str, limit: int=5):
    chunk_search = ChunkedSemanticSearch()
    movies = load_movies()
    chunk_search.load_or_create_chunk_embeddings(movies)
    return chunk_search.search_chunks(query, limit)